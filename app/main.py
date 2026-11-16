import asyncio
import hashlib
import importlib.util
import json
import os
import shutil
import uuid
import wave
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.database import Database
from app.segmentation import segment_script
from app.workers import WorkerError, WorkerRunner

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("AVATAR_DATA_DIR", BASE_DIR / "data")).resolve()
ALLOWED_SUFFIXES = {
    "image": {".png", ".jpg", ".jpeg", ".webp"},
    "audio": {".wav", ".mp3", ".m4a", ".webm", ".ogg"},
    "video": {".mp4", ".webm"},
}
UPLOAD_LIMITS = {"image": 20 * 1024 * 1024, "audio": 50 * 1024 * 1024, "video": 100 * 1024 * 1024}
TERMINAL_STATES = {"completed", "failed", "cancelled"}
VOICE_WORKER_STAGES = {"voice_clone", "tts"}
VIDEO_WORKER_STAGES = {"lip_sync", "qc", "post"}
GPU_WORKER_STAGES = {"voice_clone", "tts", "lip_sync"}


class CreateJobRequest(BaseModel):
    script: Annotated[str, Field(min_length=1, max_length=20_000)]
    language: Annotated[str, Field(min_length=2, max_length=16)] = "en"
    style: Literal["calm", "natural", "energetic"] = "natural"
    avatar_id: str | None = None
    voice_id: str | None = None

    @field_validator("script")
    @classmethod
    def script_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Script must contain non-whitespace text")
        return value.strip()


def create_app(
    database_path: Path | None = None,
    data_dir: Path | None = None,
    worker_config_path: Path | None = None,
) -> FastAPI:
    app_data_dir = Path(
        data_dir or (database_path.parent if database_path else DATA_DIR)
    ).resolve()
    config_path = Path(
        worker_config_path
        or os.environ.get("AVATAR_WORKERS_CONFIG", app_data_dir / "workers.json")
    ).resolve()
    database = Database(database_path or app_data_dir / "jobs.sqlite3")
    worker_runner = WorkerRunner(config_path)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        database.initialize()
        interrupted = {
            "queued_voice",
            "voice_generation",
            "queued_video",
            "motion",
            "lip_sync",
            "checking",
            "queued_finish",
            "finishing",
        }
        with database.connect() as connection:
            placeholders = ", ".join("?" for _ in interrupted)
            rows = connection.execute(
                f"SELECT id FROM jobs WHERE status IN ({placeholders})",
                tuple(interrupted),
            ).fetchall()
            for row in rows:
                now = datetime.now(UTC).isoformat()
                connection.execute(
                    """
                    UPDATE jobs SET status = 'failed',
                        error = 'App restarted during this job. Retry to resume completed segments.',
                        updated_at = ? WHERE id = ?
                    """,
                    (now, row["id"]),
                )
                connection.execute(
                    """
                    INSERT INTO job_events (job_id, created_at, event, detail)
                    VALUES (?, ?, 'failed', 'App restarted; completed segments are cached for retry.')
                    """,
                    (row["id"], now),
                )
            connection.execute(
                """
                UPDATE voice_profiles SET status = 'failed',
                    error = 'App restarted during voice cloning. Retry to resume.',
                    updated_at = ?
                WHERE status IN ('queued', 'cloning')
                """,
                (datetime.now(UTC).isoformat(),),
            )
        yield

    application = FastAPI(
        title="Local Avatar Video Agent",
        version="0.2.0",
        lifespan=lifespan,
    )
    application.state.database = database
    application.state.worker_runner = worker_runner

    def record_event(job_id: str, event: str, detail: str) -> None:
        now = datetime.now(UTC).isoformat()
        with database.connect() as connection:
            connection.execute(
                "INSERT INTO job_events (job_id, created_at, event, detail) VALUES (?, ?, ?, ?)",
                (job_id, now, event, detail),
            )
            connection.execute(
                "UPDATE jobs SET updated_at = ? WHERE id = ?", (now, job_id)
            )

    def set_job_status(job_id: str, status: str, detail: str) -> None:
        now = datetime.now(UTC).isoformat()
        with database.connect() as connection:
            connection.execute(
                "UPDATE jobs SET status = ?, updated_at = ?, error = NULL WHERE id = ?",
                (status, now, job_id),
            )
            connection.execute(
                "INSERT INTO job_events (job_id, created_at, event, detail) VALUES (?, ?, ?, ?)",
                (job_id, now, status, detail),
            )

    def require_worker_stages(stages: set[str]) -> None:
        missing = sorted(stages - worker_runner.configured_stages())
        if missing:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Required local worker stage(s) are not configured: {', '.join(missing)}. "
                    f"Configure them in {config_path}."
                ),
            )

    def ensure_not_cancelled(job_id: str) -> None:
        with database.connect() as connection:
            row = connection.execute(
                "SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        if row is None or row["cancel_requested"]:
            raise JobCancelled()

    def fail_job(job_id: str, error: Exception) -> None:
        with database.connect() as connection:
            row = connection.execute(
                "SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return
            status = "cancelled" if row["cancel_requested"] else "failed"
            detail = "Job cancelled." if status == "cancelled" else str(error)
            now = datetime.now(UTC).isoformat()
            connection.execute(
                "UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                (status, None if status == "cancelled" else detail, now, job_id),
            )
            connection.execute(
                "INSERT INTO job_events (job_id, created_at, event, detail) VALUES (?, ?, ?, ?)",
                (job_id, now, status, detail),
            )

    def run_voice_clone(voice_id: str) -> None:
        with database.connect() as connection:
            profile = connection.execute(
                "SELECT * FROM voice_profiles WHERE id = ?", (voice_id,)
            ).fetchone()
        if profile is None:
            return

        profile_dir = app_data_dir / "voices" / voice_id
        artifact_path = profile_dir / "voice-profile.json"
        profile_dir.mkdir(parents=True, exist_ok=True)
        try:
            with database.connect() as connection:
                connection.execute(
                    """
                    UPDATE voice_profiles SET status = 'cloning', error = NULL,
                        updated_at = ? WHERE id = ?
                    """,
                    (datetime.now(UTC).isoformat(), voice_id),
                )
            worker_runner.run(
                "voice_clone",
                {
                    "voice_id": voice_id,
                    "voice_name": profile["name"],
                    "voice_sample": profile["sample_path"],
                    "output_profile": str(artifact_path.resolve()),
                    "work_dir": str(profile_dir.resolve()),
                },
            )
            if not artifact_path.is_file() or artifact_path.stat().st_size == 0:
                raise WorkerError("Voice-cloning worker did not create a profile artifact.")
            try:
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise WorkerError("Voice profile artifact must contain valid JSON.") from error
            if (
                not isinstance(artifact, dict)
                or not isinstance(artifact.get("format"), str)
                or not artifact["format"].strip()
                or not isinstance(artifact.get("profile"), dict)
                or not artifact["profile"]
            ):
                raise WorkerError(
                    "Voice profile artifact must include a non-empty format string and profile object."
                )
            with database.connect() as connection:
                connection.execute(
                    """
                    UPDATE voice_profiles SET status = 'ready', profile_path = ?,
                        error = NULL, updated_at = ? WHERE id = ?
                    """,
                    (
                        str(artifact_path.resolve()),
                        datetime.now(UTC).isoformat(),
                        voice_id,
                    ),
                )
        except Exception as error:
            with database.connect() as connection:
                connection.execute(
                    """
                    UPDATE voice_profiles SET status = 'failed', error = ?,
                        updated_at = ? WHERE id = ?
                    """,
                    (str(error), datetime.now(UTC).isoformat(), voice_id),
                )

    def run_voice_job(job_id: str) -> None:
        try:
            with database.connect() as connection:
                job = connection.execute(
                    """
                    SELECT jobs.*, voice_profiles.sample_path, voice_profiles.profile_path,
                           voice_profiles.status AS voice_status
                    FROM jobs LEFT JOIN voice_profiles ON jobs.voice_id = voice_profiles.id
                    WHERE jobs.id = ?
                    """,
                    (job_id,),
                ).fetchone()
                segments = connection.execute(
                    "SELECT * FROM segments WHERE job_id = ? ORDER BY position", (job_id,)
                ).fetchall()
            if job is None:
                return
            if not job["voice_id"] or job["voice_status"] != "ready":
                raise WorkerError("Choose a successfully cloned voice profile before generating speech.")
            if not job["profile_path"] or not Path(job["profile_path"]).is_file():
                raise WorkerError("The selected voice clone artifact is missing.")

            set_job_status(
                job_id,
                "voice_generation",
                f"Generating speech with voice profile '{job['voice_id']}'.",
            )
            job_dir = app_data_dir / "jobs" / job_id
            for segment in segments:
                ensure_not_cancelled(job_id)
                if segment["status"] in {"ready_for_review", "approved"} and _valid_wave_file(
                    segment["audio_path"]
                ):
                    record_event(
                        job_id,
                        "segment_cached",
                        f"Reusing voice segment {segment['position'] + 1}.",
                    )
                    continue
                output_audio = (
                    job_dir
                    / "audio"
                    / f"{segment['position']:04d}-{uuid.uuid4().hex}.wav"
                )
                output_audio.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "job_id": job_id,
                    "segment_id": segment["id"],
                    "position": segment["position"],
                    "script": segment["text"],
                    "language": job["language"],
                    "style": job["style"],
                    "voice_sample": job["sample_path"],
                    "voice_profile": job["profile_path"],
                    "output_audio": str(output_audio.resolve()),
                    "work_dir": str(job_dir.resolve()),
                }
                with database.connect() as connection:
                    connection.execute(
                        """
                        UPDATE segments SET status = 'generating', attempts = attempts + 1,
                            error = NULL WHERE id = ?
                        """,
                        (segment["id"],),
                    )
                try:
                    worker_runner.run("tts", payload)
                    if not output_audio.is_file() or output_audio.stat().st_size == 0:
                        raise WorkerError(
                            f"TTS worker did not create a non-empty audio file for segment {segment['position'] + 1}."
                        )
                    with wave.open(str(output_audio), "rb") as audio:
                        if audio.getnframes() == 0:
                            raise WorkerError("TTS worker produced an empty WAV.")
                except (wave.Error, EOFError) as error:
                    validation_error = WorkerError(
                        f"TTS worker output for segment {segment['position'] + 1} is not a valid WAV file."
                    )
                    with database.connect() as connection:
                        connection.execute(
                            "UPDATE segments SET status = 'failed', error = ? WHERE id = ?",
                            (str(validation_error), segment["id"]),
                        )
                    raise validation_error from error
                except Exception as error:
                    with database.connect() as connection:
                        connection.execute(
                            "UPDATE segments SET status = 'failed', error = ? WHERE id = ?",
                            (str(error), segment["id"]),
                        )
                    raise
                with database.connect() as connection:
                    connection.execute(
                        """
                        UPDATE segments SET status = 'ready_for_review', audio_path = ?
                        WHERE id = ?
                        """,
                        (str(output_audio.resolve()), segment["id"]),
                    )
                record_event(
                    job_id,
                    "segment_ready",
                    f"Voice segment {segment['position'] + 1} is ready for review.",
                )
            set_job_status(job_id, "voice_review", "Review every generated segment.")
        except JobCancelled:
            fail_job(job_id, JobCancelled())
        except Exception as error:
            fail_job(job_id, error)

    def run_video_job(job_id: str) -> None:
        try:
            with database.connect() as connection:
                job = connection.execute(
                    """
                    SELECT jobs.*, avatars.image_path,
                           voice_profiles.name AS voice_name
                    FROM jobs JOIN avatars ON jobs.avatar_id = avatars.id
                    JOIN voice_profiles ON jobs.voice_id = voice_profiles.id
                    WHERE jobs.id = ?
                    """,
                    (job_id,),
                ).fetchone()
                segments = connection.execute(
                    "SELECT * FROM segments WHERE job_id = ? ORDER BY position", (job_id,)
                ).fetchall()
            if job is None:
                raise WorkerError("Job avatar was deleted.")
            if not Path(job["image_path"]).is_file():
                raise WorkerError("The selected avatar reference photo is unavailable.")
            job_dir = app_data_dir / "jobs" / job_id
            stitched_inputs: list[str] = []
            set_job_status(job_id, "lip_sync", "Animating the portrait mouth to match speech.")
            for segment in segments:
                ensure_not_cancelled(job_id)
                if segment["video_path"] and Path(segment["video_path"]).is_file():
                    if _is_mp4_file(Path(segment["video_path"])):
                        stitched_inputs.append(segment["video_path"])
                        record_event(
                            job_id,
                            "segment_cached",
                            f"Reusing finished video segment {segment['position'] + 1}.",
                        )
                        continue
                attempt_id = uuid.uuid4().hex
                output_video = (
                    job_dir / "segments" / f"{segment['position']:04d}-{attempt_id}.mp4"
                )
                output_video.parent.mkdir(parents=True, exist_ok=True)
                with database.connect() as connection:
                    connection.execute(
                        "UPDATE segments SET status = 'lip_sync', error = NULL WHERE id = ?",
                        (segment["id"],),
                    )
                payload = {
                    "job_id": job_id,
                    "segment_id": segment["id"],
                    "position": segment["position"],
                    "script": segment["text"],
                    "reference_image": job["image_path"],
                    "input_audio": segment["audio_path"],
                    "output_video": str(output_video.resolve()),
                    "work_dir": str(job_dir.resolve()),
                }
                try:
                    worker_runner.run("lip_sync", payload)
                    if not _is_mp4_file(output_video):
                        raise WorkerError(
                            f"Lip-sync worker did not create a valid MP4 for segment {segment['position'] + 1}."
                        )
                except Exception as error:
                    with database.connect() as connection:
                        connection.execute(
                            "UPDATE segments SET status = 'failed', error = ? WHERE id = ?",
                            (str(error), segment["id"]),
                        )
                    raise
                stitched_inputs.append(str(output_video.resolve()))
                with database.connect() as connection:
                    connection.execute(
                        "UPDATE segments SET status = 'video_ready', video_path = ? WHERE id = ?",
                        (str(output_video.resolve()), segment["id"]),
                    )

            ensure_not_cancelled(job_id)
            set_job_status(job_id, "checking", "Running configured quality checks.")
            qc_report = job_dir / "quality-report.json"
            payload = {
                "job_id": job_id,
                "segments": stitched_inputs,
                "reference_image": job["image_path"],
                "qc_report": str(qc_report.resolve()),
                "work_dir": str(job_dir.resolve()),
            }
            worker_runner.run("qc", payload)
            if not qc_report.is_file():
                raise WorkerError("QC worker did not produce its required quality-report.json.")
            try:
                report = json.loads(qc_report.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise WorkerError("QC report is unreadable JSON.") from error
            if not isinstance(report, dict) or report.get("status") not in {
                "pass",
                "review_needed",
            }:
                raise WorkerError("QC report status must be 'pass' or 'review_needed'.")
            if not isinstance(report.get("flags", []), list) or not all(
                isinstance(flag, str) for flag in report.get("flags", [])
            ):
                raise WorkerError("QC report flags must be an array of strings.")
            if report["status"] == "review_needed":
                set_job_status(
                    job_id,
                    "video_review",
                    "Review the avatar quality flags before finishing the video.",
                )
                return
            run_post_job(job_id)
        except JobCancelled:
            fail_job(job_id, JobCancelled())
        except Exception as error:
            fail_job(job_id, error)

    def run_post_job(job_id: str) -> None:
        try:
            with database.connect() as connection:
                job = connection.execute(
                    """
                    SELECT jobs.*, avatars.image_path, voice_profiles.name AS voice_name
                    FROM jobs
                    JOIN avatars ON jobs.avatar_id = avatars.id
                    JOIN voice_profiles ON jobs.voice_id = voice_profiles.id
                    WHERE jobs.id = ?
                    """,
                    (job_id,),
                ).fetchone()
                segments = connection.execute(
                    "SELECT video_path FROM segments WHERE job_id = ? ORDER BY position",
                    (job_id,),
                ).fetchall()
            if job is None:
                raise WorkerError("Job or its avatar/voice profile no longer exists.")
            if not segments or any(
                not segment["video_path"]
                or not _is_mp4_file(Path(segment["video_path"]))
                for segment in segments
            ):
                raise WorkerError("A valid video segment is missing before post-processing.")

            job_dir = app_data_dir / "jobs" / job_id
            qc_report = job_dir / "quality-report.json"
            try:
                report = json.loads(qc_report.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise WorkerError("QC report is unavailable or unreadable.") from error
            if not isinstance(report, dict) or report.get("status") not in {
                "pass",
                "review_needed",
            }:
                raise WorkerError("QC report status must be 'pass' or 'review_needed'.")
            flags = report.get("flags", [])
            if not isinstance(flags, list) or not all(isinstance(flag, str) for flag in flags):
                raise WorkerError("QC report flags must be an array of strings.")

            ensure_not_cancelled(job_id)
            final_video = job_dir / f"final-{uuid.uuid4().hex}.mp4"
            payload = {
                "job_id": job_id,
                "segments": [segment["video_path"] for segment in segments],
                "reference_image": job["image_path"],
                "output_video": str(final_video.resolve()),
                "qc_report": str(qc_report.resolve()),
                "work_dir": str(job_dir.resolve()),
            }
            set_job_status(job_id, "finishing", "Stitching and finishing the video.")
            worker_runner.run("post", payload)
            if not final_video.is_file() or final_video.stat().st_size == 0:
                raise WorkerError("Post-processing worker did not create final.mp4.")
            if not _is_mp4_file(final_video):
                raise WorkerError("Post-processing output does not have an MP4 file signature.")
            provenance = {
                "generated_by": "Local Avatar Video Agent",
                "ai_generated": True,
                "job_id": job_id,
                "avatar_id": job["avatar_id"],
                "voice_id": job["voice_id"],
                "voice_name": job["voice_name"],
                "created_at": datetime.now(UTC).isoformat(),
                "script_sha256": hashlib.sha256(job["script"].encode("utf-8")).hexdigest(),
                "quality_status": report["status"],
                "quality_flags": flags,
            }
            (job_dir / "provenance.json").write_text(
                json.dumps(provenance, indent=2), encoding="utf-8"
            )
            with database.connect() as connection:
                connection.execute(
                    "UPDATE jobs SET video_path = ? WHERE id = ?",
                    (str(final_video.resolve()), job_id),
                )
            set_job_status(job_id, "completed", "Video is ready.")
        except JobCancelled:
            fail_job(job_id, JobCancelled())
        except Exception as error:
            fail_job(job_id, error)

    @application.get("/", include_in_schema=False)
    def studio():
        return FileResponse(BASE_DIR / "static" / "index.html")

    @application.get("/api/health")
    def health():
        return {"status": "ok"}

    @application.get("/api/system")
    def system_status():
        configured = worker_runner.configured_stages()
        cuda_available = False
        if importlib.util.find_spec("torch") is None:
            torch_status = {
                "torch_installed": False,
                "cuda_available": False,
                "message": "PyTorch is not installed; GPU inference has not been configured.",
            }
        else:
            import torch

            cuda_available = torch.cuda.is_available()
            torch_status = {
                "torch_installed": True,
                "torch_version": torch.__version__,
                "cuda_available": cuda_available,
                "cuda_runtime": torch.version.cuda,
            }
            if cuda_available:
                torch_status["gpu_name"] = torch.cuda.get_device_name(0)
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                torch_status["vram_total_bytes"] = total_bytes
                torch_status["vram_free_bytes"] = free_bytes
            else:
                torch_status["message"] = "The installed PyTorch build cannot access CUDA."
        return {
            **torch_status,
            "configured_workers": sorted(configured),
            "missing_workers": sorted((VOICE_WORKER_STAGES | VIDEO_WORKER_STAGES) - configured),
            "worker_config_path": str(config_path),
            "voice_pipeline_configured": VOICE_WORKER_STAGES <= configured,
            "video_pipeline_configured": VIDEO_WORKER_STAGES <= configured,
            "full_pipeline_configured": (VOICE_WORKER_STAGES | VIDEO_WORKER_STAGES)
            <= configured,
            "gpu_worker_stages_configured": GPU_WORKER_STAGES <= configured,
        }

    @application.post("/api/avatars", status_code=201)
    async def create_avatar(
        name: Annotated[str, Form(min_length=1, max_length=80)],
        consent_text: Annotated[str, Form(min_length=20, max_length=2000)],
        consent_confirmed: Annotated[bool, Form()],
        reference_image: Annotated[UploadFile, File()],
        consent_recording: Annotated[UploadFile, File()],
        driving_video: Annotated[UploadFile | None, File()] = None,
    ):
        if not consent_confirmed:
            raise HTTPException(status_code=400, detail="Explicit face consent is required.")
        clean_name = name.strip()
        clean_consent = consent_text.strip()
        if not clean_name:
            raise HTTPException(status_code=400, detail="Avatar name must not be blank.")
        if len(clean_consent) < 20:
            raise HTTPException(status_code=400, detail="Consent statement must contain at least 20 characters.")
        avatar_id = str(uuid.uuid4())
        avatar_dir = app_data_dir / "uploads" / "avatars" / avatar_id
        avatar_dir.mkdir(parents=True, exist_ok=False)
        uploads = [
            ("image", reference_image, "reference"),
            ("audio", consent_recording, "consent"),
        ]
        if driving_video is not None and driving_video.filename:
            uploads.append(("video", driving_video, "driving"))
        paths: dict[str, str | None] = {}
        try:
            for kind, upload, label in uploads:
                paths[label] = str(
                    await _save_upload(upload, kind, avatar_dir, label)
                )
        except (OSError, ValueError) as error:
            try:
                shutil.rmtree(avatar_dir)
            except OSError as cleanup_error:
                raise HTTPException(
                    status_code=500,
                    detail=f"Upload failed and partial avatar files could not be removed: {cleanup_error}",
                ) from error
            raise HTTPException(status_code=400, detail=str(error)) from error
        now = datetime.now(UTC).isoformat()
        with database.connect() as connection:
            connection.execute(
                """
                INSERT INTO avatars
                (id, name, image_path, voice_path, consent_path, consent_text,
                 driving_video_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    avatar_id,
                    clean_name,
                    paths["reference"],
                    paths.get("voice-sample"),
                    paths["consent"],
                    clean_consent,
                    paths.get("driving"),
                    now,
                ),
            )
        return get_avatar(avatar_id)

    @application.get("/api/avatars")
    def list_avatars():
        with database.connect() as connection:
            avatars = connection.execute(
                "SELECT id, name, created_at, driving_video_path FROM avatars ORDER BY created_at DESC"
            ).fetchall()
        return [
            {
                "id": avatar["id"],
                "name": avatar["name"],
                "created_at": avatar["created_at"],
                "has_driving_video": bool(
                    avatar["driving_video_path"]
                    and Path(avatar["driving_video_path"]).is_file()
                ),
                "image_url": f"/api/avatars/{avatar['id']}/image",
            }
            for avatar in avatars
        ]

    def get_avatar(avatar_id: str):
        with database.connect() as connection:
            avatar = connection.execute(
                """
                SELECT id, name, created_at, driving_video_path
                FROM avatars WHERE id = ?
                """,
                (avatar_id,),
            ).fetchone()
        if avatar is None:
            raise HTTPException(status_code=404, detail="Avatar not found")
        return {
            "id": avatar["id"],
            "name": avatar["name"],
            "created_at": avatar["created_at"],
            "has_driving_video": bool(
                avatar["driving_video_path"]
                and Path(avatar["driving_video_path"]).is_file()
            ),
            "image_url": f"/api/avatars/{avatar['id']}/image",
        }

    @application.post("/api/avatars/{avatar_id}/image")
    async def replace_avatar_image(
        avatar_id: str,
        reference_image: Annotated[UploadFile, File()],
    ):
        with database.connect() as connection:
            avatar = connection.execute(
                "SELECT id FROM avatars WHERE id = ?", (avatar_id,)
            ).fetchone()
        if avatar is None:
            raise HTTPException(status_code=404, detail="Avatar not found")

        avatar_dir = app_data_dir / "uploads" / "avatars" / avatar_id
        avatar_dir.mkdir(parents=True, exist_ok=True)
        try:
            path = await _save_upload(
                reference_image,
                "image",
                avatar_dir,
                f"reference-{uuid.uuid4().hex}",
            )
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        with database.connect() as connection:
            connection.execute(
                "UPDATE avatars SET image_path = ? WHERE id = ?",
                (str(path), avatar_id),
            )
        return get_avatar(avatar_id)

    @application.post("/api/avatars/{avatar_id}/driving-video")
    async def update_avatar_driving_video(
        avatar_id: str, driving_video: Annotated[UploadFile, File()]
    ):
        with database.connect() as connection:
            avatar = connection.execute(
                "SELECT id FROM avatars WHERE id = ?", (avatar_id,)
            ).fetchone()
        if avatar is None:
            raise HTTPException(status_code=404, detail="Avatar not found")

        avatar_dir = app_data_dir / "uploads" / "avatars" / avatar_id
        avatar_dir.mkdir(parents=True, exist_ok=True)
        try:
            path = await _save_upload(
                driving_video, "video", avatar_dir, f"driving-{uuid.uuid4().hex}"
            )
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        with database.connect() as connection:
            connection.execute(
                "UPDATE avatars SET driving_video_path = ? WHERE id = ?",
                (str(path), avatar_id),
            )
        return get_avatar(avatar_id)

    @application.get("/api/avatars/{avatar_id}/image")
    def avatar_image(avatar_id: str):
        with database.connect() as connection:
            avatar = connection.execute(
                "SELECT image_path FROM avatars WHERE id = ?", (avatar_id,)
            ).fetchone()
        if avatar is None or not Path(avatar["image_path"]).is_file():
            raise HTTPException(status_code=404, detail="Avatar image not found")
        return FileResponse(avatar["image_path"])

    @application.get("/api/voices")
    def list_voice_profiles():
        with database.connect() as connection:
            profiles = connection.execute(
                """
                SELECT id, name, status, error, created_at, updated_at
                FROM voice_profiles ORDER BY created_at DESC
                """
            ).fetchall()
        return [dict(profile) for profile in profiles]

    @application.post("/api/voices", status_code=202)
    async def create_voice_profile(
        background_tasks: BackgroundTasks,
        name: Annotated[str, Form(min_length=1, max_length=80)],
        voice_sample: Annotated[UploadFile, File()],
    ):
        clean_name = name.strip()
        if not clean_name:
            raise HTTPException(status_code=400, detail="Voice name must not be blank.")
        if "voice_clone" not in worker_runner.configured_stages():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Voice cloning is not configured yet. Add a local voice_clone worker "
                    f"to {config_path} before creating a voice profile."
                ),
            )
        voice_id = str(uuid.uuid4())
        voice_dir = app_data_dir / "uploads" / "voices" / voice_id
        voice_dir.mkdir(parents=True, exist_ok=False)
        try:
            sample_path = await _save_upload(
                voice_sample, "audio", voice_dir, "voice-sample"
            )
        except (OSError, ValueError) as error:
            try:
                shutil.rmtree(voice_dir)
            except OSError as cleanup_error:
                raise HTTPException(
                    status_code=500,
                    detail=f"Upload failed and partial voice files could not be removed: {cleanup_error}",
                ) from error
            raise HTTPException(status_code=400, detail=str(error)) from error

        now = datetime.now(UTC).isoformat()
        with database.connect() as connection:
            connection.execute(
                """
                INSERT INTO voice_profiles
                (id, name, sample_path, consent_path, consent_text, status,
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'queued', ?, ?)
                """,
                (
                    voice_id,
                    clean_name,
                    str(sample_path),
                    "",
                    "",
                    now,
                    now,
                ),
            )
        background_tasks.add_task(run_voice_clone, voice_id)
        return get_voice_profile(voice_id)

    def get_voice_profile(voice_id: str):
        with database.connect() as connection:
            profile = connection.execute(
                """
                SELECT id, name, status, error, created_at, updated_at
                FROM voice_profiles WHERE id = ?
                """,
                (voice_id,),
            ).fetchone()
        if profile is None:
            raise HTTPException(status_code=404, detail="Voice profile not found")
        return dict(profile)

    @application.post("/api/voices/{voice_id}/retry", status_code=202)
    def retry_voice_clone(voice_id: str, background_tasks: BackgroundTasks):
        if "voice_clone" not in worker_runner.configured_stages():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Voice cloning is not configured yet. Add a local voice_clone worker "
                    f"to {config_path} before retrying."
                ),
            )
        with database.connect() as connection:
            profile = connection.execute(
                "SELECT status FROM voice_profiles WHERE id = ?", (voice_id,)
            ).fetchone()
            if profile is None:
                raise HTTPException(status_code=404, detail="Voice profile not found")
            if profile["status"] != "failed":
                raise HTTPException(
                    status_code=409, detail="Only failed voice profiles can be retried."
                )
            connection.execute(
                """
                UPDATE voice_profiles SET status = 'queued', error = NULL,
                    updated_at = ? WHERE id = ?
                """,
                (datetime.now(UTC).isoformat(), voice_id),
            )
        background_tasks.add_task(run_voice_clone, voice_id)
        return get_voice_profile(voice_id)

    @application.post("/api/jobs", status_code=201)
    def create_job(request: CreateJobRequest):
        if request.avatar_id:
            with database.connect() as connection:
                avatar = connection.execute(
                    "SELECT id FROM avatars WHERE id = ?", (request.avatar_id,)
                ).fetchone()
            if avatar is None:
                raise HTTPException(status_code=404, detail="Avatar not found")
        if request.voice_id:
            with database.connect() as connection:
                voice = connection.execute(
                    "SELECT status FROM voice_profiles WHERE id = ?",
                    (request.voice_id,),
                ).fetchone()
            if voice is None:
                raise HTTPException(status_code=404, detail="Voice profile not found")
            if voice["status"] != "ready":
                raise HTTPException(
                    status_code=409,
                    detail=f"Voice profile is {voice['status']}; only cloned profiles can be selected.",
                )
        segments = segment_script(request.script)
        now = datetime.now(UTC).isoformat()
        job_id = str(uuid.uuid4())
        with database.connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs
                (id, script, language, style, status, created_at, updated_at, avatar_id, voice_id)
                VALUES (?, ?, ?, ?, 'draft', ?, ?, ?, ?)
                """,
                (
                    job_id,
                    request.script,
                    request.language,
                    request.style,
                    now,
                    now,
                    request.avatar_id,
                    request.voice_id,
                ),
            )
            connection.executemany(
                """
                INSERT INTO segments (id, job_id, position, text, status)
                VALUES (?, ?, ?, ?, 'pending')
                """,
                [
                    (str(uuid.uuid4()), job_id, position, text)
                    for position, text in enumerate(segments)
                ],
            )
        return get_job(job_id)

    @application.get("/api/jobs")
    def list_jobs():
        with database.connect() as connection:
            rows = connection.execute(
                """
                SELECT jobs.id, jobs.script, jobs.language, jobs.style, jobs.status,
                       jobs.created_at, jobs.updated_at, jobs.error, avatars.name AS avatar_name,
                       voice_profiles.name AS voice_name
                FROM jobs LEFT JOIN avatars ON jobs.avatar_id = avatars.id
                LEFT JOIN voice_profiles ON jobs.voice_id = voice_profiles.id
                ORDER BY jobs.created_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    @application.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        with database.connect() as connection:
            job = connection.execute(
                """
                SELECT jobs.*, avatars.name AS avatar_name,
                       voice_profiles.name AS voice_name
                FROM jobs
                LEFT JOIN avatars ON jobs.avatar_id = avatars.id
                LEFT JOIN voice_profiles ON jobs.voice_id = voice_profiles.id
                WHERE jobs.id = ?
                """,
                (job_id,),
            ).fetchone()
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            segments = connection.execute(
                """
                SELECT id, position, text, status, approved, attempts, error,
                       CASE WHEN audio_path IS NOT NULL THEN
                           '/api/jobs/' || job_id || '/segments/' || id || '/audio'
                       ELSE NULL END AS audio_url,
                       CASE WHEN video_path IS NOT NULL THEN
                           '/api/jobs/' || job_id || '/segments/' || id || '/video'
                       ELSE NULL END AS video_url
                FROM segments WHERE job_id = ? ORDER BY position
                """,
                (job_id,),
            ).fetchall()
        result = {
            key: job[key]
            for key in (
                "id",
                "script",
                "language",
                "style",
                "status",
                "created_at",
                "updated_at",
                "avatar_id",
                "avatar_name",
                "voice_id",
                "voice_name",
                "error",
            )
        }
        result["quality_status"] = None
        result["quality_flags"] = []
        quality_report_path = app_data_dir / "jobs" / job_id / "quality-report.json"
        if quality_report_path.is_file():
            try:
                report = json.loads(quality_report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise HTTPException(
                    status_code=500, detail="Quality report is unreadable."
                ) from error
            if not isinstance(report, dict) or report.get("status") not in {
                "pass",
                "review_needed",
            }:
                raise HTTPException(status_code=500, detail="Quality report is invalid.")
            flags = report.get("flags", [])
            if not isinstance(flags, list) or not all(isinstance(flag, str) for flag in flags):
                raise HTTPException(status_code=500, detail="Quality report flags are invalid.")
            result["quality_status"] = report["status"]
            result["quality_flags"] = flags
        result["segments"] = [dict(row) for row in segments]
        result["video_url"] = f"/api/jobs/{job_id}/video" if job["video_path"] else None
        result["can_cancel"] = job["status"] not in TERMINAL_STATES
        return result

    @application.post("/api/jobs/{job_id}/generate-voice", status_code=202)
    def generate_voice(job_id: str, background_tasks: BackgroundTasks):
        with database.connect() as connection:
            job = connection.execute(
                "SELECT status, avatar_id, voice_id FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if not job["avatar_id"]:
            raise HTTPException(status_code=409, detail="Choose an avatar before generating voice.")
        if not job["voice_id"]:
            raise HTTPException(
                status_code=409, detail="Choose a successfully cloned voice profile before generating speech."
            )
        with database.connect() as connection:
            voice = connection.execute(
                "SELECT status FROM voice_profiles WHERE id = ?", (job["voice_id"],)
            ).fetchone()
        if voice is None or voice["status"] != "ready":
            raise HTTPException(
                status_code=409,
                detail="The selected voice clone is not ready for speech generation.",
            )
        if job["status"] not in {"draft", "failed"}:
            raise HTTPException(status_code=409, detail=f"Cannot generate voice for a {job['status']} job.")
        require_worker_stages({"tts"})
        with database.connect() as connection:
            connection.execute(
                """
                UPDATE jobs SET status = 'queued_voice', error = NULL,
                    cancel_requested = 0, updated_at = ? WHERE id = ?
                """,
                (datetime.now(UTC).isoformat(), job_id),
            )
            connection.execute(
                "UPDATE segments SET status = 'pending', error = NULL, approved = 0 WHERE job_id = ?",
                (job_id,),
            )
        record_event(job_id, "queued_voice", "Voice generation queued.")
        background_tasks.add_task(run_voice_job, job_id)
        return {"job_id": job_id, "status": "queued_voice"}

    @application.post("/api/jobs/{job_id}/retry", status_code=202)
    def retry_job(job_id: str, background_tasks: BackgroundTasks):
        with database.connect() as connection:
            job = connection.execute(
                "SELECT status, avatar_id, voice_id FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            segments = connection.execute(
                "SELECT audio_path, approved FROM segments WHERE job_id = ?",
                (job_id,),
            ).fetchall()
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] != "failed":
            raise HTTPException(status_code=409, detail="Only failed jobs can be retried.")
        if not job["voice_id"]:
            raise HTTPException(
                status_code=409, detail="Select a cloned voice profile before retrying this job."
            )
        with database.connect() as connection:
            voice = connection.execute(
                "SELECT status FROM voice_profiles WHERE id = ?", (job["voice_id"],)
            ).fetchone()
        if voice is None or voice["status"] != "ready":
            raise HTTPException(
                status_code=409, detail="The selected voice clone is no longer available."
            )
        if not segments:
            raise HTTPException(status_code=409, detail="Job has no script segments.")
        if all(
            segment["approved"] and _valid_wave_file(segment["audio_path"])
            for segment in segments
        ):
            require_worker_stages(VIDEO_WORKER_STAGES)
            with database.connect() as connection:
                avatar = connection.execute(
                    "SELECT image_path FROM avatars WHERE id = ?",
                    (job["avatar_id"],),
                ).fetchone()
                if avatar is None or not Path(avatar["image_path"]).is_file():
                    raise HTTPException(
                        status_code=409, detail="The avatar reference photo is unavailable."
                    )
                connection.execute(
                    "UPDATE jobs SET status = 'queued_video', error = NULL, cancel_requested = 0 WHERE id = ?",
                    (job_id,),
                )
            record_event(job_id, "queued_video", "Retrying video generation from cached voice.")
            background_tasks.add_task(run_video_job, job_id)
            return {"job_id": job_id, "status": "queued_video"}

        require_worker_stages({"tts"})
        with database.connect() as connection:
            connection.execute(
                "UPDATE jobs SET status = 'queued_voice', error = NULL, cancel_requested = 0 WHERE id = ?",
                (job_id,),
            )
        record_event(job_id, "queued_voice", "Retrying voice generation; completed audio segments will be reused.")
        background_tasks.add_task(run_voice_job, job_id)
        return {"job_id": job_id, "status": "queued_voice"}

    @application.post("/api/jobs/{job_id}/segments/{segment_id}/approve")
    def approve_segment(job_id: str, segment_id: str):
        with database.connect() as connection:
            job = connection.execute(
                "SELECT status FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            segment = connection.execute(
                "SELECT status FROM segments WHERE id = ? AND job_id = ?",
                (segment_id, job_id),
            ).fetchone()
            if job is None or segment is None:
                raise HTTPException(status_code=404, detail="Job segment not found")
            if job["status"] != "voice_review" or segment["status"] != "ready_for_review":
                raise HTTPException(status_code=409, detail="Segment is not waiting for voice review.")
            connection.execute(
                "UPDATE segments SET approved = 1, status = 'approved' WHERE id = ?",
                (segment_id,),
            )
        record_event(job_id, "segment_approved", "A voice segment was approved.")
        return get_job(job_id)

    @application.post("/api/jobs/{job_id}/segments/{segment_id}/regenerate", status_code=202)
    def regenerate_segment(job_id: str, segment_id: str, background_tasks: BackgroundTasks):
        with database.connect() as connection:
            job = connection.execute(
                "SELECT status FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            segment = connection.execute(
                "SELECT id, status FROM segments WHERE id = ? AND job_id = ?",
                (segment_id, job_id),
            ).fetchone()
            if job is None or segment is None:
                raise HTTPException(status_code=404, detail="Job segment not found")
            if job["status"] != "voice_review":
                raise HTTPException(status_code=409, detail="Job is not in voice review.")
            require_worker_stages({"tts"})
            connection.execute(
                """
                UPDATE segments SET status = 'pending', approved = 0, error = NULL
                WHERE id = ?
                """,
                (segment_id,),
            )
            connection.execute(
                "UPDATE jobs SET status = 'queued_voice', updated_at = ? WHERE id = ?",
                (datetime.now(UTC).isoformat(), job_id),
            )
        background_tasks.add_task(run_single_voice_segment, job_id, segment_id)
        return {"job_id": job_id, "segment_id": segment_id, "status": "queued_voice"}

    @application.post(
        "/api/jobs/{job_id}/segments/{segment_id}/regenerate-video",
        status_code=202,
    )
    def regenerate_video_segment(
        job_id: str, segment_id: str, background_tasks: BackgroundTasks
    ):
        with database.connect() as connection:
            job = connection.execute(
                "SELECT status FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            segment = connection.execute(
                """
                SELECT approved, audio_path FROM segments
                WHERE id = ? AND job_id = ?
                """,
                (segment_id, job_id),
            ).fetchone()
            if job is None or segment is None:
                raise HTTPException(status_code=404, detail="Job segment not found")
            if job["status"] != "completed":
                raise HTTPException(
                    status_code=409, detail="Only a completed video's segment can be regenerated."
                )
            if not segment["approved"] or not _valid_wave_file(segment["audio_path"]):
                raise HTTPException(
                    status_code=409, detail="Approved segment audio is not available."
                )
            require_worker_stages(VIDEO_WORKER_STAGES)
            now = datetime.now(UTC).isoformat()
            connection.execute(
                """
                UPDATE jobs SET status = 'queued_video', video_path = NULL, error = NULL,
                    cancel_requested = 0, updated_at = ? WHERE id = ?
                """,
                (now, job_id),
            )
            connection.execute(
                """
                UPDATE segments SET status = 'approved', video_path = NULL, error = NULL
                WHERE id = ?
                """,
                (segment_id,),
            )
        record_event(
            job_id,
            "queued_video",
            f"Regenerating video segment {segment_id}; other completed segments will be reused.",
        )
        background_tasks.add_task(run_video_job, job_id)
        return {"job_id": job_id, "segment_id": segment_id, "status": "queued_video"}

    def run_single_voice_segment(job_id: str, segment_id: str) -> None:
        try:
            with database.connect() as connection:
                job = connection.execute(
                    """
                    SELECT jobs.*, voice_profiles.sample_path, voice_profiles.profile_path,
                           voice_profiles.status AS voice_status
                    FROM jobs
                    JOIN voice_profiles ON jobs.voice_id = voice_profiles.id
                    WHERE jobs.id = ?
                    """,
                    (job_id,),
                ).fetchone()
                segment = connection.execute(
                    "SELECT * FROM segments WHERE id = ?", (segment_id,)
                ).fetchone()
            if job is None or segment is None:
                raise WorkerError("Job or segment no longer exists.")
            if job["voice_status"] != "ready" or not job["profile_path"]:
                raise WorkerError("The selected voice clone is no longer available.")
            set_job_status(job_id, "voice_generation", f"Regenerating segment {segment['position'] + 1}.")
            job_dir = app_data_dir / "jobs" / job_id
            output_audio = (
                job_dir
                / "audio"
                / f"{segment['position']:04d}-{uuid.uuid4().hex}.wav"
            )
            output_audio.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "job_id": job_id,
                "segment_id": segment_id,
                "position": segment["position"],
                "script": segment["text"],
                "language": job["language"],
                "style": job["style"],
                "voice_sample": job["sample_path"],
                "voice_profile": job["profile_path"],
                "output_audio": str(output_audio.resolve()),
                "work_dir": str(job_dir.resolve()),
            }
            with database.connect() as connection:
                connection.execute(
                    """
                    UPDATE segments SET status = 'generating', attempts = attempts + 1,
                        error = NULL WHERE id = ?
                    """,
                    (segment_id,),
                )
            try:
                worker_runner.run("tts", payload)
                if not _valid_wave_file(str(output_audio)):
                    raise WorkerError("TTS worker did not create a valid, non-empty WAV file.")
            except Exception as error:
                with database.connect() as connection:
                    connection.execute(
                        "UPDATE segments SET status = 'failed', error = ? WHERE id = ?",
                        (str(error), segment_id),
                    )
                raise
            with database.connect() as connection:
                connection.execute(
                    """
                    UPDATE segments SET status = 'ready_for_review', audio_path = ?
                    WHERE id = ?
                    """,
                    (str(output_audio.resolve()), segment_id),
                )
            set_job_status(job_id, "voice_review", "Review the regenerated segment.")
        except Exception as error:
            fail_job(job_id, error)

    @application.post("/api/jobs/{job_id}/approve-voice", status_code=202)
    def approve_voice(job_id: str, background_tasks: BackgroundTasks):
        with database.connect() as connection:
            job = connection.execute(
                "SELECT status, avatar_id FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            segments = connection.execute(
                "SELECT approved, audio_path FROM segments WHERE job_id = ?", (job_id,)
            ).fetchall()
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] != "voice_review":
            raise HTTPException(status_code=409, detail="Job is not waiting for voice approval.")
        if not segments or not all(
            segment["approved"] and _valid_wave_file(segment["audio_path"])
            for segment in segments
        ):
            raise HTTPException(
                status_code=409,
                detail="Approve every segment and keep its generated audio available.",
            )
        with database.connect() as connection:
            avatar = connection.execute(
                "SELECT image_path FROM avatars WHERE id = ?",
                (job["avatar_id"],),
            ).fetchone()
        if avatar is None or not Path(avatar["image_path"]).is_file():
            raise HTTPException(
                status_code=409,
                detail="The selected avatar reference photo is unavailable.",
            )
        require_worker_stages(VIDEO_WORKER_STAGES)
        with database.connect() as connection:
            connection.execute(
                "UPDATE jobs SET status = 'queued_video', updated_at = ? WHERE id = ?",
                (datetime.now(UTC).isoformat(), job_id),
            )
        record_event(job_id, "queued_video", "Voice approved; video pipeline queued.")
        background_tasks.add_task(run_video_job, job_id)
        return {"job_id": job_id, "status": "queued_video"}

    @application.post("/api/jobs/{job_id}/approve-video-review", status_code=202)
    def approve_video_review(job_id: str, background_tasks: BackgroundTasks):
        with database.connect() as connection:
            job = connection.execute(
                "SELECT status FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] != "video_review":
            raise HTTPException(
                status_code=409, detail="Job is not waiting for video quality review."
            )
        quality_report_path = app_data_dir / "jobs" / job_id / "quality-report.json"
        try:
            report = json.loads(quality_report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise HTTPException(
                status_code=500, detail="Quality report is unavailable or unreadable."
            ) from error
        if not isinstance(report, dict) or report.get("status") != "review_needed":
            raise HTTPException(
                status_code=409, detail="Job does not have a pending quality review."
            )
        require_worker_stages({"post"})
        with database.connect() as connection:
            connection.execute(
                "UPDATE jobs SET status = 'queued_finish', cancel_requested = 0, updated_at = ? WHERE id = ?",
                (datetime.now(UTC).isoformat(), job_id),
            )
        record_event(job_id, "queued_finish", "Quality review accepted; video finishing queued.")
        background_tasks.add_task(run_post_job, job_id)
        return {"job_id": job_id, "status": "queued_finish"}

    @application.post("/api/jobs/{job_id}/cancel", status_code=202)
    def cancel_job(job_id: str):
        with database.connect() as connection:
            job = connection.execute(
                "SELECT status FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            if job["status"] in TERMINAL_STATES:
                raise HTTPException(status_code=409, detail="Job is already finished.")
            now = datetime.now(UTC).isoformat()
            if job["status"] in {"draft", "voice_review", "video_review"}:
                connection.execute(
                    """
                    UPDATE jobs SET status = 'cancelled', cancel_requested = 1,
                        updated_at = ? WHERE id = ?
                    """,
                    (now, job_id),
                )
            else:
                connection.execute(
                    "UPDATE jobs SET cancel_requested = 1, updated_at = ? WHERE id = ?",
                    (now, job_id),
                )
        if job["status"] in {"draft", "voice_review", "video_review"}:
            record_event(job_id, "cancelled", "Job cancelled.")
            return {"job_id": job_id, "status": "cancelled"}
        record_event(job_id, "cancel_requested", "Cancellation requested; active worker will finish its current step.")
        return {"job_id": job_id, "status": "cancel_requested"}

    @application.get("/api/jobs/{job_id}/events")
    async def job_events(
        job_id: str,
        last_event_id: Annotated[int, Header(alias="Last-Event-ID")] = 0,
    ):
        with database.connect() as connection:
            if connection.execute(
                "SELECT 1 FROM jobs WHERE id = ?", (job_id,)
            ).fetchone() is None:
                raise HTTPException(status_code=404, detail="Job not found")

        async def stream_events():
            event_id = last_event_id
            while True:
                with database.connect() as connection:
                    rows = connection.execute(
                        """
                        SELECT id, created_at, event, detail FROM job_events
                        WHERE job_id = ? AND id > ? ORDER BY id
                        """,
                        (job_id, event_id),
                    ).fetchall()
                    job = connection.execute(
                        "SELECT status FROM jobs WHERE id = ?", (job_id,)
                    ).fetchone()
                for row in rows:
                    event_id = row["id"]
                    data = json.dumps(
                        {
                            "id": row["id"],
                            "created_at": row["created_at"],
                            "event": row["event"],
                            "detail": row["detail"],
                        }
                    )
                    yield f"id: {row['id']}\ndata: {data}\n\n"
                if job is None or job["status"] in {
                    "voice_review",
                    "draft",
                    *TERMINAL_STATES,
                }:
                    break
                yield ": keep-alive\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(stream_events(), media_type="text/event-stream")

    @application.get("/api/jobs/{job_id}/segments/{segment_id}/audio")
    def segment_audio(job_id: str, segment_id: str):
        with database.connect() as connection:
            segment = connection.execute(
                "SELECT audio_path FROM segments WHERE id = ? AND job_id = ?",
                (segment_id, job_id),
            ).fetchone()
        if segment is None or not segment["audio_path"] or not Path(segment["audio_path"]).is_file():
            raise HTTPException(status_code=404, detail="Segment audio not found")
        return FileResponse(segment["audio_path"], media_type="audio/wav")

    @application.get("/api/jobs/{job_id}/segments/{segment_id}/video")
    def segment_video(job_id: str, segment_id: str):
        with database.connect() as connection:
            segment = connection.execute(
                "SELECT video_path FROM segments WHERE id = ? AND job_id = ?",
                (segment_id, job_id),
            ).fetchone()
        if segment is None or not segment["video_path"] or not Path(segment["video_path"]).is_file():
            raise HTTPException(status_code=404, detail="Segment video not found")
        return FileResponse(segment["video_path"], media_type="video/mp4")

    @application.get("/api/jobs/{job_id}/video")
    def job_video(job_id: str):
        with database.connect() as connection:
            job = connection.execute(
                "SELECT video_path FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        if job is None or not job["video_path"] or not Path(job["video_path"]).is_file():
            raise HTTPException(status_code=404, detail="Finished video not found")
        return FileResponse(job["video_path"], media_type="video/mp4", filename=f"{job_id}.mp4")

    @application.get("/api/jobs/{job_id}/provenance")
    def job_provenance(job_id: str):
        with database.connect() as connection:
            job = connection.execute(
                "SELECT id FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        provenance_path = app_data_dir / "jobs" / job["id"] / "provenance.json"
        if not provenance_path.is_file():
            raise HTTPException(status_code=404, detail="Provenance record not found")
        try:
            return json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise HTTPException(status_code=500, detail="Provenance record is unreadable.") from error

    return application


class JobCancelled(Exception):
    pass


async def _save_upload(
    upload: UploadFile, kind: str, target_dir: Path, basename: str
) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES[kind]:
        raise ValueError(f"Unsupported {kind} file type: {suffix or '(missing extension)'}")
    content = await upload.read(UPLOAD_LIMITS[kind] + 1)
    if not content:
        raise ValueError(f"{basename} upload is empty.")
    if len(content) > UPLOAD_LIMITS[kind]:
        raise ValueError(f"{basename} exceeds the {UPLOAD_LIMITS[kind] // (1024 * 1024)} MB limit.")
    if not _has_expected_signature(suffix, content):
        raise ValueError(f"{basename} file content does not match its {suffix} extension.")
    path = target_dir / f"{basename}{suffix}"
    path.write_bytes(content)
    return path.resolve()


def _has_expected_signature(suffix: str, content: bytes) -> bool:
    if suffix == ".png":
        return content.startswith(b"\x89PNG\r\n\x1a\n")
    if suffix in {".jpg", ".jpeg"}:
        return content.startswith(b"\xff\xd8\xff")
    if suffix == ".webp":
        return content.startswith(b"RIFF") and content[8:12] == b"WEBP"
    if suffix == ".wav":
        return content.startswith(b"RIFF") and content[8:12] == b"WAVE"
    if suffix == ".mp3":
        return content.startswith(b"ID3") or (
            len(content) > 1 and content[0] == 0xFF and content[1] & 0xE0 == 0xE0
        )
    if suffix == ".m4a":
        return content[4:8] == b"ftyp"
    if suffix == ".webm":
        return content.startswith(b"\x1a\x45\xdf\xa3")
    if suffix == ".ogg":
        return content.startswith(b"OggS")
    if suffix == ".mp4":
        return content[4:8] == b"ftyp"
    return False


def _valid_wave_file(path: str | None) -> bool:
    if not path or not Path(path).is_file():
        return False
    try:
        with wave.open(path, "rb") as audio:
            return audio.getnframes() > 0
    except (wave.Error, EOFError, OSError):
        return False


def _is_mp4_file(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as video:
        return _has_expected_signature(".mp4", video.read(16))


app = create_app()
