import importlib.util
import json
import sys
import time
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.workers import WorkerError, WorkerRunner


def _wav_bytes():
    import io

    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        audio.writeframes(bytes(1600))
    return output.getvalue()


def _create_avatar(client, include_driving_video=True):
    files = {
        "reference_image": ("reference.png", b"\x89PNG\r\n\x1a\nimage", "image/png"),
        "consent_recording": (
            "consent.webm",
            b"\x1a\x45\xdf\xa3recording",
            "audio/webm",
        ),
    }
    if include_driving_video:
        files["driving_video"] = (
            "driving.mp4",
            b"\x00\x00\x00\x18ftypisomvideo",
            "video/mp4",
        )
    response = client.post(
        "/api/avatars",
        data={
            "name": "Test avatar",
            "consent_text": "I consent to using my face for this generated video.",
            "consent_confirmed": "true",
        },
        files=files,
    )
    assert response.status_code == 201, response.text
    return response


def _create_voice(client):
    response = client.post(
        "/api/voices",
        data={
            "name": "Test voice",
        },
        files={
            "voice_sample": ("voice.wav", _wav_bytes(), "audio/wav"),
        },
    )
    assert response.status_code == 202, response.text
    voice_id = response.json()["id"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        profile = next(
            (profile for profile in client.get("/api/voices").json() if profile["id"] == voice_id),
            None,
        )
        if profile and profile["status"] not in {"queued", "cloning"}:
            return profile
        time.sleep(0.05)
    raise AssertionError(f"Voice clone did not finish: {profile}")


def test_avatar_requires_explicit_consent(tmp_path):
    with TestClient(create_app(tmp_path / "jobs.sqlite3")) as client:
        response = client.post(
            "/api/avatars",
            data={
                "name": "Test avatar",
                "consent_text": "I consent to using my face in this video.",
                "consent_confirmed": "false",
            },
            files={
                "reference_image": ("reference.png", b"\x89PNG\r\n\x1a\nimage", "image/png"),
                "consent_recording": ("consent.webm", b"\x1a\x45\xdf\xa3audio", "audio/webm"),
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Explicit face consent is required."


def test_avatar_rejects_file_with_mismatched_content(tmp_path):
    with TestClient(create_app(tmp_path / "jobs.sqlite3")) as client:
        response = client.post(
            "/api/avatars",
            data={
                "name": "Test avatar",
                "consent_text": "I consent to using my face in this video.",
                "consent_confirmed": "true",
            },
            files={
                "reference_image": ("reference.png", b"not a png", "image/png"),
                "consent_recording": ("consent.webm", b"\x1a\x45\xdf\xa3audio", "audio/webm"),
            },
        )

    assert response.status_code == 400
    assert "does not match" in response.json()["detail"]


def test_existing_avatar_can_receive_a_driving_video(tmp_path):
    with TestClient(create_app(tmp_path / "jobs.sqlite3", data_dir=tmp_path)) as client:
        avatar = _create_avatar(client, include_driving_video=False).json()
        response = client.post(
            f"/api/avatars/{avatar['id']}/driving-video",
            files={
                "driving_video": (
                    "replacement.webm",
                    b"\x1a\x45\xdf\xa3video",
                    "video/webm",
                )
            },
        )
        refreshed = client.get("/api/avatars").json()[0]
        with client.app.state.database.connect() as connection:
            path = connection.execute(
                "SELECT driving_video_path FROM avatars WHERE id = ?",
                (avatar["id"],),
            ).fetchone()["driving_video_path"]

    assert response.status_code == 200
    assert response.json()["has_driving_video"] is True
    assert refreshed["has_driving_video"] is True
    assert Path(path).is_file()
    assert Path(path).suffix == ".webm"


def test_existing_avatar_driving_video_rejects_invalid_file(tmp_path):
    with TestClient(create_app(tmp_path / "jobs.sqlite3", data_dir=tmp_path)) as client:
        avatar = _create_avatar(client, include_driving_video=False).json()
        response = client.post(
            f"/api/avatars/{avatar['id']}/driving-video",
            files={
                "driving_video": (
                    "replacement.mp4",
                    b"not an mp4",
                    "video/mp4",
                )
            },
        )
        with client.app.state.database.connect() as connection:
            path = connection.execute(
                "SELECT driving_video_path FROM avatars WHERE id = ?",
                (avatar["id"],),
            ).fetchone()["driving_video_path"]

    assert response.status_code == 400
    assert "does not match" in response.json()["detail"]
    assert path is None


def test_existing_avatar_driving_video_returns_not_found(tmp_path):
    with TestClient(create_app(tmp_path / "jobs.sqlite3", data_dir=tmp_path)) as client:
        response = client.post(
            "/api/avatars/unknown/driving-video",
            files={"driving_video": ("clip.mp4", b"\x00\x00\x00\x18ftypisom", "video/mp4")},
        )

    assert response.status_code == 404


def test_voice_sample_accepts_ogg(tmp_path):
    config_path = tmp_path / "workers.json"
    config_path.write_text(
        json.dumps({"voice_clone": {"command": [sys.executable, "-c", "pass"]}}),
        encoding="utf-8",
    )
    with TestClient(
        create_app(tmp_path / "jobs.sqlite3", worker_config_path=config_path)
    ) as client:
        response = client.post(
            "/api/voices",
            data={"name": "OGG voice"},
            files={"voice_sample": ("voice.ogg", b"OggS" + bytes(40), "audio/ogg")},
        )

    assert response.status_code == 202, response.text
    assert (tmp_path / "uploads" / "voices" / response.json()["id"] / "voice-sample.ogg").is_file()


def test_voice_sample_rejects_invalid_ogg_signature(tmp_path):
    config_path = tmp_path / "workers.json"
    config_path.write_text(
        json.dumps({"voice_clone": {"command": [sys.executable, "-c", "pass"]}}),
        encoding="utf-8",
    )
    with TestClient(create_app(tmp_path / "jobs.sqlite3", worker_config_path=config_path)) as client:
        response = client.post(
            "/api/voices",
            data={"name": "Invalid OGG"},
            files={"voice_sample": ("voice.ogg", b"not an ogg file", "audio/ogg")},
        )

    assert response.status_code == 400
    assert "does not match" in response.json()["detail"]


def test_voice_profile_creation_does_not_require_consent_fields(tmp_path):
    clone_code = (
        "import json,os,pathlib;"
        "s=json.load(open(os.environ['AVATAR_JOB_SPEC']));"
        "pathlib.Path(s['output_profile']).write_text('{{\"format\":\"fake-v1\",\"profile\":{{\"speaker_id\":\"fake\"}}}}')"
    )
    config_path = tmp_path / "workers.json"
    config_path.write_text(
        json.dumps({"voice_clone": {"command": [sys.executable, "-c", clone_code]}}),
        encoding="utf-8",
    )
    with TestClient(
        create_app(
            tmp_path / "jobs.sqlite3",
            data_dir=tmp_path,
            worker_config_path=config_path,
        )
    ) as client:
        profile = _create_voice(client)

    assert profile["status"] == "ready"


def test_configured_tts_worker_creates_reviewable_segment(tmp_path):
    code = (
        "import json,os,wave;"
        "p=json.load(open(os.environ['AVATAR_JOB_SPEC']))['output_audio'];"
        "w=wave.open(p,'wb');"
        "w.setparams((1,2,8000,0,'NONE','not compressed'));"
        "w.writeframes(bytes(1600));"
        "w.close()"
    )
    clone_code = (
        "import json,os,pathlib;"
        "s=json.load(open(os.environ['AVATAR_JOB_SPEC']));"
        "assert 'consent_text' not in s and 'consent_recording' not in s;"
        "pathlib.Path(s['output_profile']).write_text('{{\"format\":\"fake-v1\",\"profile\":{{\"speaker_id\":\"fake\"}}}}')"
    )
    config = {
        "voice_clone": {"command": [sys.executable, "-c", clone_code]},
        "tts": {"command": [sys.executable, "-c", code]},
    }
    config_path = tmp_path / "workers.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with TestClient(
        create_app(
            tmp_path / "jobs.sqlite3",
            data_dir=tmp_path,
            worker_config_path=config_path,
        )
    ) as client:
        avatar = _create_avatar(client).json()
        voice = _create_voice(client)
        assert voice["status"] == "ready", voice
        with client.app.state.database.connect() as connection:
            expected_profile_path = connection.execute(
                "SELECT profile_path FROM voice_profiles WHERE id = ?",
                (voice["id"],),
            ).fetchone()["profile_path"]
        job = client.post(
            "/api/jobs",
            json={
                "script": "A test line.",
                "avatar_id": avatar["id"],
                "voice_id": voice["id"],
            },
        ).json()
        response = client.post(f"/api/jobs/{job['id']}/generate-voice")
        reviewed = client.get(f"/api/jobs/{job['id']}").json()
        job_spec = json.loads(
            (tmp_path / "jobs" / job["id"] / "tts-0.json").read_text(
                encoding="utf-8"
            )
        )

    assert response.status_code == 202
    assert reviewed["status"] == "voice_review"
    assert reviewed["segments"][0]["status"] == "ready_for_review"
    assert reviewed["segments"][0]["audio_url"].endswith("/audio")
    assert reviewed["voice_name"] == "Test voice"
    assert job_spec["voice_profile"] == expected_profile_path


def test_missing_tts_worker_is_reported_before_queueing(tmp_path):
    config_path = tmp_path / "workers.json"
    clone_code = (
        "import json,os,pathlib;"
        "s=json.load(open(os.environ['AVATAR_JOB_SPEC']));"
        "pathlib.Path(s['output_profile']).write_text('{{\"format\":\"fake-v1\",\"profile\":{{\"speaker_id\":\"fake\"}}}}')"
    )
    config_path.write_text(
        json.dumps({"voice_clone": {"command": [sys.executable, "-c", clone_code]}}),
        encoding="utf-8",
    )
    with TestClient(create_app(tmp_path / "jobs.sqlite3", data_dir=tmp_path, worker_config_path=config_path)) as client:
        avatar = _create_avatar(client).json()
        voice = _create_voice(client)
        assert voice["status"] == "ready", voice
        job = client.post(
            "/api/jobs",
            json={"script": "A test line.", "avatar_id": avatar["id"], "voice_id": voice["id"]},
        ).json()
        response = client.post(f"/api/jobs/{job['id']}/generate-voice")
        unchanged = client.get(f"/api/jobs/{job['id']}").json()
        assert response.status_code == 503
        assert "tts" in response.json()["detail"]
        assert unchanged["status"] == "draft"
        assert unchanged["segments"][0]["status"] == "pending"

        code = (
            "import json,os,wave;"
            "p=json.load(open(os.environ['AVATAR_JOB_SPEC']))['output_audio'];"
            "w=wave.open(p,'wb');"
            "w.setparams((1,2,8000,0,'NONE','not compressed'));"
            "w.writeframes(bytes(1600));"
            "w.close()"
        )
        config_path.write_text(
            json.dumps({"tts": {"command": [sys.executable, "-c", code]}}),
            encoding="utf-8",
        )
        queued = client.post(f"/api/jobs/{job['id']}/generate-voice")
        recovered = client.get(f"/api/jobs/{job['id']}").json()

    assert queued.status_code == 202
    assert recovered["status"] == "voice_review"
    assert recovered["segments"][0]["status"] == "ready_for_review"
    assert recovered["segments"][0]["attempts"] == 1


def test_missing_video_workers_are_reported_before_voice_approval(tmp_path):
    tts = (
        "import json,os,wave;"
        "p=json.load(open(os.environ['AVATAR_JOB_SPEC']))['output_audio'];"
        "w=wave.open(p,'wb');"
        "w.setparams((1,2,8000,0,'NONE','not compressed'));"
        "w.writeframes(bytes(1600));"
        "w.close()"
    )
    clone = (
        "import json,os,pathlib;"
        "s=json.load(open(os.environ['AVATAR_JOB_SPEC']));"
        "pathlib.Path(s['output_profile']).write_text('{{\"format\":\"fake-v1\",\"profile\":{{\"speaker_id\":\"fake\"}}}}')"
    )
    config_path = tmp_path / "workers.json"
    config_path.write_text(
        json.dumps(
            {
                "voice_clone": {"command": [sys.executable, "-c", clone]},
                "tts": {"command": [sys.executable, "-c", tts]},
            }
        ),
        encoding="utf-8",
    )
    with TestClient(
        create_app(
            tmp_path / "jobs.sqlite3",
            data_dir=tmp_path,
            worker_config_path=config_path,
        )
    ) as client:
        avatar = _create_avatar(client).json()
        voice = _create_voice(client)
        job = client.post(
            "/api/jobs",
            json={"script": "A test line.", "avatar_id": avatar["id"], "voice_id": voice["id"]},
        ).json()
        assert client.post(f"/api/jobs/{job['id']}/generate-voice").status_code == 202
        reviewed = client.get(f"/api/jobs/{job['id']}").json()
        assert client.post(
            f"/api/jobs/{job['id']}/segments/{reviewed['segments'][0]['id']}/approve"
        ).status_code == 200
        response = client.post(f"/api/jobs/{job['id']}/approve-voice")
        unchanged = client.get(f"/api/jobs/{job['id']}").json()

    assert response.status_code == 503
    assert all(stage in response.json()["detail"] for stage in ("motion", "lip_sync", "qc", "post"))
    assert unchanged["status"] == "voice_review"


def test_voice_clone_requires_worker_before_creating_profile(tmp_path):
    with TestClient(create_app(tmp_path / "jobs.sqlite3", data_dir=tmp_path)) as client:
        response = client.post(
            "/api/voices",
            data={"name": "Test voice"},
            files={"voice_sample": ("voice.wav", _wav_bytes(), "audio/wav")},
        )
        profiles = client.get("/api/voices").json()

    assert response.status_code == 503
    assert "local voice_clone worker" in response.json()["detail"]
    assert "workers.json" in response.json()["detail"]
    assert profiles == []


def test_cosyvoice_worker_rejects_incomplete_model_files(tmp_path):
    worker_path = Path(__file__).resolve().parents[1] / "workers" / "cosyvoice2_worker.py"
    spec = importlib.util.spec_from_file_location("cosyvoice2_worker_test", worker_path)
    assert spec is not None and spec.loader is not None
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    worker.MODEL_DIR = tmp_path

    for name in worker.REQUIRED_MODEL_FILES:
        model_file = tmp_path / name
        model_file.parent.mkdir(parents=True, exist_ok=True)
        model_file.write_bytes(b"model")
    (tmp_path / "llm.pt").write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="llm.pt"):
        worker.validate_model_files()

    (tmp_path / "llm.pt").write_bytes(b"model")
    worker.validate_model_files()


@pytest.mark.parametrize("qc_status", ["pass", "review_needed"])
def test_fake_workers_complete_reviewed_job_and_write_provenance(tmp_path, qc_status):
    tts = (
        "import json,os,wave;"
        "p=json.load(open(os.environ['AVATAR_JOB_SPEC']))['output_audio'];"
        "w=wave.open(p,'wb');"
        "w.setparams((1,2,8000,0,'NONE','not compressed'));"
        "w.writeframes(bytes(1600));"
        "w.close()"
    )
    write_motion = (
        "import json,os,pathlib;"
        "p=json.load(open(os.environ['AVATAR_JOB_SPEC']))['motion_video'];"
        "pathlib.Path(p).write_bytes(bytes([0,0,0,24])+b'ftypisom')"
    )
    write_lipsync = (
        "import json,os,pathlib;"
        "p=json.load(open(os.environ['AVATAR_JOB_SPEC']))['output_video'];"
        "pathlib.Path(p).write_bytes(bytes([0,0,0,24])+b'ftypisom')"
    )
    write_qc = (
        "import json,os,pathlib;"
        "s=json.load(open(os.environ['AVATAR_JOB_SPEC']));"
        "p=s['qc_report'];"
        f"status={qc_status!r};"
        "flags=['face drift'] if status == 'review_needed' else [];"
        "pathlib.Path(p).write_text(json.dumps(dict(status=status,flags=flags)))"
    )
    write_final = (
        "import json,os,pathlib;"
        "p=json.load(open(os.environ['AVATAR_JOB_SPEC']))['output_video'];"
        "pathlib.Path(p).write_bytes(bytes([0,0,0,24])+b'ftypisom')"
    )
    config = {
        stage: {"command": [sys.executable, "-c", code]}
        for stage, code in {
            "tts": tts,
            "motion": write_motion,
            "lip_sync": write_lipsync,
            "qc": write_qc,
            "post": write_final,
                "voice_clone": (
                    "import json,os,pathlib;"
                    "s=json.load(open(os.environ['AVATAR_JOB_SPEC']));"
                    "pathlib.Path(s['output_profile']).write_text('{{\"format\":\"fake-v1\",\"profile\":{{\"speaker_id\":\"fake\"}}}}')"
                ),
            }.items()
    }
    config_path = tmp_path / "workers.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with TestClient(
        create_app(
            tmp_path / "jobs.sqlite3",
            data_dir=tmp_path,
            worker_config_path=config_path,
        )
    ) as client:
        avatar = _create_avatar(client).json()
        voice = _create_voice(client)
        assert voice["status"] == "ready", voice
        job = client.post(
            "/api/jobs",
            json={"script": "Approved sample.", "avatar_id": avatar["id"], "voice_id": voice["id"]},
        ).json()
        client.post(f"/api/jobs/{job['id']}/generate-voice")
        job = client.get(f"/api/jobs/{job['id']}").json()
        assert client.post(
            f"/api/jobs/{job['id']}/approve-voice"
        ).status_code == 409
        client.post(
            f"/api/jobs/{job['id']}/segments/{job['segments'][0]['id']}/approve"
        )
        response = client.post(f"/api/jobs/{job['id']}/approve-voice")
        after_qc = client.get(f"/api/jobs/{job['id']}").json()
        if qc_status == "review_needed":
            assert after_qc["status"] == "video_review"
            assert after_qc["quality_status"] == "review_needed"
            assert after_qc["quality_flags"] == ["face drift"]
            assert not (tmp_path / "jobs" / job["id"] / "post-job.log").exists()
            quality_response = client.post(
                f"/api/jobs/{job['id']}/approve-video-review"
            )
            assert quality_response.status_code == 202
            assert client.post(
                f"/api/jobs/{job['id']}/approve-video-review"
            ).status_code == 409
        else:
            assert after_qc["status"] == "completed"
            quality_response = None
        finished = client.get(f"/api/jobs/{job['id']}").json()
        assert finished["status"] == "completed", finished["error"]
        video = client.get(f"/api/jobs/{job['id']}/video")
        if qc_status == "pass":
            with client.app.state.database.connect() as connection:
                previous_video_path = connection.execute(
                    "SELECT video_path FROM segments WHERE job_id = ?", (job["id"],)
                ).fetchone()["video_path"]
            regenerated = client.post(
                f"/api/jobs/{job['id']}/segments/{job['segments'][0]['id']}/regenerate-video"
            )
            after_regeneration = client.get(f"/api/jobs/{job['id']}").json()
            with client.app.state.database.connect() as connection:
                new_video_path = connection.execute(
                    "SELECT video_path FROM segments WHERE job_id = ?", (job["id"],)
                ).fetchone()["video_path"]
        provenance = json.loads(
            (tmp_path / "jobs" / job["id"] / "provenance.json").read_text(
                encoding="utf-8"
            )
        )

    assert response.status_code == 202
    assert finished["status"] == "completed"
    assert finished["video_url"] == f"/api/jobs/{job['id']}/video"
    assert finished["quality_status"] == qc_status
    assert video.status_code == 200
    assert video.content[4:8] == b"ftyp"
    assert provenance["ai_generated"] is True
    assert provenance["quality_status"] == qc_status
    if qc_status == "review_needed":
        assert quality_response.json()["status"] == "queued_finish"
    else:
        assert regenerated.status_code == 202
        assert after_regeneration["status"] == "completed"
        assert previous_video_path != new_video_path
    assert (tmp_path / "jobs" / job["id"] / "segments").exists()


def test_worker_timeout_preserves_log_output(tmp_path):
    config_path = tmp_path / "workers.json"
    config_path.write_text(
        json.dumps(
            {
                "tts": {
                    "command": [
                        sys.executable,
                        "-c",
                        "import time;print('worker started',flush=True);time.sleep(5)",
                    ],
                    "timeout_seconds": 1,
                }
            }
        ),
        encoding="utf-8",
    )
    runner = WorkerRunner(config_path)

    try:
        runner.run("tts", {"work_dir": str(tmp_path), "position": 0})
    except WorkerError as error:
        assert "exceeded its 1s timeout" in str(error)
        assert "worker started" in str(error)
    else:
        raise AssertionError("Expected the worker timeout to be reported.")
