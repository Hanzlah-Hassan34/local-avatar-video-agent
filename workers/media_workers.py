"""Local technical-QC and MP4-finishing workers for avatar jobs."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def read_spec() -> dict:
    spec_path = os.environ.get("AVATAR_JOB_SPEC")
    if not spec_path:
        raise RuntimeError("AVATAR_JOB_SPEC is not set.")
    return json.loads(Path(spec_path).read_text(encoding="utf-8"))


def ffmpeg_executable() -> str:
    import imageio_ffmpeg

    path = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    os.environ["PATH"] = str(path.parent) + os.pathsep + os.environ.get("PATH", "")
    return str(path)


def inspect_segments(paths: list[Path]) -> list[str]:
    import cv2

    if not paths:
        raise ValueError("No video segments were supplied for quality checks.")
    dimensions: set[tuple[int, int]] = set()
    durations: list[float] = []
    for path in paths:
        if not path.is_file() or path.stat().st_size < 16:
            raise FileNotFoundError(f"Video segment is empty or missing: {path}")
        with path.open("rb") as stream:
            if stream.read(8)[4:8] != b"ftyp":
                raise ValueError(f"Segment does not have an MP4 signature: {path}")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"OpenCV could not open video segment: {path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        if width < 2 or height < 2 or frame_count < 1 or fps <= 0:
            raise ValueError(f"Video segment has invalid video stream metadata: {path}")
        dimensions.add((width, height))
        durations.append(frame_count / fps)
    if len(dimensions) != 1:
        raise ValueError("All video segments must have matching dimensions before stitching.")
    if any(duration <= 0 for duration in durations):
        raise ValueError("One or more video segments have zero duration.")
    return [
        "Please review lip timing and portrait detail in the segment previews before finishing."
    ]


def run_qc(spec: dict) -> None:
    segments = [Path(path).resolve() for path in spec.get("segments", [])]
    flags = inspect_segments(segments)
    report = {
        "status": "review_needed",
        "flags": flags,
        "checks": {
            "mp4_signature": "pass",
            "video_stream_readable": "pass",
            "segment_dimensions_match": "pass",
            "audio_video_sync": "human_review_required",
            "portrait_identity": "human_review_required",
        },
    }
    report_path = Path(spec["qc_report"]).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def run_post(spec: dict) -> None:
    ffmpeg = ffmpeg_executable()
    inputs = [Path(path).resolve() for path in spec.get("segments", [])]
    if not inputs or any(not path.is_file() for path in inputs):
        raise FileNotFoundError("One or more video segments are missing before stitching.")
    output = Path(spec["output_video"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    manifest_path = output.with_name(f"{output.stem}-concat-list.txt")
    try:
        with manifest_path.open("w", encoding="utf-8", newline="\n") as manifest:
            for path in inputs:
                escaped = path.as_posix().replace("'", "'\\''")
                manifest.write(f"file '{escaped}'\n")

        command = [
            ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0",
            "-i", str(manifest_path), "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-movflags", "+faststart", str(output),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg stitching failed: {result.stderr[-2000:]}")
    finally:
        manifest_path.unlink(missing_ok=True)

    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("FFmpeg did not create the finished MP4.")


def main() -> None:
    spec = read_spec()
    stage = os.environ.get("AVATAR_STAGE")
    if stage == "qc":
        run_qc(spec)
    elif stage == "post":
        run_post(spec)
    else:
        raise ValueError(f"Unsupported media worker stage: {stage!r}")


if __name__ == "__main__":
    main()
