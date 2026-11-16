"""Run MuseTalk on one approved audio segment and a still portrait."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
MUSE_ROOT = APP_ROOT / "data" / "vendor" / "MuseTalk"
MODEL_ROOT = APP_ROOT / "data" / "models" / "MuseTalk"
MUSE_PACKAGE_ROOT = APP_ROOT / "data" / "venvs" / "musetalk" / "Lib" / "site-packages"


def read_spec() -> dict:
    spec_path = os.environ.get("AVATAR_JOB_SPEC")
    if not spec_path:
        raise RuntimeError("AVATAR_JOB_SPEC is not set.")
    return json.loads(Path(spec_path).read_text(encoding="utf-8"))


def _read_images(image_paths: list[str]):
    import cv2

    images = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"MuseTalk could not read the reference photo: {image_path}")
        images.append(image)
    return images


def _portrait_boxes(image_paths: list[str], upperbondrange: int = 0):
    """Get a padded face crop for a still photo without loading DWPose/MMPose."""
    import cv2

    frames = _read_images(image_paths)
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"OpenCV face detector is unavailable: {cascade_path}")

    boxes = []
    for frame in frames:
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(64, 64),
        )
        if len(faces) == 0:
            raise ValueError("No clear face was detected in the avatar photo.")

        x, y, face_width, face_height = max(faces, key=lambda item: item[2] * item[3])
        pad_x = int(face_width * 0.18)
        pad_top = int(face_height * 0.30)
        pad_bottom = int(face_height * 0.16)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_top + upperbondrange)
        x2 = min(width, x + face_width + pad_x)
        y2 = min(height, y + face_height + pad_bottom + upperbondrange)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("The detected face crop is empty.")
        boxes.append((x1, y1, x2, y2))

    return boxes, frames


def _patch_still_image_preprocessing() -> None:
    """MuseTalk's default preprocessing imports a pose stack not needed for a still."""
    module = types.ModuleType("musetalk.utils.preprocessing")
    module.coord_placeholder = (0.0, 0.0, 0.0, 0.0)
    module.read_imgs = _read_images
    module.get_landmark_and_bbox = _portrait_boxes
    sys.modules[module.__name__] = module


def _validate_model_files() -> dict[str, Path]:
    paths = {
        "unet": MODEL_ROOT / "musetalkV15" / "unet.pth",
        "unet_config": MODEL_ROOT / "musetalkV15" / "musetalk.json",
        "vae_config": MODEL_ROOT / "sd-vae" / "config.json",
        "vae_weights": MODEL_ROOT / "sd-vae" / "diffusion_pytorch_model.bin",
        "whisper_config": MODEL_ROOT / "whisper" / "config.json",
        "whisper_weights": MODEL_ROOT / "whisper" / "pytorch_model.bin",
        "whisper_preprocessor": MODEL_ROOT / "whisper" / "preprocessor_config.json",
        "face_parser": MODEL_ROOT / "face-parse-bisent" / "79999_iter.pth",
        "face_parser_resnet": MODEL_ROOT / "face-parse-bisent" / "resnet18-5c106cde.pth",
    }
    missing = [str(path) for path in paths.values() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("MuseTalk model files are missing: " + ", ".join(missing))
    return paths


def run_lip_sync(spec: dict) -> None:
    sys.path.insert(0, str(MUSE_PACKAGE_ROOT))
    import yaml

    reference_image = Path(spec["reference_image"]).resolve()
    input_audio = Path(spec["input_audio"]).resolve()
    output_video = Path(spec["output_video"]).resolve()
    for path in (reference_image, input_audio):
        if not path.is_file():
            raise FileNotFoundError(f"MuseTalk input is missing: {path}")
    model_files = _validate_model_files()

    import imageio_ffmpeg

    ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    os.environ["PATH"] = str(ffmpeg_exe.parent) + os.pathsep + os.environ.get("PATH", "")
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    sys.path.insert(0, str(MUSE_ROOT))
    _patch_still_image_preprocessing()

    from musetalk.utils.face_parsing import FaceParsing

    original_model_init = FaceParsing.model_init

    def model_init(self, resnet_path=None, model_pth=None):
        return original_model_init(
            self,
            resnet_path=str(model_files["face_parser_resnet"]),
            model_pth=str(model_files["face_parser"]),
        )

    FaceParsing.model_init = model_init

    from omegaconf import OmegaConf
    import scripts.inference as inference_module
    from argparse import Namespace

    original_subprocess_run = subprocess.run

    def run_with_ffmpeg_path(command, *args, **kwargs):
        if (
            isinstance(command, (list, tuple))
            and command
            and Path(str(command[0])).name.lower() in {"ffmpeg", "ffmpeg.exe"}
        ):
            command = list(command)
            command[0] = str(ffmpeg_exe)
            if "-vf" in command:
                filter_index = command.index("-vf") + 1
                if command[filter_index] == "format=yuv420p":
                    command[filter_index] = (
                        "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p"
                    )
        return original_subprocess_run(command, *args, **kwargs)

    inference_module.subprocess.run = run_with_ffmpeg_path

    work_dir = Path(spec["work_dir"]).resolve()
    result_dir = work_dir / "musetalk" / f"segment-{int(spec['position']):04d}"
    result_dir.mkdir(parents=True, exist_ok=True)
    config_path = result_dir / "inference.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "task_0": {
                    "video_path": str(reference_image),
                    "audio_path": str(input_audio),
                    "result_name": output_video.name,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    args = Namespace(
        ffmpeg_path=str(ffmpeg_exe.parent),
        gpu_id=0,
        vae_type=str(MODEL_ROOT / "sd-vae"),
        unet_config=str(model_files["unet_config"]),
        unet_model_path=str(model_files["unet"]),
        whisper_dir=str(MODEL_ROOT / "whisper"),
        inference_config=str(config_path),
        bbox_shift=0,
        result_dir=str(result_dir),
        extra_margin=10,
        fps=25,
        audio_padding_length_left=2,
        audio_padding_length_right=2,
        batch_size=1,
        output_vid_name=output_video.name,
        use_saved_coord=False,
        saved_coord=False,
        use_float16=True,
        parsing_mode="jaw",
        left_cheek_width=90,
        right_cheek_width=90,
        version="v15",
    )
    try:
        inference_module.main(args)
    finally:
        inference_module.subprocess.run = original_subprocess_run

    generated = result_dir / "v15" / output_video.name
    if not generated.is_file() or generated.stat().st_size == 0:
        raise RuntimeError(f"MuseTalk did not create its expected MP4: {generated}")
    output_video.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(generated, output_video)


def main_worker() -> None:
    if os.environ.get("AVATAR_STAGE") != "lip_sync":
        raise ValueError("MuseTalk worker only handles the lip_sync stage.")
    run_lip_sync(read_spec())


if __name__ == "__main__":
    main_worker()
