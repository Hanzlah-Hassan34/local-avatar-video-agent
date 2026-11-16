import json
import os
from pathlib import Path
import sys


APP_ROOT = Path(__file__).resolve().parents[1]
COSYVOICE_ROOT = APP_ROOT / "data" / "vendor" / "CosyVoice"
MATCHA_ROOT = COSYVOICE_ROOT / "third_party" / "Matcha-TTS"
MODEL_DIR = Path(
    os.environ.get(
        "AVATAR_COSYVOICE2_MODEL",
        APP_ROOT / "data" / "models" / "CosyVoice2-0.5B",
    )
).resolve()
REQUIRED_MODEL_FILES = (
    "cosyvoice2.yaml",
    "llm.pt",
    "flow.pt",
    "hift.pt",
    "campplus.onnx",
    "speech_tokenizer_v2.onnx",
    "CosyVoice-BlankEN/model.safetensors",
    "CosyVoice-BlankEN/tokenizer_config.json",
)
sys.path.insert(0, str(MATCHA_ROOT))
sys.path.insert(0, str(COSYVOICE_ROOT))


def read_spec() -> dict:
    spec_path = os.environ.get("AVATAR_JOB_SPEC")
    if not spec_path:
        raise RuntimeError("AVATAR_JOB_SPEC is not set.")
    with Path(spec_path).open(encoding="utf-8") as spec_file:
        return json.load(spec_file)


def validate_model_files() -> None:
    missing = [
        name
        for name in REQUIRED_MODEL_FILES
        if not (MODEL_DIR / name).is_file() or (MODEL_DIR / name).stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(
            f"CosyVoice 2 model is incomplete at {MODEL_DIR}; missing or empty: {', '.join(missing)}."
        )


def prepare_voice_profile(spec: dict) -> None:
    import torch
    import torchaudio

    validate_model_files()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The CosyVoice worker cannot access CUDA. Install a CUDA-enabled "
            "PyTorch build in the CosyVoice environment."
        )

    source = Path(spec["voice_sample"])
    if not source.is_file():
        raise FileNotFoundError(f"Voice sample not found: {source}")

    try:
        audio, sample_rate = torchaudio.load(str(source), backend="soundfile")
    except Exception as error:
        raise RuntimeError(
            "Could not decode the voice sample. Use a valid WAV, MP3, M4A, WebM, or OGG file."
        ) from error
    if audio.numel() == 0 or sample_rate <= 0:
        raise ValueError("Voice sample contains no audio.")
    audio = audio.mean(dim=0, keepdim=True)
    audio = audio[:, : int(sample_rate * 28)]
    if sample_rate != 24000:
        audio = torchaudio.functional.resample(audio, sample_rate, 24000)
    if not torch.isfinite(audio).all() or torch.max(torch.abs(audio)).item() == 0:
        raise ValueError("Voice sample must contain finite, audible speech.")

    work_dir = Path(spec["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    prompt_audio = work_dir / "cosyvoice-prompt.wav"
    torchaudio.save(str(prompt_audio), audio.cpu(), 24000)
    artifact = {
        "format": "cosyvoice2-zero-shot-v1",
        "profile": {
            "model": "CosyVoice2-0.5B",
            "prompt_audio": str(prompt_audio.resolve()),
        },
    }
    Path(spec["output_profile"]).write_text(
        json.dumps(artifact, indent=2),
        encoding="utf-8",
    )


def synthesize(spec: dict) -> None:
    import torch
    import torchaudio

    # The optional WeText frontend otherwise downloads its full multilingual
    # normalizer on first startup. CosyVoice's built-in number handling and
    # tokenizer are sufficient for the reviewed English script workflow.
    os.environ.setdefault("COSYVOICE_USE_WETEXT", "0")
    from cosyvoice.cli.cosyvoice import CosyVoice2

    validate_model_files()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The CosyVoice worker cannot access CUDA. Install a CUDA-enabled "
            "PyTorch build in the CosyVoice environment."
        )

    profile_path = Path(spec["voice_profile"])
    artifact = json.loads(profile_path.read_text(encoding="utf-8"))
    if artifact.get("format") != "cosyvoice2-zero-shot-v1":
        raise ValueError("Voice profile was not created for CosyVoice 2.")
    profile = artifact.get("profile")
    if not isinstance(profile, dict):
        raise ValueError("CosyVoice voice profile is invalid.")
    prompt_audio = Path(profile["prompt_audio"])
    if not prompt_audio.is_file():
        raise FileNotFoundError(f"CosyVoice prompt audio not found: {prompt_audio}")

    model = CosyVoice2(str(MODEL_DIR), fp16=True)
    instructions = {
        "calm": "Speak calmly and clearly, at a measured pace.<|endofprompt|>",
        "natural": "Speak naturally and clearly in a conversational tone.<|endofprompt|>",
        "energetic": "Speak with energetic, expressive delivery.<|endofprompt|>",
    }
    style = spec.get("style", "natural")
    if style not in instructions:
        raise ValueError(f"Unsupported voice style: {style}")

    chunks = []
    with torch.inference_mode():
        for result in model.inference_instruct2(
            spec["script"],
            instructions[style],
            str(prompt_audio),
            stream=False,
        ):
            speech = result["tts_speech"]
            if speech.numel():
                chunks.append(speech.detach().cpu())
    if not chunks:
        raise RuntimeError("CosyVoice returned no speech audio.")

    output_audio = Path(spec["output_audio"])
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(
        str(output_audio),
        torch.cat(chunks, dim=1),
        model.sample_rate,
        encoding="PCM_S",
        bits_per_sample=16,
    )


def main() -> None:
    spec = read_spec()
    stage = os.environ.get("AVATAR_STAGE")
    if stage == "voice_clone":
        prepare_voice_profile(spec)
    elif stage == "tts":
        synthesize(spec)
    else:
        raise ValueError(f"Unsupported CosyVoice worker stage: {stage!r}")


if __name__ == "__main__":
    main()
