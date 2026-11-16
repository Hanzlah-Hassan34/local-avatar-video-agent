# Local Avatar Video Agent

A local-first FastAPI application for an avatar-video workflow. Appearance avatars and cloned voice profiles are separate: create a voice profile in the Voices screen, then select it independently when starting a script job.

**The cloned-voice path has generated a short CUDA WAV sample, and still-photo lip-sync, QC, and MP4 finishing worker code is now present. All configured MuseTalk model files are downloaded, and its key dependencies import in the configured CUDA environment. No lip-sync inference or final MP4 has been verified. The 6 GB GTX 1660 Super has been confirmed for CosyVoice CUDA inference; MuseTalk fit and output quality remain unknown. Do not treat a configured worker as proof that its model loads or fits in VRAM.

## Requirements

- Windows 10/11
- Python 3.12+
- CUDA-enabled PyTorch and compatible NVIDIA dependencies for GPU model workers
- Local worker scripts/environments for the stages you configure

Model weights are downloaded separately; the app does not manage model downloads. Check model and weight licenses before use. Use only reference faces and voice recordings you have permission to use. Avatar face consent recordings are stored locally. The voice-profile form does not collect consent text, recordings, or a permission checkbox.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload --reload-dir app --reload-dir static
```

Open `http://127.0.0.1:8000`. Keep the server bound to loopback; the app does not implement user authentication and is not intended to be exposed to a network.

By default, the app stores SQLite data, uploaded avatars/consent recordings, job files, generated media, logs, and provenance under `data\`. Override that root with `AVATAR_DATA_DIR`. The worker configuration defaults to `data\workers.json` and can be changed with `AVATAR_WORKERS_CONFIG`.

The API and interactive docs are available at `http://127.0.0.1:8000/docs`. `GET /api/system` reports PyTorch/CUDA visibility and configured worker stages.

## Worker contract

Workers run as separate local subprocesses. Configure them in `data\workers.json` using argument arrays; the app does not invoke a shell. Worker commands are trusted local code, so review every configured executable and script. Each command may use `{spec}` for the absolute JSON job-spec path. The same path is also supplied in `AVATAR_JOB_SPEC`, with the stage name in `AVATAR_STAGE`.

Voice samples and avatar consent recordings accept WAV, MP3, M4A, WebM, and OGG uploads. OGG uploads are signature-checked, but audio decoding is performed by the configured worker; ensure its model/audio stack supports OGG, or convert the sample to WAV first.

## CosyVoice 2 setup

CosyVoice 2 is isolated from the app in `data\venvs\cosyvoice` (Python 3.10), with its source checkout in `data\vendor\CosyVoice` and the `FunAudioLLM/CosyVoice2-0.5B` model targeted at `data\models\CosyVoice2-0.5B`. The model card and source repository declare Apache-2.0. The model download is about 4.85 GB. The local adapter has prepared a cloned-voice profile and generated WAV samples with CUDA on the GTX 1660 Super. A 6 GB GPU is a tight fit; longer scripts and repeated use can take several minutes.

The configured `workers\cosyvoice2_worker.py` prepares each voice sample as a 24 kHz WAV prompt, downmixing to mono and retaining at most the first 28 seconds (CosyVoice 2's speech-token extractor rejects prompts longer than 30 seconds). The TTS stage loads CosyVoice 2 and performs zero-shot, instruction-conditioned speech generation. This setup does not require a separate transcript field. OGG decoding is handled by the CosyVoice environment's SoundFile/Torchaudio backend.

The development server's reload watcher is limited to `app\` and `static\` so model downloads and environment files under `data\` do not continuously restart the server.

Example configuration (replace the paths and worker scripts with your own tested adapters):

```json
{
  "voice_clone": {
    "command": [
      "C:\\avatar-envs\\cosyvoice\\python.exe",
      "C:\\avatar-workers\\voice_clone_worker.py",
      "--job-spec",
      "{spec}"
    ],
    "working_directory": "C:\\avatar-workers",
    "timeout_seconds": 1800
  },
  "tts": {
    "command": [
      "C:\\avatar-envs\\cosyvoice\\python.exe",
      "C:\\avatar-workers\\tts_worker.py",
      "--job-spec",
      "{spec}"
    ],
    "working_directory": "C:\\avatar-workers",
    "timeout_seconds": 900
  },
  "motion": {
    "command": ["C:\\avatar-envs\\liveportrait\\python.exe", "C:\\avatar-workers\\motion_worker.py", "--job-spec", "{spec}"],
    "timeout_seconds": 1800
  },
  "lip_sync": {
    "command": ["C:\\avatar-envs\\musetalk\\python.exe", "C:\\avatar-workers\\lip_sync_worker.py", "--job-spec", "{spec}"],
    "timeout_seconds": 1800
  },
  "qc": {
    "command": ["C:\\avatar-envs\\qc\\python.exe", "C:\\avatar-workers\\qc_worker.py", "--job-spec", "{spec}"],
    "timeout_seconds": 900
  },
  "post": {
    "command": ["C:\\avatar-workers\\post_worker.exe", "--job-spec", "{spec}"],
    "timeout_seconds": 900
  }
}
```

Each job spec is JSON and contains the inputs and outputs for that stage. The `voice_clone` worker receives the voice sample and output paths; it does not receive consent text or a consent recording. The worker must write a reusable JSON voice-profile artifact to `output_profile`; it must include a non-empty `format` string identifying the model/artifact format and a non-empty `profile` object containing the model-specific speaker profile/embedding reference. The TTS worker receives one script segment, language/style, voice-sample path, reusable `voice_profile` artifact path, and `output_audio`; it must write a non-empty WAV. The still-photo lip-sync worker receives the reference image and approved segment audio and must write `output_video` as MP4. The QC worker receives the segment video paths and must write `qc_report`, a JSON object with `"status": "pass"` or `"review_needed"` and optional `"flags": ["..."]`. A `review_needed` report pauses before finishing and presents the flags and generated segment previews for user review. The post worker receives all segment paths and must write the final MP4 to `output_video`.

The orchestrator serializes configured worker invocations so two GPU-heavy stages do not overlap. A worker process exits after each invocation, which releases its process-owned model state. Worker logs and job specs are retained under the local job folder. A failed job can be retried; valid completed audio/video segments are reused. API preflight checks prevent starting voice/video work when its required stage entries are missing from the worker configuration. This checks configuration presence, not that a model loads or inference succeeds.

## Included workflow

1. Create a visual avatar with a reference photo and face-consent record. Replace the reference photo from the Avatars screen when needed.
2. Separately create a voice profile with a voice sample. The local clone worker prepares a reusable profile.
3. Paste a finished script, select an avatar photo, a ready cloned voice profile, and a style; generate per-segment speech.
4. Listen to each segment, regenerate it if needed, and approve all voice before video starts.
5. Follow job progress over SSE; cancel between worker steps.
6. If QC requests review, inspect the avatar video segments and flags before allowing finishing to continue.
7. Review the finished video and downloadable provenance sidecar.

`GET /api/jobs/{id}/events` emits SSE events. The API also exposes job, segment audio/video, result, and system-status endpoints in `/docs`.

## Current boundaries

- CosyVoice 2 voice-profile preparation and TTS have generated real WAV output with CUDA. MuseTalk, QC, and FFmpeg finishing worker code exists; all configured MuseTalk model files are present, and key MuseTalk dependencies import in the configured CUDA environment. Lip-sync quality and final MP4 finishing still need end-to-end review.
- No model-specific VRAM presets or precision toggles are exposed yet. The GPU feasibility spike must establish safe settings first.
- The browser is the UI for this local single-user app. It does not include authentication or LAN access controls.
- MuseTalk CUDA fit, peak VRAM, speed, lip-sync quality, and output identity remain unverified. Review generated speech and video output before using them.

See [IMPLEMENTATION_CHECKLIST.md](./IMPLEMENTATION_CHECKLIST.md) for the verified progress and remaining gates.
