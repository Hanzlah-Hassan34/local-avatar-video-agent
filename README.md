# AI Voice Cloning Studio

A local web app for preparing a voice profile from a reference recording and generating reviewed narration with CosyVoice 2. The app creates a reusable zero-shot voice profile; it does not train a new model.

## Voice workflow

1. Add a voice recording in **Voices**. Use your own voice or a voice you have permission to use.
2. Prepare the profile. The worker converts the recording to a 24 kHz mono prompt and keeps up to 28 seconds.
3. Enter a script, choose the voice and speaking style, and generate narration segments.
4. Listen to each segment, regenerate any you want to change, then approve the audio.

Generated audio and job files are stored locally under `data\`. The app keeps voice profiles separate from avatar photos. Review generated speech before using it.

## Run the web app

On Windows, from this folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload --reload-dir app --reload-dir static
```

Open <http://127.0.0.1:8000>. The server is intended for local use and does not provide user authentication. API documentation is at <http://127.0.0.1:8000/docs>.

## Voice model setup

The app and voice model use separate Python environments:

- App: Python 3.12 or later.
- CosyVoice worker: Python 3.10 with CUDA-enabled PyTorch and the CosyVoice dependencies.
- Model: `FunAudioLLM/CosyVoice2-0.5B` stored locally at `data\models\CosyVoice2-0.5B`.
- Worker source: `data\vendor\CosyVoice` and `workers\cosyvoice2_worker.py`.

Model files, voice recordings, generated audio, and machine-specific worker settings are not included in this repository. Set up the model and worker environment, then create `data\workers.json` with the local Python and source paths. Both `voice_clone` and `tts` use the CosyVoice worker; it selects the task from `AVATAR_STAGE` supplied by the app.

Example worker entries (replace `C:\\project` with this checkout's absolute path):

```json
{
  "voice_clone": {
    "command": [
      "C:\\project\\data\\venvs\\cosyvoice\\Scripts\\python.exe",
      "C:\\project\\workers\\cosyvoice2_worker.py"
    ],
    "working_directory": "C:\\project\\data\\vendor\\CosyVoice",
    "timeout_seconds": 1800
  },
  "tts": {
    "command": [
      "C:\\project\\data\\venvs\\cosyvoice\\Scripts\\python.exe",
      "C:\\project\\workers\\cosyvoice2_worker.py"
    ],
    "working_directory": "C:\\project\\data\\vendor\\CosyVoice",
    "timeout_seconds": 1800
  }
}
```

The voice worker has generated WAV samples using CUDA on a GTX 1660 Super. Initial model loading and longer scripts can take several minutes. GPU memory and speed depend on the model, script length, and other applications using the GPU.

## Avatar support

The app lets you attach or replace an avatar photo. A MuseTalk worker is included for photo lip sync, but its output and end-to-end video pipeline are still under development; this README does not promise a finished avatar video.

## Local data and Git

Uploads, voice profiles, databases, generated jobs, model weights, and worker settings live under `data\` and are ignored by Git. Do not commit recordings or model files. Keep the app bound to `127.0.0.1`; it is not designed for network exposure.
