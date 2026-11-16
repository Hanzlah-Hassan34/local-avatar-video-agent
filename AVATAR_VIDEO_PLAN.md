# Still-photo talking avatar implementation plan

## Goal

Generate a downloadable MP4 from the user's own portrait and own cloned voice. The portrait should stay visually still except for mouth movement synchronized to the approved narration. Keep processing local where practical and retain the app's voice review, progress, retry, and provenance features.

## Planning checklist

### Setup and feasibility

- [x] Inspect the assignment app, existing avatar/voice records, GPU, model folders, and worker setup.
- [x] Repair the app virtual environment and point voice workers at this assignment directory.
- [x] Rebase saved avatar and voice sample paths into this assignment's `data/` folder; keep a database backup.
- [x] Confirm CUDA-enabled PyTorch can see the GTX 1660 Super in the CosyVoice environment.
- [x] Prepare a cloned-voice profile and generate a short WAV sample.
- [ ] Get the user's attached portrait into a local file and set it as the avatar reference.
- [x] Install and verify the needed OpenCV, Accelerate, matching torchvision, and bundled FFmpeg packages in the CosyVoice environment; key MuseTalk diffusers/Whisper imports also succeed there under CUDA.
- [ ] Finish an isolated MuseTalk environment if first inference reveals a package-version conflict. The configured worker currently uses the verified CUDA CosyVoice environment; no separate environment install is running.
- [x] Download the MuseTalk v1.5 UNet, config, SD VAE, Whisper, and face-parser checkpoints.
- [ ] Measure a real short lip-sync run: VRAM, runtime, synchronization, and identity preservation.

### Still-photo video workflow

- [x] Remove the required driving-video/motion step from the main video path.
- [x] Add avatar photo replacement while preserving the existing avatar consent record.
- [x] Update the Studio flow to describe mouth-only photo lip sync.
- [x] Add worker code for MuseTalk invocation, technical review flags, and MP4 stitching; point the worker config at those stages.
- [x] Implement the MuseTalk worker adapter and configure its stage entry; runtime dependencies and checkpoint are still missing, so inference is not operational.
- [x] Implement technical QC and MP4 finishing/segment stitching worker code and configure their stage entries; runtime validation is still outstanding.
- [ ] Verify the full app flow from approved speech through downloadable video and provenance.

### Documentation and handoff

- [ ] Update the README and implementation checklist with working setup commands and measured results.
- [ ] Document model/checkpoint versions, licenses, VRAM/runtime, and any quality limits.
- [ ] Provide a final verified sample MP4 and reproducible local instructions.

Checklist items marked complete indicate code/setup changes or observations already made; they do not imply the complete video pipeline has been verified.

## Current starting point

- The FastAPI app, separate avatar and voice-profile records, script segmentation, voice review, job tracking, and worker subprocess contract are implemented.
- The app venv now runs on Python 3.12, and its SQLite avatar/voice sample paths point inside this assignment folder. An original database backup is kept at `data/jobs.sqlite3.pre-path-repair.bak`.
- The cloned voice profile was prepared successfully, and CosyVoice generated a short PCM 16-bit WAV using CUDA. The sample job is at voice review.
- The app's video path and Studio UI were changed to remove the mandatory motion/driving-video step. Avatar photo replacement now preserves the existing consent record.
- MuseTalk source was cloned and its inference script was adjusted to invoke FFmpeg with argument arrays, including paths with spaces. The app has a MuseTalk worker adapter. Required v1.5 UNet, config, SD VAE, Whisper, and face-parser files are present. OpenCV, Accelerate, CUDA PyTorch/torchvision, diffusers, and Whisper imports succeed in the configured CosyVoice environment; no lip-sync inference has run yet.
- QC and post-processing worker code and config entries are present. Their runtime behavior is not verified. FFmpeg is not on PATH; post-processing uses the imageio-ffmpeg bundled executable. The concat manifest uses a named file beside the output and removes it after processing.
- The user attached a suitable portrait in chat, but it is not present as a local project file yet. The app now has a photo-replacement UI/API; upload it there to make it the avatar source.
- The NVIDIA GeForce GTX 1660 SUPER has 6 GB VRAM. CosyVoice CUDA works; MuseTalk fit and output quality remain unverified.

### Initial feasibility findings

- `nvidia-smi` sees an NVIDIA GeForce GTX 1660 SUPER with 6,144 MiB VRAM and 5,213 MiB free at the time of inspection (driver 616.92).
- FFmpeg is not found on PATH.
- The app `.venv` points to the missing base interpreter `C:\Python313\python.exe`, so that environment cannot currently start.
- The local CosyVoice environment is Python 3.10, but the saved voice-clone log shows `ModuleNotFoundError: No module named 'torch'`.
- Initial inspection found worker and model paths outside this assignment directory; those local paths were rebased before voice generation. Machine-specific worker settings remain local in ignored `data\workers.json`.
- The model weights folder exists, but no successful real voice or video inference is recorded.

## Implementation stages and exit checks

### 1. Feasibility checks

- Inspect GPU driver, CUDA visibility in the isolated model environments, free disk space, and FFmpeg availability without disturbing the current app environment.
- Run a short real CosyVoice clone/TTS sample if the model files and environment are complete. Confirm the WAV is valid and recognizably uses the supplied voice.
- Evaluate MuseTalk as the first lip-sync candidate: create a short static video from the portrait, synchronize the sample narration, and inspect mouth alignment, identity preservation, runtime, and peak VRAM.
- Check upstream code, checkpoint, and dependency license terms before adopting a model. MuseTalk's official repo is the initial candidate; its upstream installation recommends Python 3.10/CUDA 11.7 and describes audio-driven lip-sync. Compatibility with this 6 GB GPU is not assumed.
- If MuseTalk cannot fit/run acceptably, compare a lower-resource alternative and report the tradeoff before changing the model direction.

**Exit:** one verified sample voice WAV and one short portrait lip-sync MP4, or a documented concrete blocker and fallback recommendation.

### 2. Simplify the video pipeline for a still portrait

- Pass the still portrait image directly to the configured lip-sync worker with approved audio. MuseTalk's image input path is used; no separate static base clip is required.
- Do not require an uploaded driving video for this mode.
- Preserve the existing motion workflow only if it remains useful as an optional future mode; do not make it part of the target flow.
- Add FFmpeg-based audio/video muxing and segment stitching with consistent dimensions, frame rate, and audio settings.

**Exit:** one segment can run image → lip sync → valid MP4, without a driving clip.

### 3. Integrate with the app

- Update API preflight, job state/progress, worker specs, retry handling, cancellation boundaries, and provenance for the still-photo path.
- Update the Studio and avatar UI so the user selects a photo and cloned voice, enters a script, reviews the voice, then generates the talking photo without uploading a driving video.
- Keep generated speech review before video generation, and expose segment previews and the final MP4 download.
- Surface clear errors when a model stage is unavailable or inference fails; configuration presence must not imply model verification.

**Exit:** the existing app can complete the target user flow end-to-end with configured real workers.

### 4. End-to-end quality and handoff

- Run a short real sample through voice generation, manual voice approval, lip sync, finishing, and download.
- Check audio/video duration agreement, MP4 playback, visible lip timing, image identity, and that head/body motion stays minimal.
- Record measured processing time and peak VRAM, exact model/checkpoint versions, licensing notes, setup/run instructions, and limitations in the README/checklist.

**Exit:** reproducible local instructions and a verified sample artifact exist; remaining known limitations are documented.

## Expected effort

Estimate roughly 1–3 focused days for a basic demo if model inference works on the available GPU; several additional days may be needed for model incompatibility, VRAM limits, or quality fixes. The feasibility stage determines the actual schedule.

## Model direction and constraints

- First candidate: [MuseTalk](https://github.com/TMElyralab/MuseTalk), because its upstream describes audio-driven lip synchronization over an input face/video and the target requires mouth movement while keeping the rest nearly still.
- Avoid relying on the original [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) pretrained release for this implementation unless its terms fit the intended use; its upstream states personal/research/non-commercial restrictions.
- Use only the user's own image and voice sample or assets for which they have permission. Verify model, checkpoint, and dependency terms before enabling a worker.
- Do not promise 6 GB GPU compatibility until the short inference test passes.
