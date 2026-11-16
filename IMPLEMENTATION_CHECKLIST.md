# Implementation checklist: local AI avatar video agent

Built for Windows and a GTX 1660 Super (6 GB). Real-model compatibility remains gated on the feasibility spike.

## Workspace progress

- Confirmed Windows, Python 3.12.10, and an NVIDIA GTX 1660 Super with 6 GB VRAM.
- The app environment has PyTorch `2.14.1+cpu`; CUDA is unavailable to that Python runtime, so GPU inference has not been tested there.
- FFmpeg is not available on `PATH`.
- Built a FastAPI + SQLite app with separate face avatars and consented voice profiles, script voice selection, review gates, a library, progress events, and subprocess worker contracts.
- The CosyVoice 2 adapter and local `voice_clone`/`tts` worker configuration are present, but a real inference has not passed. The model environment and weights must be validated before use.
- Only the fake voice-clone + TTS + motion + lip-sync + QC + post pipeline has passed end-to-end tests. Real motion, lip-sync, QC, and post workers are not configured.
- Intended use is personal-only until all model and weight licenses are verified. Model downloads are handled outside the app.

### Upstream video-model evidence

- [MuseTalk upstream README](https://github.com/TMElyralab/MuseTalk) documents Windows inference and reports a Windows test on an RTX 3050 Ti Laptop GPU with 4 GB VRAM: fp16 generation of an 8-second video took about 5 minutes. This is not a benchmark or guarantee for the GTX 1660 Super.
- [LivePortrait upstream README](https://github.com/KwaiVGI/LivePortrait) documents Windows support, but gives no target-GPU VRAM/performance result. Its [license](https://github.com/KwaiVGI/LivePortrait/blob/main/LICENSE) says the code is MIT while InsightFace detection models are restricted to non-commercial research use.
- The selected GPU, model weights, dependency licenses, and output quality still require a local feasibility run before those stages are enabled.

## 0. Define the first release

- [x] Limit the MVP to: reference image + separately cloned voice profile + finished script → reviewed voice segments → talking-avatar video.
- [x] Exclude script writing, B-roll, live chat, and multi-user support.
- [x] Choose the initial platform: Windows, Python 3.12, FastAPI, SQLite, and a lightweight HTML/JavaScript frontend.
- [x] Set intended use to personal-only for now.
- [ ] Verify and record licenses for every selected model, weight, and dependency before enabling that worker.
- [x] Require face consent for avatars; voice profiles no longer collect consent text, recordings, or a permission checkbox.
- [ ] Set measurable quality targets: maximum segment length, acceptable lip-sync quality, output resolution, and processing time.

**Exit gate:** MVP and consent scope are defined. The model-license and quality policy is still open.

## 1. Feasibility spike — do this before using real models

- [ ] Create isolated environments and minimal test scripts for each candidate model.
- [ ] Verify CUDA-enabled inference on the target GPU and record peak model VRAM; only the GPU/driver and CPU-only PyTorch runtime have been inspected.
- [ ] Test voice cloning candidates with the same sample text and voice recording.
- [ ] Test LivePortrait with an approved reference image and calm driving clip.
- [ ] Test MuseTalk with a short video and generated audio, including intended precision settings.
- [ ] Measure peak VRAM, seconds per second of video, crashes, and output quality.
- [ ] Verify every selected model’s license and usage restrictions.
- [ ] Decide whether the pipeline can run on 6 GB, needs CPU/offload options, or requires a fallback.

**Exit gate:** Still blocked on CUDA-enabled model tests on the target machine.

## 2. Create the app and worker contracts

- [x] Set up the FastAPI backend, SQLite database, local configuration, and Studio UI.
- [x] Define shared job and segment states, explicit errors, progress events, and cancellation between worker steps.
- [x] Define JSON job files and input/output paths for subprocess workers.
- [x] Add a worker runner that captures logs, reports failures explicitly, and enforces timeouts.
- [x] Serialize worker invocations; each worker process exits after its stage.
- [x] Add local job storage, restart recovery, retries, and reuse of validated segment outputs.

**Exit gate:** Verified with fake workers through audio review, video stages, QC, post, and provenance.

## 3. Implement voice generation and review

- [x] Separate face-avatar records from voice profiles; avatar voice samples are no longer used for narration.
- [x] Add a voice library to create voice profiles from a sample without consent fields in the form or API.
- [x] Add a `voice_clone` subprocess contract, profile artifact validation, retry, and ready/failed state.
- [x] Let each script job select a ready cloned voice profile independently from its visual avatar.
- [x] Record the CosyVoice 2 source/model card's stated Apache-2.0 license in the README; verify all dependency/weight terms before enabling broader use.
- [x] Implement a CosyVoice 2 adapter for prompt preparation and instruction-conditioned TTS; real model inference remains untested.
- [ ] Validate the CosyVoice 2 voice-profile and TTS stages on CUDA, including peak VRAM and output quality.
- [x] Pass the selected voice profile artifact and source sample to the configurable TTS subprocess contract; validate generated WAVs.
- [x] Parse and segment scripts, preserving punctuation and sensible sentence boundaries.
- [x] Orchestrate one audio output per segment and persist its review state and attempt count.
- [x] Add the review screen: listen, regenerate a segment, approve, or cancel.
- [x] Validate audio structure and non-empty output before advancing.
- [ ] Run voice review with a real cloned-voice model.

## 4. Implement motion and lip sync

- [x] Define a motion-worker contract with reference image, driving clip, and output validation.
- [ ] Implement and run the LivePortrait model worker.
- [ ] Decide how the driving clip is selected, looped, or extended; test transitions between segments.
- [x] Define a lip-sync worker contract that takes the motion video and approved segment audio.
- [x] Process segments independently and reuse validated completed outputs on retry.
- [x] Isolate stages in subprocesses so each model process exits before the next invocation.
- [x] Preserve per-segment worker logs, attempts, and errors for diagnosis.
- [ ] Measure timing, VRAM, and output quality for real model workers.

## 5. Add quality checks and finishing

- [x] Define a QC worker contract and validate its machine-readable report.
- [ ] Implement speech recognition, voice-match, and face-drift checks with verified license suitability.
- [x] Surface configured QC review flags and pause for human review when requested; fail explicitly for invalid reports.
- [ ] Stitch segments, normalize loudness, and create the chosen aspect ratio with FFmpeg.
- [ ] Add optional face restoration/upscaling only if it fits the measured GPU budget and license policy.
- [x] Write a provenance sidecar and disclose AI generation in the result UI.

## 6. Build the UI

- [x] Implement Studio, Voice Review, Progress, Result, Avatars, Voices, Library, and System screens.
- [x] Connect progress updates through SSE, including retries, failures, and cancellation.
- [x] Show model/CUDA status and worker configuration without implying unsupported hardware capability.
- [x] Apply the warm ivory, sage, and clay visual system; avoid blue tones.
- [ ] Keep fonts and app assets local if the app is intended to work offline.

## 7. Harden and package

- [x] Handle interrupted jobs, retries, and cancellation; test invalid uploads and missing workers.
- [x] Add input validation and explicit errors for unsupported formats or missing workers.
- [x] Preflight TTS and video/QC/post worker stages before queueing work; expose stage configuration separately from inference verification.
- [x] Remove voice consent fields from the voice-profile UI and API; retain the separate avatar face-consent flow.
- [ ] Add model download and integrity checks, with clear disk-space and license notices.
- [ ] Provide per-model VRAM/precision presets based on measured results.
- [x] Document installation, GPU prerequisites, worker setup, consent handling, and known limitations.
- [ ] Test a clean install and complete real-model sample job on the target Windows machine.

**Overall status:** Voice profiles are separate from avatars and selectable for scripts. CosyVoice 2 adapter/configuration is present, but no real inference has been verified. Motion, lip-sync, QC, and post-processing remain contracts only; the full pipeline has only been exercised with fake workers. CUDA/VRAM feasibility and FFmpeg finishing remain incomplete. Checked adapter/configuration items do not imply that a model has run.
