# Tone Twin

A small guitar tone learning studio: record paired dry/processed audio from a Quad Cortex Mini, train an official Neural Amp Modeler A2 model on Modal, and compare it against the recorded reference.

## Studio

[Open Tone Twin](https://drchrislevy--tone-twin-web.us-east.modal.run)

The Blackroom UI has three views: Record, Train, and Compare. Train shows live batch-loss and per-epoch validation curves, progress, and the best validation epoch. Charts have separate scales because training uses a waveform/spectral objective while validation uses ESR. Toggle logarithmic/linear axes, hover or use arrow keys to inspect measurements, and return to the saved curves after training. Older runs show the validation history they recorded; training values are never invented. Source setup, file uploads, and training options are in Settings. Previous takes opens the recording library. Synthetic demo data is explicitly labelled.

From this project directory, connect your Quad Cortex Mini over USB and run:

```sh
uv run --env-file .env tone-twin bridge --url https://drchrislevy--tone-twin-web.us-east.modal.run
```

Then record in the browser, train, and switch between dry, reference, and model playback. Download the exported `.nam` to use in a compatible NAM player. Keep the bridge running while recording; Ctrl+C disconnects it. The local `.env` already contains this studio's private workspace key and is excluded from Git.

## Local setup

```sh
uv sync
uv run tone-twin devices
uv run tone-twin serve
```

Start a bridge in a second terminal after connecting the Quad Cortex Mini:

```sh
uv run tone-twin bridge --url http://127.0.0.1:8000
```

The bridge selects a Quad Cortex by name, never the laptop microphone automatically. By default it captures USB channels 1 (dry) and 3 (processed) at 48 kHz. Check the Mini's dry/wet swap setting. Use `--dry-channel`, `--wet-channel`, and `--device` to choose explicitly. Channels are numbered from 1. Monitor through the Mini's own outputs.

For a hardware-free transport check, the explicit `--synthetic` flag generates a labelled synthetic source. No audio input is opened.

## Hosting on Modal

`modal_app.py` hosts the web app and a separate GPU training function, preferring H100 with an A10G fallback. H100 was faster in the training benchmark, but has a higher per-second price. CPU memory snapshots preload PyTorch and NAM imports for subsequent cold starts; the first snapshot creation still takes extra startup time. Recordings, training checkpoints and models live in the `tone-twin-data` Modal Volume. Both services scale to zero when idle. A running training job has a one-hour timeout and training concurrency is limited to one.

The GPU worker has `min_containers=0` and a 1,200-second idle window (20 minutes, Modal's documented maximum), increased from the default 60 seconds. Idle GPU time is billable. Starting a recording asynchronously warms that same training function and initializes CUDA; periodic requests keep it ready during capture. Warm-up failures do not interrupt recording. Warm-up requests stop when recording ends, disconnects, or reaches the ten-minute recording limit. Opening the studio or monitoring alone does not warm the GPU. The web container uses a separate 120-second idle window.

One web container owns the active audio connection and browser controls. This initial single-user design needs no sticky-session routing. The bridge sends ordered 48 kHz PCM pairs over WebSocket; sequence gaps stop a take and mark it interrupted. Recordings survive web-container restarts after saving. A container interruption during recording can lose the unsaved take.

Training publishes actual batch measurements about every two seconds and after each validation epoch. The browser polls once per second during active jobs. Up to 2,000 training points are retained; longer histories thin older measured points without smoothing their values. Completed histories remain in the job status on the Volume.

Progress snapshots are written by one background writer, with at most one pending snapshot. Each contains the full history; intermediate snapshots can be combined without losing measurements. Completion waits for the final write and Volume commit. Loss values cross from GPU to CPU together at publication time, and best-checkpoint files stay on local disk until the winning checkpoint is copied to the Volume at the end. Interrupted runs cannot resume from those temporary checkpoints.

Deploy only when recording and training are idle. Open WebSockets can prevent a rolling deployment from replacing the sole web container. For an intentional restart use `uv run modal deploy --strategy recreate modal_app.py`, then reconnect the audio bridge. If the same version is already deployed, use `uv run modal app rollover tone-twin --strategy recreate`.

For a new deployment, create an untracked `.env` with a random `TONE_TWIN_KEY`, then:

```sh
uv run modal secret create tone-twin-auth --from-dotenv .env
uv run modal deploy modal_app.py
```

The hosted UI asks for that workspace key. The bridge reads it from the `TONE_TWIN_KEY` environment variable; keep it out of URLs and source control. Local training is deliberately unavailable unless the real NAM training dependencies and worker have been configured. The local UI does not simulate a successful training run.

## Training

Uses Steven Atkinson's `neural-amp-modeler==0.13.0` and its canonical A2 recipe, with Lite, Full, or both model sizes. GPU dependencies are installed in the Modal image, not the local capture environment. Training checks silence, clipping and timing; reserves a chronological validation segment; and exports a `.nam`, measured validation error and synchronized comparison audio. Start with one fixed amp/drive/cabinet preset and a few minutes of varied playing. This initial trainer uses one dry and one processed channel.

New runs default to 100 epochs, matching NAM's official full-trainer default. Shorter 10/25/50-epoch runs remain available for quick experiments. More epochs allow further optimization, but the best validation checkpoint is still selected rather than assuming the final epoch sounds best.

Training keeps the upstream waveform/spectral loss, FP32 precision, batch size 16, and learning-rate schedule. The complete aligned audio pair stays in GPU memory (about 69 MB for three minutes), and batches are gathered on the device. CUDA graphs replay NAM's forward/loss/backward operations, with a separate graph for the last smaller batch; Adam uses its fused CUDA implementation. Graph wrappers do not change exported weights or checkpoint keys. Validation uses total squared error divided by total reference energy over the entire held-out recording, including the final partial chunk. This avoids near-silent chunks dominating an average of individual ESR ratios. Only ESR is computed during validation; unused spectral losses are skipped. The best validation checkpoint is exported, which may be earlier than the final epoch. Existing runs retain their original metrics and models; new runs identify this metric as `whole_recording_esr`.

Batch size 64 and a larger learning rate were benchmarked on a real capture but produced worse validation error for little additional speed, so the default remains 16. BF16 was slower for this network. Inductor compilation added about 28 seconds of setup in the microbenchmark, so the runtime uses CUDA graph capture without compiling new kernels. Training configuration records the GPU and execution settings; completed status includes total worker time and peak allocated GPU memory. Platform queue/cold-start time is additional.

On a 178.7-second real capture with Lite/25 epochs, a same-H100 comparison measured 56.0 seconds for the old worker versus 14.1 seconds for the optimized worker, including preparation/export but excluding container startup. Median epoch time fell from 1.85 to 0.41 seconds. The deployed verification completed training in 12.2 seconds, worker processing in 16.7 seconds, and submission-to-result in 39 seconds including first-time CPU snapshot initialization. Peak allocated GPU memory was 309 MB. Timings vary by GPU availability, capture, model size, and cold-start state; numerical regression checks do not guarantee equivalent perceptual quality for independently trained runs.

Job status includes `timings_seconds` for reading audio, quality checks, alignment, model/data setup, fitting, and rendering/export. On a warm H100 with that capture, reading took 0.24 seconds, checks 0.11 seconds, and alignment 0.11 seconds: about half a second combined. Alignment still uses only the training portion of the finished take, preserving the held-out validation segment. Capture already checks packet continuity, finite samples, and levels; there is no denoising, EQ, or independent loudness normalization before training.

The 12-second minimum is for pipeline experiments, not a guarantee of a useful model. Include quiet and hard picking, sustained notes, and different parts of the neck. Turn off delay, reverb, and modulation for the first captures. Compare uses the held-out end of the recording and a shared playback gain. It does not yet perform live model inference on new guitar playing. Continue monitoring through the Mini while capturing.

Original upstream: https://github.com/sdatkinson/neural-amp-modeler

## Checks

```sh
uv run pytest -q
```

36 local tests plus 6 subtests cover upload/authentication, dataset validation, timing alignment, capture framing, background progress writing, and nonblocking worker warm-up with failure/disconnect handling. Seven additional runtime regression tests require the NAM training image (and are skipped locally without NAM/PyTorch); they cover silence, chunk boundaries, partial tails, packed models, batch equivalence to NAM, shuffled coverage, and graph/eager loss and gradient agreement. GPU checks exercise Lite on a real three-minute capture and Full/Both on synthetic audio, verifying export, live measurements, and agreement between checkpoint ESR and rendered audio. These numerical checks do not establish perceptual tone quality.
