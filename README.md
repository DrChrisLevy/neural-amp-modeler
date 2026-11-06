# Tone Twin

Capture an edited Quad Cortex Mini preset, train a Neural Amp Modeler on Modal, and compare the result. [Open the studio](https://drchrislevy--tone-twin-web.us-east.modal.run).

## Capture a preset

Run from this repository. Setup and the standard input download are needed once:

```sh
uv sync --extra capture
uv run tone-twin input
```

1. Connect the Mini by USB. Choose a preset and scene on the device, then edit it as you like—for example, bypass delay and reverb. **Unsaved edits (the star beside the name) are supported.**
2. Quit Cortex Control so the script can use USB control. Stop other recording/playback software. Set the Mini's USB assignment to **1/2 dry, 3/4 wet**.
3. Capture the current sound:

```sh
uv run --extra capture tone-twin capture --name "Brit crunch - no reverb"
```

The command temporarily routes Input 1 lanes from USB 5, plays the WAV at its original level, records USB 3/4, and restores the original input routes. It **never recalls, changes scenes, saves, or reloads a preset**. Leave the device controls unchanged during capture. Every attempt has its own timestamped directory under `.data/pairs/`:

```text
20261105-...-brit-crunch-no-reverb/
  input.wav       Original dry signal
  output.wav      Matching mono processed signal, ready to upload
  raw.wav         Untrimmed stereo return, including pre-roll and tail
  metadata.json   Edited preset snapshot, gains, routing, checksums, result
```

Repeat after selecting/editing the next sound. Existing captures are never overwritten. `--wet-channel 4` selects the right processed channel instead of the default left. `--input /path/to/input.wav` chooses another mono 48 kHz source, between 12 seconds and 10 minutes. The source isn't normalized or resampled. Hardware latency is calibrated during training.

To check the current preset without playback or routing changes:

```sh
uv run --extra capture tone-twin capture --inspect
```

Automatic routing uses the pinned, unofficial [qc-mcp](https://github.com/lexasoft123/qc-mcp/tree/f634ecd3dcf426090b90bc4cc176e4b500494b9e) protocol implementation and is limited to macOS / Mini CorOS 4.1.1. The narrow adapter was adapted from the decision-model project's implementation; it has no runtime dependency on that project. Ctrl-C attempts routing restoration. Disconnecting USB or force-killing the process can prevent it; the metadata records whether restoration completed.

## Upload, train, compare

Open **Capture library → Upload audio** in the studio. Choose `input.wav` for **Dry input**, `output.wav` for **Processed tone**, and give the pair a name. Saved pairs remain in the library.

Training settings offer **Full** (default), **Lite**, or **Both**, and 10, 25, 50, 100, 500, or 1,000 epochs. The default is Full / 100 epochs. One job runs at a time. **Train a new version** reuses the same pair; the training-run selector lets you revisit previous models, curves, and comparison audio. More epochs don't guarantee a better sound: export uses the best validation checkpoint.

## Training details

The worker pins `neural-amp-modeler==0.13.0` and uses its A2 networks, waveform/spectral objective, batch size 16, FP32, and learning-rate schedule. Training starts from scratch.

- The unmodified standard NAM v3 input is recognized by its audio checksum. It uses upstream's timing-pulse calibration and consistency check. Training covers 10–181 seconds; the final nine seconds are validation. The repeated opening passage is excluded from training.
- Other recordings use a chronological 80/20 split and correlation-based timing. Each partition is aligned separately with its own model context.
- Upstream's normalization sets training-target RMS to −18 dBFS using **training data only** and applies that same gain to validation. The export hook compensates the model's output gain. Source WAVs stay unchanged; comparison audio is restored to the original level.
- Audio stays on the GPU; CUDA graphs reduce dispatch overhead. The runtime scores every validation sample after model context and selects the best whole-recording ESR. Progress reports actual batch losses and per-epoch validation.

The runtime has custom batching, progress, and chunked validation, so it is not the unchanged upstream trainer. Older runs retain their original recipe and scores; they are not directly comparable to new runs. Independent guitar performances and listening tests are still needed to establish perceptual quality.

A fixed amp/drive/cab sound is the intended target. Disable time effects and changing modulation for these captures. Compare plays saved validation audio; live model inference is not implemented.

## Live recording and local development

Live guitar recording is still available:

```sh
uv run --env-file .env tone-twin bridge --url https://drchrislevy--tone-twin-web.us-east.modal.run
```

The bridge records USB 1 (dry) and USB 3 (processed); it doesn't play a reamp signal. Don't run the bridge and capture command together. `uv run tone-twin devices` lists interfaces. For local UI development, use `uv run tone-twin serve`; local training is deliberately unavailable without a configured worker.

## Modal

`modal_app.py` defines a web function and an isolated H100/A10G training worker. The GPU scales to zero after 20 idle minutes; web idle timeout is two minutes. Idle GPU time is billable. Live recording warms the GPU; local WAV capture currently does not. Web concurrency is target 16 / maximum 64, with one web container owning the live bridge, so sticky sessions are unnecessary. Recordings and models persist in the `tone-twin-data` Volume.

The private workspace key lives in `.env` and the `tone-twin-auth` Modal Secret, never in Git. For a new deployment, provision that secret with `uv run modal secret create tone-twin-auth --from-dotenv .env`. Deploy when recording/training are idle:

```sh
uv run modal deploy --strategy recreate modal_app.py
```

Reconnect the live bridge after deployment. Captures, `.env`, and scratch experiments are ignored by Git.

## Checks

```sh
uv run --extra capture pytest -q
```

Offline tests exercise upload/auth, recording transport, interrupted capture restoration, blocked preset writes, timing/split rules, and responsive storage/streaming. Training-runtime tests additionally require the Modal training image and include normalization/export gain and GPU graph equivalence. Hardware playback is always an explicit capture command, never part of the tests.
