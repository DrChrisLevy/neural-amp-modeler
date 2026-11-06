# Neural Amp Modeler

Capture an edited Quad Cortex Mini preset, train a Neural Amp Modeler on Modal, and compare the result. [Open the studio](https://drchrislevy--neural-amp-modeler-web.us-east.modal.run).

## Your first capture

The standard input WAV plays through your chosen preset for **3 minutes 10 seconds**. You do not need to play guitar. On a fresh checkout, complete the one-time setup below first.

1. **Prepare the Mini.** Connect it by USB, choose a preset and scene, and disable delay, reverb, and modulation. **Unsaved edits (the star beside the name) are supported.** Quit Cortex Control, stop the Neural Amp Modeler live bridge and other recording/playback software, and set the Mini's USB assignment to **1/2 dry, 3/4 wet**.

2. **Run the capture in Terminal.** Change the name to describe your sound:

   ```sh
   cd /Users/christopher/personal/neural-amp-modeler
   uv run --extra capture neural-amp-modeler capture --name "My preset - no delay or reverb"
   ```

   Leave the preset and controls unchanged until the command finishes—roughly 3:10 plus setup. It plays the test WAV, records the processed sound, and restores the original input routes.

3. **Open the saved captures in Finder:**

   ```sh
   open .data/pairs
   ```

   Open the newest timestamped folder. You will upload `input.wav` and `output.wav` from that same folder.

4. **Upload to [Neural Amp Modeler](https://drchrislevy--neural-amp-modeler-web.us-east.modal.run/).** The studio opens on **Your Library**. Click **Add pair**, name the pair, and choose:

   | App field | File |
   | --- | --- |
   | Input · dry WAV | `input.wav` |
   | Output · processed WAV | `output.wav` |

   Click **Add pair**. The dialog closes immediately and the new row shows upload progress, then processing. You can add more pairs while earlier uploads continue. Failed uploads can be retried on their row without duplicating a saved pair. Keep the browser tab open until uploads finish. You do not need to connect the live bridge for this workflow.

5. **Train and compare.** Click **Train** on the pair’s row. Training uses **A2 Full / 400 epochs**; adjust the epoch count, then click **Start training**. Watch the live loss curves, then choose **Compare audio**. Dry, Reference, and Model appear as three aligned waveform rows. Click or drag a waveform to select its audio and start playing from that position. Click a row’s label to switch audio without changing the playback position. Use the small play/pause button or **Space** to pause and resume. Back in the library, click **View runs** to revisit previous runs. **New training run** saves another version from the same pair; no new recording is needed.

Use the trash icon on a library row to delete that pair and all its training runs, models, and comparison audio. Deletion asks for confirmation and is unavailable while that pair is training.

For the next preset, select and edit it on the Mini, then repeat the capture command with a new name. Each capture gets its own folder; existing captures are never overwritten.

## One-time setup

Run from the repository on a new setup. If the dependencies and standard input are already installed, skip this:

```sh
cd /Users/christopher/personal/neural-amp-modeler
uv sync --extra capture
uv run neural-amp-modeler input
```

The input command downloads and verifies the standard NAM WAV at `.data/input.wav`. On another computer, replace the `cd` path with your checkout location.

## Capture files and options

The command temporarily routes Input 1 lanes from USB 5, plays the WAV at its original level, records USB 3/4, and restores the original input routes. It **never recalls, changes scenes, saves, or reloads a preset**. Leave the device controls unchanged during capture. Every attempt has its own timestamped directory under `.data/pairs/`:

```text
20261105-...-brit-crunch-no-reverb/
  input.wav       Original dry signal
  output.wav      Matching mono processed signal, ready to upload
  raw.wav         Untrimmed stereo return, including pre-roll and tail
  metadata.json   Edited preset snapshot, gains, routing, checksums, result
```

Before saving `output.wav`, the capture command measures the recording latency and trims the archived return to match the input. It uses the real recorded tail, preserving the full file length without adding silence. Standard NAM captures retain its one-sample causal margin and must pass pulse timing and repeated-passage checks. The input stays byte-for-byte unchanged; the mono output stays 48 kHz / 24-bit at its recorded level. `metadata.json` records the timing correction.

`--wet-channel 4` selects the right processed channel instead of the default left. `--input /path/to/input.wav` chooses another mono 48 kHz source, between 12 seconds and 10 minutes. Custom inputs use correlation with a small causal margin; ambiguous timing fails the export. The standard NAM input is recommended. Neither file is normalized or resampled.

### Repair an older capture

Older captures may contain enough USB buffering delay to fall outside another trainer's pulse search. Export a corrected copy from the original capture folder, which must still contain `input.wav`, `raw.wav`, and `metadata.json`:

```sh
uv run neural-amp-modeler repair-capture .data/pairs/20261105-221012-347018-inferno \
  --output "$HOME/Downloads/Inferno - aligned"
```

Choose a new output folder; an existing folder is never overwritten. This runs offline, verifies the archived file checksums, and leaves the originals intact. No Mini connection or new recording is needed. Upload the corrected `input.wav` and `output.wav` together to Neural Amp Modeler or a dry/wet trainer. For a sweep workflow using the same standard NAM v3 input, upload the corrected `output.wav` as the recorded response.

To check the current preset without playback or routing changes:

```sh
uv run --extra capture neural-amp-modeler capture --inspect
```

Automatic routing uses the pinned, unofficial [qc-mcp](https://github.com/lexasoft123/qc-mcp/tree/f634ecd3dcf426090b90bc4cc176e4b500494b9e) protocol implementation and is limited to macOS / Mini CorOS 4.1.1. The narrow adapter was adapted from the decision-model project's implementation; it has no runtime dependency on that project. Ctrl-C attempts routing restoration. Disconnecting USB or force-killing the process can prevent it; the metadata records whether restoration completed.

## Training options

Every new run trains **A2 Full**, with one Full model in the downloaded `.nam` file. Click **Train** on a pair’s library row to choose 50, 100, 400, 500, or 1,000 epochs; the default is **400**. Full prioritizes capture accuracy for desktop playback. Lite is useful for constrained playback hardware, but is not offered for new runs in this app.

Every pair can have multiple runs, which can train concurrently, including different epoch settings for the same pair. Each run gets its own GPU container; the app sets no GPU-container cap, while Modal workspace quotas and capacity still apply. The collapsed row shows the active or latest run’s model and epochs. Expand a row’s training history to see every run’s settings and result, then open its curves and comparison audio. Completed runs have a **Download .nam** link directly in their library row and on the training-results page. More epochs don’t guarantee a better sound: export uses the best validation checkpoint.

NAM downloads use the pair’s name and the run’s epoch count: a 400-epoch run for **Victor’s Mega Squid** saves as `victors_mega_squid_400_epochs.nam`. Input and output audio save as `victors_mega_squid_dry.wav` and `victors_mega_squid_wet.wav`.

Existing Lite and Full + Lite runs remain available with their original labels, curves, and downloads. Older Full + Lite runs contain both sizes in one packed `.nam` file; their curves and best-checkpoint selection use the average across both models, and Compare plays Full. New Full runs report only the Full model’s loss and validation score.

## Training details

The worker pins `neural-amp-modeler==0.13.0` and uses its A2 networks, waveform/spectral objective, batch size 16, FP32, and learning-rate schedule. Training starts from scratch.

- The unmodified standard NAM v3 input is recognized by its audio checksum. The trainer still runs upstream's timing-pulse calibration and consistency check on every pair, including aligned exports. For older or externally recorded files with buffered USB latency, both pulses must agree within 20 samples in an extended 500 ms window before upstream calibration is retried. Training does not rewrite uploaded WAVs. Training covers 10–181 seconds; the final nine seconds are validation. The repeated opening passage is excluded from training.
- Other recordings use a chronological 80/20 split and correlation-based timing. Each partition is aligned separately with its own model context.
- Upstream's normalization sets training-target RMS to −18 dBFS using **training data only** and applies that same gain to validation. The export hook compensates the model's output gain. Source WAVs stay unchanged; comparison audio is restored to the original level.
- Audio stays on the GPU; CUDA graphs reduce dispatch overhead. The runtime scores every validation sample after model context and selects the best whole-recording ESR. Progress reports actual batch losses and per-epoch validation.
- The training curve shows the example-weighted average loss for every epoch, with a live running average during the current epoch. Batch history is sampled uniformly across the run so long sessions retain their early measurements. Training loss and validation ESR use different objectives and scales.

The runtime has custom batching, progress, and chunked validation, so it is not the unchanged upstream trainer. Older runs retain their original recipe and scores; they are not directly comparable to new runs. Independent guitar performances and listening tests are still needed to establish perceptual quality.

A fixed amp/drive/cab sound is the intended target. Disable time effects and changing modulation for these captures. Compare plays saved validation audio; live model inference is not implemented.

## Live recording and local development

Live recording is available under **Add pair → Record a live pair**. The Record button stays visible. If the bridge is not connected, clicking it opens the setup instructions; the connection is detected automatically when the bridge starts. The command is:

```sh
uv run --env-file .env neural-amp-modeler bridge --url https://drchrislevy--neural-amp-modeler-web.us-east.modal.run
```

The bridge records USB 1 (dry) and USB 3 (processed); it doesn't play a reamp signal. Don't run the bridge and capture command together. `uv run neural-amp-modeler devices` lists interfaces. For local UI development, see the frontend setup below. Local training is deliberately unavailable without a configured worker.

## Frontend development

The UI uses React, TypeScript, shadcn/ui (Radix), Tailwind CSS, and Recharts. UI source lives in `frontend/`; the Python API serves the built assets from the same origin.

Use Node 22.12+ (Node 24 recommended). Before serving or deploying:

```sh
npm ci --prefix frontend
npm run build --prefix frontend
uv run neural-amp-modeler serve
```

The build writes hashed assets to `src/nam_studio/static/`, which is generated and ignored by Git. For hot reload, keep the Python server running and run `npm run dev --prefix frontend` in another terminal. Open the printed Vite URL; API and live monitoring requests proxy to port 8000. URLs preserve the selected pair, run, and panel.

## Modal

`modal_app.py` defines a web function and an isolated H100/A10G training worker. The GPU scales to zero after 20 idle minutes; web idle timeout is two minutes. Idle GPU time is billable. Live recording warms the GPU; local WAV capture currently does not. Web concurrency is target 16 / maximum 64, with one web container owning the live bridge, so sticky sessions are unnecessary. Recordings and models persist in the `neural-amp-modeler-data` Volume.

The private workspace key is `NAM_STUDIO_KEY` in `.env` and the `neural-amp-modeler-auth` Modal Secret, never in Git. For a new deployment, provision that secret with `uv run modal secret create neural-amp-modeler-auth --from-dotenv .env`. Deploy when recording/training are idle:

```sh
npm ci --prefix frontend
npm run build --prefix frontend
uv run modal deploy --strategy recreate modal_app.py
```

The deployed app is `neural-amp-modeler`. Locally, `NAM_STUDIO_DATA` can override the `.data` folder. The Python package is `nam_studio` (distribution `neural-amp-modeler-studio`), keeping it separate from the upstream `neural-amp-modeler` training library and its `nam` module.

Reconnect the live bridge after deployment. Captures, `.env`, and scratch experiments are ignored by Git.

## Checks

```sh
npm run lint --prefix frontend
npm run build --prefix frontend
uv run --extra capture pytest -q
node --test tests/test_*.cjs
```

Offline tests exercise upload/auth, recording transport, interrupted capture restoration, blocked preset writes, timing/split rules, and responsive storage/streaming. Training-runtime tests additionally require the Modal training image and include normalization/export gain and GPU graph equivalence. Hardware playback is always an explicit capture command, never part of the tests.
