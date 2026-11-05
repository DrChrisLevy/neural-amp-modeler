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

`modal_app.py` hosts the web app and a separate GPU training function, preferring L4 with A10G and T4 fallbacks. Recordings, training checkpoints and models live in the `tone-twin-data` Modal Volume. The single-user web service scales to zero when idle. A running training job has a one-hour timeout and training concurrency is limited to one.

One web container owns the active audio connection and browser controls. This initial single-user design needs no sticky-session routing. The bridge sends ordered 48 kHz PCM pairs over WebSocket; sequence gaps stop a take and mark it interrupted. Recordings survive web-container restarts after saving. A container interruption during recording can lose the unsaved take.

Training publishes actual batch measurements about every two seconds and after each validation epoch. The browser polls once per second during active jobs. Up to 2,000 training points are retained; longer histories thin older measured points without smoothing their values. Completed histories remain in the job status on the Volume.

Deploy only when recording and training are idle. Open WebSockets can prevent a rolling deployment from replacing the sole web container. For an intentional restart use `uv run modal deploy --strategy recreate modal_app.py`, then reconnect the audio bridge. If the same version is already deployed, use `uv run modal app rollover tone-twin --strategy recreate`.

For a new deployment, create an untracked `.env` with a random `TONE_TWIN_KEY`, then:

```sh
uv run modal secret create tone-twin-auth --from-dotenv .env
uv run modal deploy modal_app.py
```

The hosted UI asks for that workspace key. The bridge reads it from the `TONE_TWIN_KEY` environment variable; keep it out of URLs and source control. Local training is deliberately unavailable unless the real NAM training dependencies and worker have been configured. The local UI does not simulate a successful training run.

## Training

Uses Steven Atkinson's `neural-amp-modeler==0.13.0` and its canonical A2 recipe, with Lite, Full, or both model sizes. GPU dependencies are installed in the Modal image, not the local capture environment. Training checks silence, clipping and timing; reserves a chronological validation segment; and exports a `.nam`, measured validation error and synchronized comparison audio. Start with one fixed amp/drive/cabinet preset and a few minutes of varied playing. This initial trainer uses one dry and one processed channel.

The 12-second minimum is for pipeline experiments, not a guarantee of a useful model. Include quiet and hard picking, sustained notes, and different parts of the neck. Turn off delay, reverb, and modulation for the first captures. Compare uses the held-out end of the recording and a shared playback gain. It does not yet perform live model inference on new guitar playing. Continue monitoring through the Mini while capturing.

Original upstream: https://github.com/sdatkinson/neural-amp-modeler

## Checks

```sh
uv run pytest -q
```

31 tests plus 6 subtests cover upload/authentication, dataset validation, timing alignment, and capture framing. JavaScript syntax and the browser workflow are checked separately. A real Modal T4 smoke run completed 10 epochs, exported a valid `.nam`, and verified finite, aligned comparison files and their measured error. Hosted WebSocket capture, browser-triggered GPU training, model download, and comparison playback also passed using labelled synthetic audio. The hosted result's three 48 kHz previews each contain 147,251 samples; recomputed ESR agrees with the saved metric. Actual guitar audio and the Mini's dry/wet routing still need a first listening check.
