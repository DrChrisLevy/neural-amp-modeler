"""Reamp one WAV through the currently edited preset and save a reusable pair."""

from datetime import datetime
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import tempfile

import numpy as np
import soundfile as sf

from .storage import MAX_SECONDS, SAMPLE_RATE, atomic_json, now
from .alignment import measure_v3_latency
from .training import analyze_audio, is_standard_input, estimate_delay, check_repeated_output

INPUT_SHA256 = "70f8ec7f25686a1bd77f25973de8e51a6721e957e81eec121822e5e53366bc41"
INPUT_URL = "https://drive.usercontent.google.com/download?id=1KbaS4oXXNEuh2aCPLwKrPdf5KFOjda8G&export=download&confirm=t"


@contextmanager
def capture_lock():
    """Two terminal commands must never reroute/play the same device together."""
    path = Path(tempfile.gettempdir()) / f"neural-amp-modeler-capture-{os.getuid()}.lock"
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another capture is running. Wait for it to finish.") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def download_input(path=Path(".data/input.wav")):
    """Download the unmodified, checksum-verified NAM v3 input from its docs."""
    import httpx

    path = Path(path)
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != INPUT_SHA256:
            raise ValueError(f"A different file exists at {path}; choose another --output.")
        return path
    response = httpx.get(INPUT_URL, follow_redirects=True, timeout=120)
    response.raise_for_status()
    if hashlib.sha256(response.content).hexdigest() != INPUT_SHA256:
        raise ValueError("Download did not match the official NAM input; no file was saved.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(response.content)
    return path


def load_input(path):
    audio, rate = sf.read(path, dtype="float32")
    if rate != SAMPLE_RATE or audio.ndim != 1:
        raise ValueError("Use a mono, 48 kHz input WAV. The 'input' command downloads NAM's standard signal.")
    if not 12 * SAMPLE_RATE <= len(audio) <= MAX_SECONDS * SAMPLE_RATE:
        raise ValueError("Input must be between 12 seconds and 10 minutes.")
    analyze_audio(audio, "Input WAV")
    return audio


def audio_device(sd):
    devices = [(i, d) for i, d in enumerate(sd.query_devices())
               if "quad cortex" in d["name"].lower()
               and d["max_input_channels"] >= 4 and d["max_output_channels"] >= 5]
    if len(devices) != 1:
        raise RuntimeError(f"Expected one Quad Cortex USB audio device; found {len(devices)}.")
    return devices[0][0]


def play_and_record(audio, device, sd):
    # One full-duplex clock, unity gain, one second of pre-roll and tail. Only
    # USB 5 is driven. Stereo return is archived; training selects one channel.
    playback = np.pad(audio, (SAMPLE_RATE, SAMPLE_RATE))[:, None]
    try:
        result = sd.playrec(playback, samplerate=SAMPLE_RATE, device=(device, device),
                            input_mapping=[3, 4], output_mapping=[5], dtype="float32",
                            latency="high", blocking=True)
        if sd.get_status():
            raise RuntimeError("Audio dropped during capture; retry this take.")
        if result.shape != (len(playback), 2) or not np.isfinite(result).all():
            raise RuntimeError("Invalid or incomplete audio returned by the device.")
        return result
    finally:
        sd.stop()


def aligned_output(audio, raw, wet_channel, preroll_samples=SAMPLE_RATE):
    """Remove measured recording latency using real pre-roll/tail samples.

    Keep the dry file unchanged and retain a small causal margin. Never pad a
    missing tail with silence or normalize the output to disguise bad levels.
    """
    if wet_channel not in (3, 4) or raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError("Capture requires stereo USB 3/4 and a wet channel of 3 or 4.")
    if preroll_samples < 0 or len(raw) < preroll_samples + len(audio):
        raise ValueError("Raw recording is incomplete; cannot export an aligned pair.")
    if not np.isfinite(raw).all():
        raise ValueError("Raw recording contains invalid audio samples.")
    wet = raw[preroll_samples:preroll_samples + len(audio), wet_channel - 3]
    analyze_audio(wet, "Processed audio")
    standard = is_standard_input(audio)
    if standard:
        alignment = measure_v3_latency(wet)
    else:
        calibration_stop = int(len(audio) * .8)
        alignment = estimate_delay(audio[:calibration_stop], wet[:calibration_stop], max_lag=24_000)
        if alignment["warnings"]:
            raise ValueError("Custom input has ambiguous timing. Use the standard NAM input for a reliable capture.")
        alignment["method"] = "training_audio_correlation"
    start = preroll_samples + alignment["applied_delay_samples"]
    stop = start + len(audio)
    if start < 0 or stop > len(raw):
        raise ValueError("Raw recording has insufficient pre-roll or tail for timing correction.")
    output = raw[start:stop, wet_channel - 3]
    quality = analyze_audio(output, "Aligned processed audio")
    if standard:
        # Verify the exported PCM24 samples, including the recovered ending.
        encoded = io.BytesIO()
        sf.write(encoded, output, SAMPLE_RATE, format="WAV", subtype="PCM_24")
        encoded.seek(0)
        exported, _ = sf.read(encoded, dtype="float32")
        measured = measure_v3_latency(exported)
        if measured["applied_delay_samples"] != 0:
            raise ValueError("Exported timing pulses did not align correctly.")
        quality["repeat_esr"] = check_repeated_output(exported)
    alignment.update(export_version=1, output_start_sample=start,
                     output_frames=len(audio), estimated_delay_ms=alignment["estimated_delay_samples"] * 1000 / SAMPLE_RATE)
    return output, alignment, quality


def repair_capture(source, output, *, emit=print):
    """Export a separate aligned pair from an intact archived capture; no USB."""
    source, output = Path(source), Path(output)
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata.get("status") != "ready":
        raise ValueError("Only a completed capture can be repaired automatically.")
    for filename in ("input.wav", "raw.wav"):
        expected = metadata.get("files", {}).get(filename)
        if not expected or hashlib.sha256((source / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Archived {filename} does not match the capture checksum.")
    source_bytes = (source / "input.wav").read_bytes()
    audio = load_input(io.BytesIO(source_bytes))
    raw, rate = sf.read(source / "raw.wav", dtype="float32")
    if rate != SAMPLE_RATE:
        raise ValueError("Raw recording must be 48 kHz.")
    wet, alignment, quality = aligned_output(audio, raw, metadata["wet_channel"],
                                             round(metadata["preroll_seconds"] * SAMPLE_RATE))
    output.mkdir(parents=True, exist_ok=False)
    (output / "input.wav").write_bytes(source_bytes)
    sf.write(output / "output.wav", wet, SAMPLE_RATE, subtype="PCM_24")
    metadata.update(alignment=alignment, quality=quality, repaired_at=now(),
                    repaired_from=str(source.resolve()), raw_source=str((source / "raw.wav").resolve()),
                    files={name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                           for name in ("input.wav", "output.wav")})
    atomic_json(output / "metadata.json", metadata)
    emit(f"Saved aligned pair: {output.resolve()}")
    emit(f"Corrected {alignment['estimated_delay_ms']:.2f} ms of recording latency; original capture preserved.")
    return output


def capture_pair(source, name, output=Path(".data/pairs"), wet_channel=3, *, controller=None, sd=None, emit=print):
    """No preset changes. Each attempt gets a new directory, even on failure."""
    if wet_channel not in (3, 4):
        raise ValueError("Choose USB return channel 3 (left) or 4 (right).")
    source_bytes = Path(source).read_bytes()
    audio = load_input(io.BytesIO(source_bytes))
    if controller is None:
        from .quad_cortex import Mini
        controller = Mini()
    if sd is None:
        import sounddevice as sd
    device = audio_device(sd)
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:70] or "capture"
    folder = Path(output) / f"{datetime.now():%Y%m%d-%H%M%S-%f}-{slug}"
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "input.wav").write_bytes(source_bytes)
    metadata = {"name": name, "created_at": now(), "status": "starting", "sample_rate": SAMPLE_RATE,
                "seconds": len(audio) / SAMPLE_RATE, "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "playback_channel": 5, "wet_channel": wet_channel, "preroll_seconds": 1, "tail_seconds": 1,
                "alignment": None}
    previous_signal = signal.getsignal(signal.SIGTERM)
    def stop(signum, frame):
        raise KeyboardInterrupt("Capture interrupted")
    signal.signal(signal.SIGTERM, stop)
    try:
        with capture_lock(), controller as mini:
            baseline, state = mini.snapshot()
            metadata["device"] = state
            atomic_json(folder / "metadata.json", metadata)
            emit(f"Capturing {state['preset_name']}{' *' if state['unsaved_edits'] else ''} · {len(audio)/SAMPLE_RATE:.1f} seconds")
            emit("Keep this preset and its controls unchanged until capture finishes.")
            with mini.usb_input(baseline, state):
                raw = play_and_record(audio, device, sd)
                # Retain the return even if the subsequent state/restoration
                # check fails. Such attempts never publish a ready pair.
                sf.write(folder / "raw.wav", raw, SAMPLE_RATE, subtype="PCM_24")
            wet, metadata["alignment"], metadata["quality"] = aligned_output(audio, raw, wet_channel)
            sf.write(folder / "output.wav", wet, SAMPLE_RATE, subtype="PCM_24")
            metadata.update(status="ready", completed_at=now(), files={
                filename: hashlib.sha256((folder / filename).read_bytes()).hexdigest()
                for filename in ("input.wav", "output.wav", "raw.wav")})
        emit(f"Saved pair: {folder.resolve()}")
        emit(f"Aligned recording latency: {metadata['alignment']['estimated_delay_ms']:.2f} ms")
        emit("Upload input.wav as Dry input and output.wav as Processed tone.")
        return folder
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        emit(f"Capture incomplete. Details: {folder.resolve() / 'metadata.json'}")
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_signal)
        atomic_json(folder / "metadata.json", metadata)
