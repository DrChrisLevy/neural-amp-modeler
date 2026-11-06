"""Train an official NAM A2 model from a paired, mono capture.

The public web process can import this module without importing PyTorch or NAM.
Only ``train_job`` needs the training image's GPU dependencies.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import importlib.resources
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

SAMPLE_RATE = 48_000
NAM_VERSION = "0.13.0"
MIN_SECONDS = 12
MAX_ALIGNMENT_SAMPLES = 4096
ALIGNMENT_HEADROOM = 16
NAM_V3_AUDIO_SHA256 = "c3c2b88c8fdead4f958ab146de34bd4f85c4ba5808c7b933c55e85044a1910f2"


class _LossHistory:
    """Bound batch history with a uniform stride across the entire run."""

    def __init__(self, limit=2000):
        self.limit = limit
        self.stride = 1
        self.points = []
        self.latest = None

    def append(self, point):
        self.latest = point
        if not self.points or point["step"] % self.stride == 0:
            self.points.append(point)
        if len(self.points) >= self.limit:
            self.stride *= 2
            first = self.points[0]
            self.points = [p for p in self.points if p is first or p["step"] % self.stride == 0]

    def snapshot(self):
        points = list(self.points)
        if self.latest is not None and (not points or points[-1] is not self.latest):
            points.append(self.latest)
        return points


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=".status-", suffix=".json", delete=False
        ) as handle:
            name = handle.name
            json.dump(value, handle, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def _identifier(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise ValueError("Invalid capture or training job identifier")
    return value


def analyze_audio(audio: Any, label: str) -> dict[str, float]:
    """Reject unusable input without independently normalizing either signal."""
    import numpy as np

    a = np.asarray(audio)
    if a.ndim != 1 or not len(a):
        raise ValueError(f"{label} must be a nonempty mono recording")
    if not np.isfinite(a).all():
        raise ValueError(f"{label} contains invalid audio samples")
    peak = float(np.max(np.abs(a)))
    rms = float(np.sqrt(np.mean(np.square(a, dtype=np.float64))))
    if rms < 1e-5 or peak < 1e-4:
        raise ValueError(f"{label} is silent or too quiet; check the selected USB channel")
    clipped = float(np.mean(np.abs(a) >= 0.999969))
    if peak >= 1 or clipped > 0.001:
        raise ValueError(f"{label} clips at digital full scale; lower its level and record again")
    return {"peak": peak, "rms": rms, "rms_dbfs": 20 * math.log10(rms), "clipped_fraction": clipped}


def estimate_delay(dry: Any, wet: Any, max_lag: int = MAX_ALIGNMENT_SAMPLES) -> dict[str, Any]:
    """Estimate wet-minus-dry lag from multiple active windows of training audio.

    Polarity-independent waveform/derivative correlations find sample timing;
    magnitude-envelope derivatives make the estimate less sensitive to drive.
    This is an estimate for an amp/cab chain, not a calibration of time effects.
    """
    import numpy as np
    from scipy.signal import correlate
    from scipy.ndimage import uniform_filter1d

    length = min(len(dry), len(wet), 30 * SAMPLE_RATE)
    x = np.asarray(dry[:length], dtype=np.float64)
    y = np.asarray(wet[:length], dtype=np.float64)
    window = min(2 * SAMPLE_RATE, length)
    if window <= max_lag * 3:
        raise ValueError("Not enough audio to estimate the dry/wet timing")
    starts = np.unique(np.linspace(0, length - window, 6, dtype=int))
    lags = np.arange(-max_lag, max_lag + 1)
    scores, candidates = [], []

    def correlation(a, b, absolute=True):
        a, b = a - a.mean(), b - b.mean()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-12:
            return np.zeros(len(lags))
        values = correlate(b, a, mode="full", method="fft")
        values = values[window - 1 - max_lag : window + max_lag] / denom
        return np.abs(values) if absolute else np.maximum(values, 0)

    for start in starts:
        a, b = x[start : start + window], y[start : start + window]
        if np.sqrt(np.mean(a * a)) < 1e-5 or np.sqrt(np.mean(b * b)) < 1e-5:
            continue
        da, db = np.diff(a, prepend=a[0]), np.diff(b, prepend=b[0])
        ea = np.diff(uniform_filter1d(np.abs(a), size=48), prepend=0)
        eb = np.diff(uniform_filter1d(np.abs(b), size=48), prepend=0)
        # Avoid a fabricated onset at the beginning of an analysis window.
        ea[:48], eb[:48] = 0, 0
        score = 0.55 * correlation(a, b) + 0.30 * correlation(da, db) + 0.15 * correlation(ea, eb, False)
        scores.append(score)
        candidates.append(int(lags[np.argmax(score)]))
    if not scores:
        raise ValueError("No active guitar audio was available for timing alignment")
    score = np.mean(scores, axis=0)
    index = int(np.argmax(score))
    lag = int(lags[index])
    confidence = float(score[index])
    if abs(lag) == max_lag:
        raise ValueError("Dry/wet delay reached the alignment limit; check routing and disable delay/reverb")
    if confidence < 0.12:
        raise ValueError("Could not align dry and wet audio reliably; check channels and use a fixed amp/cab preset")
    alternatives = score[np.abs(lags - lag) > 64]
    prominence = float(confidence / max(float(alternatives.max()), 1e-12)) if len(alternatives) else 1.0
    warnings = []
    if confidence < 0.35 or prominence < 1.02:
        warnings.append("Timing is estimated with limited confidence; compare the reference and prediction carefully.")
    # Leave a small causal delay instead of making the model anticipate an amp's
    # response if its phase/frequency shaping moved the correlation peak.
    applied = max(-max_lag, lag - ALIGNMENT_HEADROOM)
    return {
        "estimated_delay_samples": lag,
        "applied_delay_samples": applied,
        "estimated_delay_ms": lag * 1000 / SAMPLE_RATE,
        "retained_delay_samples": lag - applied,
        "confidence": confidence,
        "peak_prominence": prominence,
        "window_delays_samples": candidates,
        "warnings": warnings,
    }


def align_pair(dry: Any, wet: Any, delay: int):
    """Positive delay means wet lags: discard wet's start and dry's end."""
    if len(dry) != len(wet):
        raise ValueError("Dry and wet recordings must contain the same number of samples")
    if abs(delay) >= len(dry):
        raise ValueError("Alignment delay is longer than the recording")
    if delay > 0:
        return dry[:-delay], wet[delay:]
    if delay < 0:
        return dry[-delay:], wet[:delay]
    return dry, wet


def is_standard_input(dry) -> bool:
    """Recognize v3 by sample content, even after a WAV header is rewritten."""
    import numpy as np

    return (len(dry) == 190 * SAMPLE_RATE
            and hashlib.sha256(np.asarray(dry, dtype="<f4").tobytes()).hexdigest() == NAM_V3_AUDIO_SHA256)


def calibrate_standard_delay(wet) -> dict[str, Any]:
    """Keep NAM's calibration, recentering long USB delays only when necessary."""
    from nam.train.core import _calibrate_latency_v3
    from .alignment import measure_v3_latency

    options = dict(manual_available=False, show_plots=False, _override_suppress_plots=True)
    calibration = _calibrate_latency_v3(wet, **options)
    coarse_offset = 0
    pulse_delays = None
    if calibration.recommended is None:
        pulse_delays = measure_v3_latency(wet)["pulse_delays_samples"]
        # Leave 1,000 samples for upstream's final onset measurement and its
        # causal safety factor. This view does not modify the saved recording.
        coarse_offset = max(0, min(pulse_delays) - 1000)
        calibration = _calibrate_latency_v3(wet[coarse_offset:], **options)
        if calibration.recommended is None:
            raise ValueError("NAM could not calibrate the test pulses. Check the USB routing and levels.")
    applied = int(calibration.recommended) + coarse_offset
    estimated = applied + calibration.safety_factor
    if pulse_delays is not None and any(abs(estimated - delay) >= 20 for delay in pulse_delays):
        raise ValueError("NAM's test pulses do not have a consistent delay. Check noise, gates, and time effects.")
    details = calibration.model_dump(mode="json")
    details["recommended"] = applied
    details["delays"] = [int(delay) + coarse_offset for delay in calibration.delays]
    return {
        "method": "nam_v3_blips",
        "applied_delay_samples": applied,
        "estimated_delay_samples": estimated,
        "estimated_delay_ms": estimated * 1000 / SAMPLE_RATE,
        "retained_delay_samples": calibration.safety_factor,
        "coarse_offset_samples": coarse_offset,
        "pulse_delays_samples": pulse_delays,
        "calibration": details,
        "warnings": [],
    }


def check_repeated_output(wet) -> float:
    """NAM v3 repeats its validation stimulus; the device should be consistent."""
    import numpy as np

    first, last = wet[:9 * SAMPLE_RATE], wet[-9 * SAMPLE_RATE:]
    energy = float(np.sum(np.square(last, dtype=np.float64)))
    error = float(np.sum(np.square(first - last, dtype=np.float64))) / max(energy, 1e-30)
    if error > 0.01:
        raise ValueError("The tone changed between NAM's repeated test passages. Check noise, gates, time effects, and preset controls, then capture again.")
    return error


def split_segments(length: int, receptive_field: int, *, standard=False) -> dict[str, int]:
    """Keep train and validation inputs disjoint, including model context."""
    if standard and length != 190 * SAMPLE_RATE:
        raise ValueError("NAM v3 must be exactly 190 seconds before alignment")
    split = 181 * SAMPLE_RATE if standard else int(length * 0.8)
    validation_target_start = split + receptive_field - 1
    if split < receptive_field + 8192 or length - validation_target_start < SAMPLE_RATE:
        raise ValueError("Recording is too short for separate training and validation audio")
    return {
        "train_start": 10 * SAMPLE_RATE if standard else 0,
        "train_stop": split,
        "validation_input_start": split,
        "validation_target_start": validation_target_start,
        "validation_stop": length,
    }


def _select_model_config(recipe: dict[str, Any], model_size: str) -> dict[str, Any]:
    result = copy.deepcopy(recipe)
    if model_size == "both":
        return result
    if model_size not in {"lite", "full"}:
        raise ValueError("Model size must be lite, full, or both")
    width = 3 if model_size == "lite" else 8
    matches = [
        sub for sub in recipe["net"]["config"]["submodels"]
        if sub["config"]["layers_configs"][0]["channels"] == width
    ]
    if len(matches) != 1:
        raise RuntimeError("Installed NAM A2 recipe does not match the expected architecture")
    result["net"] = {"name": "WaveNet", "config": copy.deepcopy(matches[0]["config"])}
    return result


def _render(net, dry, start: int, stop: int):
    """Render held-out audio with real preceding context, in bounded GPU chunks."""
    import numpy as np
    import torch

    context = int(net.receptive_field) - 1
    outputs = []
    with torch.inference_mode():
        for at in range(start, stop, 131_072):
            end = min(at + 131_072, stop)
            chunk = torch.from_numpy(np.ascontiguousarray(dry[at - context : end])).to(net.device)
            prediction = net(chunk, pad_start=False).detach().cpu().numpy()
            if prediction.shape[-1] != end - at or not np.isfinite(prediction).all():
                raise RuntimeError("Model produced invalid validation audio")
            outputs.append(prediction)
    return np.concatenate(outputs, axis=-1)


def _write_audition(directory: Path, dry, reference, predictions: dict[str, Any], selected: str) -> float:
    """Apply one shared safety gain so A/B levels and differences stay honest."""
    import numpy as np
    import soundfile as sf

    signals = {"dry": dry, "reference": reference, **{f"prediction_{k}": v for k, v in predictions.items()}}
    if any(not np.isfinite(a).all() for a in signals.values()):
        raise RuntimeError("Cannot export non-finite audio")
    peak = max(float(np.max(np.abs(a))) for a in signals.values())
    gain = min(1.0, 0.95 / max(peak, 1e-12))
    for name, audio in signals.items():
        sf.write(directory / f"{name}.wav", audio * gain, SAMPLE_RATE, subtype="PCM_24")
    sf.write(directory / "prediction.wav", predictions[selected] * gain, SAMPLE_RATE, subtype="PCM_24")
    return gain


def train_job(root: str, job_id: str, commit: Callable[[], Any] | None = None) -> dict[str, Any]:
    """Run one queued job; persist its progress and actual exported model/audio.

    ``commit`` is normally a Modal Volume commit. Errors are written to status
    and re-raised, so both the UI and Modal invocation reflect the failure.
    """
    directory = Path(root) / "jobs" / _identifier(job_id)
    job_started = time.monotonic()
    status_path = directory / "status.json"
    status = json.loads(status_path.read_text())
    publisher = None
    timings = {}

    def phase_done(name, started):
        timings[name] = round(time.monotonic() - started, 4)

    def publish(snapshot):
        _atomic_json(status_path, snapshot)
        if commit:
            commit()

    def update(**fields):
        status.update(fields, updated_at=_now())
        if publisher is None:
            publish(status)
        else:
            publisher.submit(status)

    try:
        import numpy as np
        import soundfile as sf

        epochs = int(status["epochs"])
        if not 10 <= epochs <= 1000:
            raise ValueError("Training epochs must be between 10 and 1000")
        size = status["model_size"]
        if size not in {"lite", "full", "both"}:
            raise ValueError("Model size must be lite, full, or both")
        capture = Path(root) / "captures" / _identifier(status["capture_id"])
        metadata = json.loads((capture / "meta.json").read_text())
        update(status="training", stage="preparing", started_at=_now(), epoch=0, progress=0.01,
               metrics=[], training_metrics=[], metric_schema=2, error=None)
        phase_started = time.monotonic()
        dry, dry_rate = sf.read(capture / "dry.wav", dtype="float32")
        wet, wet_rate = sf.read(capture / "wet.wav", dtype="float32")
        phase_done("read_audio", phase_started)
        phase_started = time.monotonic()
        if dry_rate != SAMPLE_RATE or wet_rate != SAMPLE_RATE:
            raise ValueError("Training requires dry and wet recordings at 48 kHz")
        if len(dry) != len(wet):
            raise ValueError("Dry and wet recordings must have matching lengths")
        if len(dry) < MIN_SECONDS * SAMPLE_RATE:
            raise ValueError(f"Record at least {MIN_SECONDS} seconds before training")
        quality = {"dry": analyze_audio(dry, "Dry audio"), "wet": analyze_audio(wet, "Wet audio"), "warnings": []}
        standard = is_standard_input(dry)
        if standard:
            quality["repeat_esr"] = check_repeated_output(wet)
        if len(dry) < 30 * SAMPLE_RATE:
            quality["warnings"].append("This is a short test capture; record a few minutes of varied playing for a useful model.")
        if min(quality["dry"]["rms_dbfs"], quality["wet"]["rms_dbfs"]) < -55:
            quality["warnings"].append("Capture level is low; more input level may improve the training signal.")
        phase_done("audio_checks", phase_started)
        phase_started = time.monotonic()
        if standard:
            alignment = calibrate_standard_delay(wet)
        else:
            # Arbitrary paired takes use correlation on training audio only.
            calibration_stop = int(len(dry) * 0.8)
            alignment = estimate_delay(dry[:calibration_stop], wet[:calibration_stop])
        phase_done("alignment", phase_started)
        quality["warnings"].extend(alignment["warnings"])
        update(alignment=alignment, quality=quality, duration_seconds=len(dry) / SAMPLE_RATE, progress=0.03)

        phase_started = time.monotonic()
        # All training imports remain inside this GPU-only entry point.
        import torch
        import pytorch_lightning as pl
        from torch.utils.data import DataLoader
        from nam.train.lightning_module import PackedMaskCallback
        from .training_runtime import TrainingBatches, ValidationChunks, ValidatedLightningModule, ValidatedPackedLightningModule, prepare_datasets

        installed = importlib.metadata.version("neural-amp-modeler")
        if installed != NAM_VERSION:
            raise RuntimeError(f"Expected neural-amp-modeler {NAM_VERSION}, found {installed}")
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required for this training job")
        torch.cuda.reset_peak_memory_stats()
        pl.seed_everything(17, workers=True)
        torch.set_float32_matmul_precision("high")
        # Tune convolution kernels for this GPU before CUDA graphs capture them.
        # The startup cost is amortized over the repeated training-window shapes.
        torch.backends.cudnn.benchmark = True
        recipe = json.loads(importlib.resources.files("nam.train._resources").joinpath("config_model_packed.json").read_text())
        config = _select_model_config(recipe, size)
        # Same Adam update, with the many small parameter operations fused on CUDA.
        config["optimizer"]["fused"] = True
        is_packed = size == "both"
        model_class = ValidatedPackedLightningModule if is_packed else ValidatedLightningModule
        model = model_class.init_from_config(config)
        model.net.sample_rate = SAMPLE_RATE
        receptive_field = int(model.net.receptive_field)
        split = split_segments(len(dry), receptive_field, standard=standard)
        training, validation, normalization_gain = prepare_datasets(
            dry, wet, split, receptive_field, alignment["applied_delay_samples"])
        analyze_audio(validation.x.numpy(), "Validation dry audio")
        analyze_audio(validation.y.numpy() / normalization_gain, "Validation wet audio")
        for dataset in (training, validation):
            dataset.handshake(model.net)
            model.net.handshake(dataset)
        # At the ten-minute capture limit the mono pair occupies only ~230 MB.
        # Keep it on the GPU and gather batches there instead of copying each one.
        training_batches = TrainingBatches(training.x.cuda(), training.y.cuda(), receptive_field)
        train_loader = DataLoader(training_batches, batch_size=None, num_workers=0)
        validation_chunks = ValidationChunks(validation.x.cuda(), validation.y.cuda(), receptive_field)
        validation_loader = DataLoader(validation_chunks, batch_size=None, num_workers=0)
        _atomic_json(directory / "model_config.json", config)
        _atomic_json(directory / "training_config.json", {"nam_version": installed, "epochs": epochs, "split": split, "alignment": alignment, "sample_rate": SAMPLE_RATE, "capture": metadata,
                    "input_signal": "nam_v3" if standard else "custom", "data_recipe": "a2_rms_v3_split_v1",
                    "target_rms_dbfs": -18, "normalization_gain": normalization_gain,
                    "validation_metric": "whole_recording_esr", "batch_size": 16, "fused_adam": True, "precision": "32-true", "audio_device": "cuda", "cuda_graphs": True, "cudnn_benchmark": True,
                    "gpu": torch.cuda.get_device_name(), "audio_bytes": dry.nbytes + wet.nbytes})
        from .progress_writer import ProgressWriter
        publisher = ProgressWriter(publish)
        phase_done("model_and_data_setup", phase_started)
        update(stage="training", timings_seconds=timings.copy(), validation_metric="whole_recording_esr",
               data_recipe="a2_rms_v3_split_v1", input_signal="nam_v3" if standard else "custom",
               receptive_field_samples=receptive_field, validation_seconds=validation_chunks.samples / SAMPLE_RATE, progress=0.05)

        class Progress(pl.Callback):
            last_update = 0.0

            def __init__(self):
                self.started = time.monotonic()
                self.history = _LossHistory()
                self.pending = []
                self.epoch_loss_sum = 0.0
                self.epoch_examples = 0

            def on_train_epoch_start(self, trainer, module):
                self.epoch_loss_sum = 0.0
                self.epoch_examples = 0

            def fields(self):
                return {"training_metrics": self.history.snapshot(),
                        "train_loss": self.history.latest["loss"] if self.history.latest else None,
                        "train_epoch_loss": self.epoch_loss_sum / self.epoch_examples if self.epoch_examples else None,
                        "elapsed_seconds": round(time.monotonic() - self.started, 2)}

            def flush(self):
                if not self.pending:
                    return
                # One device synchronization per UI update, not one per batch.
                values = torch.stack([item[2] for item in self.pending]).cpu().tolist()
                for (step, epoch, _, examples), value in zip(self.pending, values):
                    value /= 2 if is_packed else 1
                    if not math.isfinite(value):
                        raise RuntimeError("Training loss became non-finite")
                    self.history.append({"step": step, "epoch": epoch, "loss": value})
                    self.epoch_loss_sum += value * examples
                    self.epoch_examples += examples
                self.pending.clear()

            def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
                loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
                if loss is None:
                    raise RuntimeError("NAM did not report a training loss")
                fraction = (batch_idx + 1) / max(int(trainer.num_training_batches), 1)
                # Graph replay reuses its output storage; snapshot each measured loss.
                self.pending.append((int(trainer.global_step), round(trainer.current_epoch + fraction, 6), loss.detach().clone(), len(batch[0])))
                if time.monotonic() - self.last_update < 2:
                    return
                self.flush()
                update(epoch=int(trainer.current_epoch) + 1,
                       progress=0.05 + 0.85 * (trainer.current_epoch + fraction) / epochs,
                       **self.fields())
                self.last_update = time.monotonic()

            def on_validation_end(self, trainer, module):
                if trainer.sanity_checking:
                    return
                self.flush()
                value = trainer.callback_metrics.get("val_loss")
                if value is None:
                    raise RuntimeError("NAM did not report a validation loss")
                loss = float(value.detach().cpu()) / (2 if is_packed else 1)
                if not math.isfinite(loss):
                    raise RuntimeError("Validation loss became non-finite")
                epoch = int(trainer.current_epoch) + 1
                metrics = [*status["metrics"], {"epoch": epoch, "loss": loss,
                           "train_loss": self.epoch_loss_sum / max(self.epoch_examples, 1)}]
                if loss < status.get("best_loss", math.inf):
                    status.update(best_epoch=epoch, best_loss=loss)
                update(epoch=epoch, loss=loss, metric_name="Validation ESR", metrics=metrics,
                       progress=0.05 + 0.85 * epoch / epochs, **self.fields())
                self.last_update = time.monotonic()

        # Keep per-epoch checkpoint I/O on local disk. Persist the winning file once.
        checkpoint_temp = tempfile.TemporaryDirectory(prefix="neural-amp-modeler-checkpoints-")
        checkpoint = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_temp.name, filename="best", monitor="val_loss", mode="min", save_top_k=1)
        callbacks = [Progress(), checkpoint]
        if is_packed:
            callbacks.append(PackedMaskCallback())
        trainer = pl.Trainer(
            accelerator="gpu", devices=1, max_epochs=epochs,
            precision="32-true", callbacks=callbacks, default_root_dir=directory,
            logger=False, enable_progress_bar=False, enable_model_summary=False,
            num_sanity_val_steps=0, gradient_clip_val=1.0,
        )
        try:
            phase_started = time.monotonic()
            trainer.fit(model, train_loader, validation_loader)
            phase_done("fit", phase_started)
            if not checkpoint.best_model_path:
                raise RuntimeError("Training finished without a validated checkpoint")
            saved_checkpoint = directory / "checkpoints" / "best.ckpt"
            saved_checkpoint.parent.mkdir(exist_ok=True)
            shutil.copyfile(checkpoint.best_model_path, saved_checkpoint)
        finally:
            checkpoint_temp.cleanup()
        update(stage="exporting", progress=0.92)
        phase_started = time.monotonic()
        best = model_class.load_from_checkpoint(saved_checkpoint, map_location="cpu", **model_class.parse_config(config))
        best.net.sample_rate = SAMPLE_RATE
        # Dataset export hooks are not checkpoint weights. Reattach the official
        # gain compensation to the selected model before exporting its .nam.
        training.handshake(best.net)
        best.net.handshake(training)
        best.eval().cuda()
        start = receptive_field - 1
        validation_dry = validation.x.numpy()
        output = _render(best.net, validation_dry, start, len(validation_dry)) / normalization_gain
        predictions = {"lite": output[0], "full": output[1]} if is_packed else {size: output}
        reference = validation.y.numpy()[start:] / normalization_gain
        if float(np.mean(np.square(reference, dtype=np.float64))) < 1e-10:
            raise ValueError("The held-out end of the capture is silent; finish with varied guitar playing")
        esrs = {
            key: float(np.sum(np.square(pred - reference, dtype=np.float64)) / np.sum(np.square(reference, dtype=np.float64)))
            for key, pred in predictions.items()
        }
        if not all(math.isfinite(value) for value in esrs.values()):
            raise RuntimeError("Model produced non-finite validation metrics")
        selected = "full" if is_packed else size
        audition_gain = _write_audition(directory, validation_dry[start:], reference, predictions, selected)
        with torch.inference_mode():
            best.net.export(directory, basename="model_pending", other_metadata={"name": f"Neural Amp Modeler {job_id}", "capture_id": status["capture_id"]})
        # Invalid model numbers must never become a downloadable successful run.
        model_path = directory / "model_pending.nam"
        def invalid_number(value):
            raise ValueError(f"Invalid model value: {value}")
        json.loads(model_path.read_text(), parse_constant=invalid_number)
        os.replace(model_path, directory / "model.nam")
        phase_done("render_and_export", phase_started)
        update(
            status="completed", stage="completed", completed_at=_now(), epoch=epochs,
            progress=1.0, loss=esrs[selected], esr=esrs[selected], esr_by_model=esrs,
            comparison_model=selected, audition_gain=audition_gain,
            comparison_seconds=len(reference) / SAMPLE_RATE,
            total_seconds=round(time.monotonic() - job_started, 2),
            timings_seconds=timings.copy(),
            gpu=torch.cuda.get_device_name(), peak_gpu_memory_mb=round(torch.cuda.max_memory_allocated() / 1e6, 1),
            model_file="model.nam", prediction_file="prediction.wav",
            reference_file="reference.wav", dry_file="dry.wav", error=None,
        )
        publisher.close()
        publisher = None
        return status
    except Exception as exc:
        if publisher is not None:
            try:
                publisher.close()
            except Exception:
                pass
            publisher = None
        try:
            update(status="failed", stage="failed", failed_at=_now(), error=str(exc))
        except Exception:
            # Preserve the original failure in Modal's traceback if committing
            # the failure record itself also failed.
            pass
        raise
