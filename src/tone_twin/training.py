"""Train an official NAM A2 model from a paired, mono capture.

The public web process can import this module without importing PyTorch or NAM.
Only ``train_job`` needs the training image's GPU dependencies.
"""

from __future__ import annotations

import copy
import importlib.metadata
import importlib.resources
import json
import math
import os
from pathlib import Path
import re
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


def split_segments(length: int, receptive_field: int) -> dict[str, int]:
    """Keep train and validation inputs disjoint, including model context."""
    split = int(length * 0.8)
    validation_target_start = split + receptive_field - 1
    if split < receptive_field + 8192 or length - validation_target_start < SAMPLE_RATE:
        raise ValueError("Recording is too short for separate training and validation audio")
    return {
        "train_start": 0,
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
        for at in range(start, stop, 32_768):
            end = min(at + 32_768, stop)
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
    status_path = directory / "status.json"
    status = json.loads(status_path.read_text())

    def update(**fields):
        status.update(fields, updated_at=_now())
        _atomic_json(status_path, status)
        if commit:
            commit()

    try:
        import numpy as np
        import soundfile as sf

        epochs = int(status["epochs"])
        if not 10 <= epochs <= 100:
            raise ValueError("Training epochs must be between 10 and 100")
        size = status["model_size"]
        if size not in {"lite", "full", "both"}:
            raise ValueError("Model size must be lite, full, or both")
        capture = Path(root) / "captures" / _identifier(status["capture_id"])
        metadata = json.loads((capture / "meta.json").read_text())
        update(status="training", stage="preparing", started_at=_now(), epoch=0, progress=0.01, metrics=[], error=None)
        dry, dry_rate = sf.read(capture / "dry.wav", dtype="float32")
        wet, wet_rate = sf.read(capture / "wet.wav", dtype="float32")
        if dry_rate != SAMPLE_RATE or wet_rate != SAMPLE_RATE:
            raise ValueError("Training requires dry and wet recordings at 48 kHz")
        if len(dry) != len(wet):
            raise ValueError("Dry and wet recordings must have matching lengths")
        if len(dry) < MIN_SECONDS * SAMPLE_RATE:
            raise ValueError(f"Record at least {MIN_SECONDS} seconds before training")
        quality = {"dry": analyze_audio(dry, "Dry audio"), "wet": analyze_audio(wet, "Wet audio"), "warnings": []}
        if len(dry) < 30 * SAMPLE_RATE:
            quality["warnings"].append("This is a short test capture; record a few minutes of varied playing for a useful model.")
        if min(quality["dry"]["rms_dbfs"], quality["wet"]["rms_dbfs"]) < -55:
            quality["warnings"].append("Capture level is low; more input level may improve the training signal.")
        # The held-out fifth is never used to calibrate alignment.
        calibration_stop = int(len(dry) * 0.8)
        alignment = estimate_delay(dry[:calibration_stop], wet[:calibration_stop])
        dry, wet = align_pair(dry, wet, alignment["applied_delay_samples"])
        dry, wet = np.ascontiguousarray(dry), np.ascontiguousarray(wet)
        quality["warnings"].extend(alignment["warnings"])
        update(alignment=alignment, quality=quality, duration_seconds=len(dry) / SAMPLE_RATE, progress=0.03)

        # All training imports remain inside this GPU-only entry point.
        import torch
        import pytorch_lightning as pl
        from torch.utils.data import DataLoader
        from nam.data import Dataset
        from nam.train.lightning_module import LightningModule, PackedLightningModule, PackedMaskCallback

        installed = importlib.metadata.version("neural-amp-modeler")
        if installed != NAM_VERSION:
            raise RuntimeError(f"Expected neural-amp-modeler {NAM_VERSION}, found {installed}")
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required for this training job")
        pl.seed_everything(17, workers=True)
        torch.set_float32_matmul_precision("high")
        recipe = json.loads(importlib.resources.files("nam.train._resources").joinpath("config_model_packed.json").read_text())
        config = _select_model_config(recipe, size)
        is_packed = size == "both"
        model_class = PackedLightningModule if is_packed else LightningModule
        model = model_class.init_from_config(config)
        model.net.sample_rate = SAMPLE_RATE
        receptive_field = int(model.net.receptive_field)
        split = split_segments(len(dry), receptive_field)
        train_stop = split["train_stop"]
        analyze_audio(dry[split["validation_target_start"] :], "Validation dry audio")
        analyze_audio(wet[split["validation_target_start"] :], "Validation wet audio")
        # Each partition starts with its own unscored receptive-field context.
        # No training target or input sample appears in the validation partition.
        kwargs = dict(nx=receptive_field, ny=8192, sample_rate=SAMPLE_RATE, require_input_pre_silence=None)
        training = Dataset(torch.from_numpy(dry[:train_stop]), torch.from_numpy(wet[:train_stop]), **kwargs)
        validation = Dataset(torch.from_numpy(dry[train_stop:]), torch.from_numpy(wet[train_stop:]), **kwargs)
        for dataset in (training, validation):
            dataset.handshake(model.net)
            model.net.handshake(dataset)
        train_loader = DataLoader(training, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
        validation_loader = DataLoader(validation, batch_size=8, num_workers=0, pin_memory=True)
        _atomic_json(directory / "model_config.json", config)
        _atomic_json(directory / "training_config.json", {"nam_version": installed, "epochs": epochs, "split": split, "alignment": alignment, "sample_rate": SAMPLE_RATE, "capture": metadata})
        update(stage="training", receptive_field_samples=receptive_field, validation_seconds=(len(dry) - split["validation_target_start"]) / SAMPLE_RATE, progress=0.05)

        class Progress(pl.Callback):
            last_update = 0.0

            def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
                loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
                if loss is not None and not bool(torch.isfinite(loss).all()):
                    raise RuntimeError("Training loss became non-finite")
                if time.monotonic() - self.last_update < 10:
                    return
                fraction = (batch_idx + 1) / max(int(trainer.num_training_batches), 1)
                update(epoch=int(trainer.current_epoch) + 1, progress=0.05 + 0.85 * (trainer.current_epoch + fraction) / epochs)
                self.last_update = time.monotonic()

            def on_validation_end(self, trainer, module):
                if trainer.sanity_checking:
                    return
                value = trainer.callback_metrics.get("val_loss")
                if value is None:
                    raise RuntimeError("NAM did not report a validation loss")
                loss = float(value.detach().cpu()) / (2 if is_packed else 1)
                if not math.isfinite(loss):
                    raise RuntimeError("Validation loss became non-finite")
                epoch = int(trainer.current_epoch) + 1
                metrics = [*status["metrics"], {"epoch": epoch, "loss": loss}]
                if loss < status.get("best_loss", math.inf):
                    status.update(best_epoch=epoch, best_loss=loss)
                update(epoch=epoch, loss=loss, metric_name="Validation ESR", metrics=metrics, progress=0.05 + 0.85 * epoch / epochs)
                self.last_update = time.monotonic()

        checkpoint = pl.callbacks.ModelCheckpoint(dirpath=directory / "checkpoints", filename="best", monitor="val_loss", mode="min", save_top_k=1)
        callbacks = [Progress(), checkpoint]
        if is_packed:
            callbacks.append(PackedMaskCallback())
        trainer = pl.Trainer(
            accelerator="gpu", devices=1, max_epochs=epochs,
            precision="32-true", callbacks=callbacks, default_root_dir=directory,
            logger=False, enable_progress_bar=False, enable_model_summary=False,
            num_sanity_val_steps=0, gradient_clip_val=1.0,
        )
        trainer.fit(model, train_loader, validation_loader)
        if not checkpoint.best_model_path:
            raise RuntimeError("Training finished without a validated checkpoint")
        update(stage="exporting", progress=0.92)
        best = model_class.load_from_checkpoint(checkpoint.best_model_path, map_location="cpu", **model_class.parse_config(config))
        best.net.sample_rate = SAMPLE_RATE
        best.eval().cuda()
        start = split["validation_target_start"]
        output = _render(best.net, dry, start, len(dry))
        predictions = {"lite": output[0], "full": output[1]} if is_packed else {size: output}
        reference = wet[start:]
        if float(np.mean(np.square(reference, dtype=np.float64))) < 1e-10:
            raise ValueError("The held-out end of the capture is silent; finish with varied guitar playing")
        esrs = {
            key: float(np.sum(np.square(pred - reference, dtype=np.float64)) / np.sum(np.square(reference, dtype=np.float64)))
            for key, pred in predictions.items()
        }
        if not all(math.isfinite(value) for value in esrs.values()):
            raise RuntimeError("Model produced non-finite validation metrics")
        selected = "full" if is_packed else size
        audition_gain = _write_audition(directory, dry[start:], reference, predictions, selected)
        with torch.inference_mode():
            best.net.export(directory, basename="model_pending", other_metadata={"name": f"Tone Twin {job_id}", "capture_id": status["capture_id"]})
        # Invalid model numbers must never become a downloadable successful run.
        model_path = directory / "model_pending.nam"
        def invalid_number(value):
            raise ValueError(f"Invalid model value: {value}")
        json.loads(model_path.read_text(), parse_constant=invalid_number)
        os.replace(model_path, directory / "model.nam")
        update(
            status="completed", stage="completed", completed_at=_now(), epoch=epochs,
            progress=1.0, loss=esrs[selected], esr=esrs[selected], esr_by_model=esrs,
            comparison_model=selected, audition_gain=audition_gain,
            comparison_seconds=len(reference) / SAMPLE_RATE,
            model_file="model.nam", prediction_file="prediction.wav",
            reference_file="reference.wav", dry_file="dry.wav", error=None,
        )
        return status
    except Exception as exc:
        try:
            update(status="failed", stage="failed", failed_at=_now(), error=str(exc))
        except Exception:
            # Preserve the original failure in Modal's traceback if committing
            # the failure record itself also failed.
            pass
        raise
