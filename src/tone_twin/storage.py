"""Small, inspectable recording store shared by the web app and training worker."""

from __future__ import annotations

import json
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 48_000
MAX_SECONDS = 600


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, allow_nan=False))
    temp.replace(path)


def identifier(value: str) -> str:
    if not re.fullmatch(r"[a-f0-9]{32}", value):
        raise ValueError("Invalid recording or model ID.")
    return value


def waveform(audio: np.ndarray, bins: int = 320) -> list[float]:
    if len(audio) == 0:
        return []
    return [round(float(np.max(np.abs(x))), 5) for x in np.array_split(audio, min(bins, len(audio)))]


def validate_pair(dry: np.ndarray, wet: np.ndarray, sample_rate: int) -> None:
    if sample_rate != SAMPLE_RATE:
        raise ValueError("Use 48 kHz WAV files from the Quad Cortex Mini.")
    if dry.ndim != 1 or wet.ndim != 1:
        raise ValueError("Choose one mono dry channel and one mono processed channel.")
    if len(dry) != len(wet):
        raise ValueError("Dry and processed files must cover exactly the same recording.")
    if not SAMPLE_RATE <= len(dry) <= SAMPLE_RATE * MAX_SECONDS:
        raise ValueError("Record between 1 second and 10 minutes.")
    if not np.isfinite(dry).all() or not np.isfinite(wet).all():
        raise ValueError("The recording contains invalid audio samples.")
    if max(float(np.max(np.abs(dry))), float(np.max(np.abs(wet)))) > 1.0:
        raise ValueError("The recording exceeds digital full scale. Reduce its input level.")


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "captures").mkdir(parents=True, exist_ok=True)
        (self.root / "jobs").mkdir(parents=True, exist_ok=True)

    def path(self, kind: str, record_id: str) -> Path:
        if kind not in {"captures", "jobs"}:
            raise ValueError("Invalid resource type.")
        return self.root / kind / identifier(record_id)

    def capture(self, dry, wet, name: str, source: str, status: str = "ready", record_id: str | None = None) -> dict:
        validate_pair(dry, wet, SAMPLE_RATE)
        record_id = record_id or uuid.uuid4().hex
        folder = self.path("captures", record_id)
        folder.mkdir(parents=True, exist_ok=False)
        sf.write(folder / "dry.wav", dry, SAMPLE_RATE, subtype="PCM_24")
        sf.write(folder / "wet.wav", wet, SAMPLE_RATE, subtype="PCM_24")
        meta = {
            "id": record_id, "name": name.strip()[:100] or "Untitled take",
            "created_at": now(), "duration": len(dry) / SAMPLE_RATE,
            "sample_rate": SAMPLE_RATE, "source": source, "status": status,
            "peaks": {"dry": waveform(dry), "wet": waveform(wet)},
            "clipped": bool(np.max(np.abs(dry)) >= 0.999 or np.max(np.abs(wet)) >= 0.999),
        }
        atomic_json(folder / "meta.json", meta)
        return self.present_capture(meta)

    @staticmethod
    def present_capture(meta: dict) -> dict:
        return {**meta, **{f"{name}_url": f"/api/files/capture/{meta['id']}/{name}.wav" for name in ("dry", "wet")}}

    def captures(self) -> list[dict]:
        values = [self.present_capture(json.loads(p.read_text())) for p in (self.root / "captures").glob("*/meta.json")]
        return sorted(values, key=lambda x: x["created_at"], reverse=True)

    def jobs(self) -> list[dict]:
        values = []
        for path in (self.root / "jobs").glob("*/status.json"):
            values.append(self.present_job(json.loads(path.read_text())))
        return sorted(values, key=lambda x: x.get("created_at", ""), reverse=True)

    @staticmethod
    def present_job(job):
        # Completion is published only after all four artifacts are persisted.
        # Avoid four remote stat calls per historical job on every UI poll.
        if job["status"] == "completed":
            job = {**job, **{f"{name}_url": f"/api/files/job/{job['id']}/{file}" for name, file in
                           {"model": "model.nam", "prediction": "prediction.wav", "reference": "reference.wav", "dry": "dry.wav"}.items()}}
        return job

    def job(self, job_id):
        return self.present_job(json.loads((self.path("jobs", job_id) / "status.json").read_text()))

    def snapshot(self):
        captures = list((self.root / "captures").glob("*/meta.json"))
        jobs = list((self.root / "jobs").glob("*/status.json"))
        with ThreadPoolExecutor(max_workers=8) as pool:
            metadata = list(pool.map(lambda path: json.loads(path.read_text()), [*captures, *jobs]))
        return (
            sorted((self.present_capture(m) for m in metadata[:len(captures)]), key=lambda m: m["created_at"], reverse=True),
            sorted((self.present_job(m) for m in metadata[len(captures):]), key=lambda m: m.get("created_at", ""), reverse=True),
        )
