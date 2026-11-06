"""Same-origin studio UI, recording ingestion and background training dispatch."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import secrets
import struct
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.gzip import GZipMiddleware

from .artifacts import Artifacts, ArtifactResponse
from .catalog import Catalog
from .storage import MAX_SECONDS, SAMPLE_RATE, Store, atomic_json, identifier, now


class Login(BaseModel):
    key: str


class RecordCommand(BaseModel):
    action: str
    name: str = "Untitled take"


class TrainCommand(BaseModel):
    capture_id: str
    epochs: int = Field(default=100, ge=10, le=1000)
    model_size: str = "full"


class UICompression(GZipMiddleware):
    async def __call__(self, scope, receive, send):
        # Byte ranges must describe the original WAV, not a gzipped fragment.
        if scope.get("path", "").startswith("/api/files/"):
            await self.app(scope, receive, send)
        else:
            await super().__call__(scope, receive, send)


def create_app(root: str | Path | None = None, *, key: str | None = None, volume=None, submit=None, warm=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            await asyncio.gather(*list(finalizers), return_exceptions=True)
            await catalog.close()
            await artifacts.close()

    studio = FastAPI(title="Tone Twin", docs_url=None, redoc_url=None, lifespan=lifespan)
    studio.add_middleware(UICompression, minimum_size=1000, compresslevel=3)
    store = Store(root or os.environ.get("TONE_TWIN_DATA", ".data"))
    access_key = key if key is not None else os.environ.get("TONE_TWIN_KEY", "")
    storage_lock = asyncio.Lock()
    bridge = {"connected": False, "device": None, "sample_rate": SAMPLE_RATE, "recording": False, "capture_id": None, "levels": {"dry": 0, "wet": 0}, "samples": 0}
    bridge_ws = None
    monitors: set[WebSocket] = set()
    recording = None
    finalizers: set[asyncio.Task] = set()
    app_cookie = "tone_twin_key"
    launch_tickets: dict[str, float] = {}
    if os.environ.get("TONE_TWIN_MODAL") and not access_key:
        raise RuntimeError("Hosted Tone Twin requires the tone-twin-auth Modal secret.")

    async def keep_worker_ready(current):
        # Warm only during a deliberate recording, with a wall-time bound even
        # if a connected bridge stops delivering samples without disconnecting.
        deadline = time.monotonic() + MAX_SECONDS
        while recording is current and time.monotonic() < deadline:
            try:
                await asyncio.wait_for(warm(), timeout=60)
            except Exception as exc:
                logging.getLogger(__name__).warning("Training warm-up failed (%s); recording continues", type(exc).__name__)
            await asyncio.sleep(min(40, max(0, deadline - time.monotonic())))

    def authorized(token: str | None) -> bool:
        return not access_key or (isinstance(token, str) and secrets.compare_digest(token, access_key))

    def request_token(request) -> str | None:
        header = request.headers.get("authorization", "")
        return header[7:] if header.startswith("Bearer ") else request.cookies.get(app_cookie)

    async def reload_store():
        if volume is not None:
            await volume.reload.aio()

    async def commit_store():
        if volume is not None:
            await volume.commit.aio()

    catalog = Catalog(store, storage_lock, reload_store)
    artifacts = Artifacts()

    def operation_finished(task):
        finalizers.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logging.getLogger(__name__).error("Studio write failed", exc_info=task.exception())

    async def persist(operation):
        # Keep ownership and filesystem locks until a write has really finished,
        # even if the initiating browser or audio connection disappears.
        task = asyncio.create_task(operation)
        finalizers.add(task)
        task.add_done_callback(operation_finished)
        return await asyncio.shield(task)

    async def save_capture(*args, **kwargs):
        async with storage_lock:
            result = await asyncio.to_thread(store.capture, *args, **kwargs)
            await commit_store()
            catalog.put_capture(result)
            return result

    @studio.middleware("http")
    async def auth_and_headers(request: Request, call_next):
        if request.url.path.startswith("/api/") and request.url.path not in {"/api/config", "/api/auth"}:
            if not authorized(request_token(request)):
                return JSONResponse({"detail": "Unlock your workspace to continue."}, status_code=401)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    @studio.exception_handler(ValueError)
    async def invalid_input(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @studio.get("/api/config")
    async def config(request: Request):
        return {"auth_required": bool(access_key), "authenticated": authorized(request_token(request)), "training_available": submit is not None, "version": "0.1.0"}

    @studio.post("/api/auth")
    async def login(body: Login, request: Request):
        if not authorized(body.key):
            await asyncio.sleep(0.3)
            raise HTTPException(401, "That workspace key is incorrect.")
        response = JSONResponse({"ok": True})
        response.set_cookie(app_cookie, body.key, httponly=True, secure=request.url.scheme == "https", samesite="strict", max_age=7 * 86400)
        return response

    @studio.post("/api/launch")
    async def launch(request: Request):
        # A short-lived, single-use handoff lets the local CLI open the hosted
        # studio without putting the durable workspace key in a browser URL.
        for ticket, expiry in list(launch_tickets.items()):
            if expiry < time.monotonic():
                launch_tickets.pop(ticket, None)
        ticket = secrets.token_urlsafe(24)
        launch_tickets[ticket] = time.monotonic() + 60
        return {"url": f"{str(request.base_url).rstrip('/')}/launch?ticket={ticket}"}

    @studio.get("/launch")
    async def redeem(request: Request, ticket: str):
        expiry = launch_tickets.pop(ticket, 0)
        if expiry < time.monotonic():
            raise HTTPException(401, "This launch link expired. Open the studio again from your project.")
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(app_cookie, access_key, httponly=True, secure=request.url.scheme == "https", samesite="strict", max_age=7 * 86400)
        response.headers["Cache-Control"] = "no-store"
        return response

    @studio.get("/api/state")
    async def state(revision: str | None = None):
        await catalog.get()
        result = {"revision": catalog.revision, "bridge": dict(bridge)}
        if revision != catalog.revision:
            result.update(captures=catalog.captures, jobs=catalog.jobs)
        return JSONResponse(result)

    async def broadcast(payload):
        async def send(ws):
            try:
                await asyncio.wait_for(ws.send_json(payload), 0.5)
            except Exception:
                monitors.discard(ws)
        await asyncio.gather(*(send(ws) for ws in list(monitors)))

    @studio.websocket("/ws/monitor")
    async def monitor(ws: WebSocket):
        if not authorized(ws.cookies.get(app_cookie)):
            await ws.close(code=1008)
            return
        try:
            await ws.accept()
        except (WebSocketDisconnect, RuntimeError):
            return
        monitors.add(ws)
        await ws.send_json({"type": "state", "bridge": bridge})
        try:
            while True:
                await ws.receive_text()
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            monitors.discard(ws)

    async def finish_recording(interrupted=False):
        nonlocal recording
        current, recording = recording, None
        bridge["recording"] = False
        bridge["capture_id"] = None
        if current is None:
            return None
        if current.get("warm_task"):
            current["warm_task"].cancel()
        current["file"].close()

        # Persist independently of the HTTP/WebSocket request. A disconnected
        # client must not cancel the save once its audio has been recorded.
        return await persist(save_recording(current, interrupted))

    async def save_recording(current, interrupted):
        try:
            audio, _ = await asyncio.to_thread(sf.read, current["path"], dtype="float32", always_2d=True)
            if len(audio) < SAMPLE_RATE:
                return None
            result = await save_capture(audio[:, 0], audio[:, 1], current["name"], current["source"], "interrupted" if interrupted else "ready", current["id"])
            await broadcast({"type": "state", "bridge": bridge})
            return result
        finally:
            Path(current["path"]).unlink(missing_ok=True)

    @studio.post("/api/captures/record")
    async def record(body: RecordCommand):
        nonlocal recording
        if body.action == "stop":
            result = await finish_recording()
            if bridge_ws:
                await bridge_ws.send_json({"type": "record", "action": "stop"})
            return {"capture": result}
        if body.action != "start":
            raise HTTPException(400, "Choose start or stop.")
        if not bridge["connected"]:
            raise HTTPException(409, "Connect the local audio bridge first.")
        if recording is not None:
            raise HTTPException(409, "A take is already recording.")
        record_id = uuid.uuid4().hex
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        recording = {"id": record_id, "name": body.name, "path": path, "source": bridge.get("source", "bridge"), "samples": 0, "file": sf.SoundFile(path, mode="w", samplerate=SAMPLE_RATE, channels=2, subtype="FLOAT")}
        bridge.update(recording=True, capture_id=record_id, samples=0)
        await bridge_ws.send_json({"type": "record", "action": "start"})
        if warm is not None:
            recording["warm_task"] = asyncio.create_task(keep_worker_ready(recording))
        await broadcast({"type": "state", "bridge": bridge})
        return {"id": record_id}

    @studio.websocket("/ws/bridge")
    async def capture_bridge(ws: WebSocket):
        nonlocal bridge_ws
        try:
            await ws.accept()
        except (WebSocketDisconnect, RuntimeError):
            return
        try:
            hello = await asyncio.wait_for(ws.receive_json(), 10)
        except Exception:
            await ws.close(code=1008)
            return
        if not authorized(hello.get("token")) or hello.get("type") != "hello":
            await ws.close(code=1008)
            return
        if bridge_ws is not None or hello.get("sample_rate") != SAMPLE_RATE or hello.get("channels") != 2:
            await ws.close(code=1008, reason="Use one 48 kHz, two-channel bridge.")
            return
        bridge_ws = ws
        bridge.update(connected=True, device=str(hello.get("device", "Audio interface"))[:100], source=hello.get("source", "bridge"))
        await ws.send_json({"type": "ready"})
        await broadcast({"type": "state", "bridge": bridge})
        expected, last_update = 0, 0.0
        try:
            while True:
                packet = await ws.receive()
                if packet["type"] == "websocket.disconnect":
                    break
                if packet.get("text"):
                    notice = json.loads(packet["text"])
                    if notice.get("type") == "gap":
                        raise ValueError("Audio dropped. This take is incomplete; record again.")
                    continue
                data = packet.get("bytes", b"")
                if len(data) < 12 or len(data) > 12 + 16_384 * 8:
                    raise ValueError("Invalid audio frame.")
                start, count = struct.unpack("<QI", data[:12])
                if start != expected or count == 0 or len(data) != 12 + count * 8:
                    raise ValueError("Audio sequence interrupted. Record a new take.")
                audio = np.frombuffer(data, dtype="<f4", offset=12).reshape(count, 2)
                if not np.isfinite(audio).all() or np.max(np.abs(audio)) > 1.0:
                    raise ValueError("Invalid audio levels.")
                expected += count
                if recording is not None:
                    remaining = MAX_SECONDS * SAMPLE_RATE - recording["samples"]
                    recording["file"].write(audio[:remaining])
                    recording["samples"] += min(count, remaining)
                    bridge["samples"] = recording["samples"]
                    if recording["samples"] >= MAX_SECONDS * SAMPLE_RATE:
                        await finish_recording()
                        await ws.send_json({"type": "record", "action": "stop"})
                if time.monotonic() - last_update > 0.08:
                    dry_peak, wet_peak = np.max(np.abs(audio), axis=0).tolist()
                    bridge["levels"] = {"dry": dry_peak, "wet": wet_peak}
                    await broadcast({"type": "levels", "dry": dry_peak, "wet": wet_peak, "peaks": np.max(np.abs(audio.reshape(-1, 2)), axis=1)[::max(1, count // 48)].tolist(), "samples": bridge["samples"]})
                    last_update = time.monotonic()
        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception as exc:
            try:
                await ws.send_json({"type": "error", "message": str(exc)})
            except Exception:
                pass
        finally:
            try:
                await finish_recording(interrupted=True)
            finally:
                bridge_ws = None
                bridge.update(connected=False, device=None, recording=False, levels={"dry": 0, "wet": 0})
                await broadcast({"type": "state", "bridge": bridge})

    @studio.post("/api/captures/upload")
    async def upload(dry: UploadFile = File(...), wet: UploadFile = File(...), name: str = Form("Untitled take")):
        audio = []
        for file in (dry, wet):
            contents = await file.read(128 * 1024 * 1024 + 1)
            if len(contents) > 128 * 1024 * 1024:
                raise HTTPException(413, "Each WAV must be smaller than 128 MB.")
            try:
                values, sr = await asyncio.to_thread(sf.read, io.BytesIO(contents), dtype="float32")
            except Exception as exc:
                raise HTTPException(400, "Upload readable WAV audio files.") from exc
            if sr != SAMPLE_RATE:
                raise HTTPException(400, "Export both files at 48 kHz.")
            audio.append(values)
        return await persist(save_capture(*audio, name=name, source="upload"))

    @studio.post("/api/demo")
    async def demo():
        rng = np.random.default_rng(42)
        t = np.arange(SAMPLE_RATE * 16) / SAMPLE_RATE
        phase = 2 * np.pi * np.cumsum(np.repeat([110, 146.83, 164.81, 130.81], SAMPLE_RATE * 4)) / SAMPLE_RATE
        envelope = np.exp(-3 * (t % 0.5))
        dry = (0.11 * np.sin(phase) + 0.04 * np.sin(2 * phase) + 0.02 * rng.normal(size=len(t))) * envelope
        wet = np.tanh(dry * 4) * 0.6
        return await persist(save_capture(dry.astype("float32"), wet.astype("float32"), "Demo · synthetic plucks", "demo"))

    @studio.post("/api/train")
    async def train(body: TrainCommand):
        if submit is None:
            raise HTTPException(503, "Training is available in your deployed Modal workspace.")
        if body.model_size not in {"lite", "full", "both"}:
            raise HTTPException(400, "Choose lite, full, or both.")
        identifier(body.capture_id)
        return await persist(start_training(body))

    async def start_training(body):
        await catalog.get(fresh=True)
        async with storage_lock:
            capture = next((c for c in catalog.captures if c["id"] == body.capture_id), None)
            if capture is None:
                raise HTTPException(404, "Take not found.")
            if capture["status"] != "ready" or capture["duration"] < 12:
                raise HTTPException(400, "Record at least 12 uninterrupted seconds. A few varied minutes work best.")
            if capture.get("clipped"):
                raise HTTPException(400, "This take is clipping. Lower the input level and record again.")
            if any(job["status"] in {"queued", "training"} for job in catalog.jobs):
                raise HTTPException(409, "A model is already training. Let it finish first.")
            job = {"id": uuid.uuid4().hex, "capture_id": body.capture_id, "name": capture["name"], "status": "queued", "progress": 0, "epoch": 0, "epochs": body.epochs, "model_size": body.model_size, "created_at": now(), "metrics": []}
            status_path = store.path("jobs", job["id"]) / "status.json"
            await asyncio.to_thread(atomic_json, status_path, job)
            await commit_store()
            catalog.put_job(job)
        try:
            await submit(job["id"])
        except Exception as exc:
            async with storage_lock:
                failed = {**job, "status": "failed", "error": "Could not start the training worker. Try again."}
                await asyncio.to_thread(atomic_json, status_path, failed)
                await commit_store()
                catalog.put_job(failed)
            raise HTTPException(503, "Could not start the training worker.") from exc
        return job

    @studio.get("/api/files/{kind}/{record_id}/{filename}")
    async def audio_file(kind: str, record_id: str, filename: str):
        if kind not in {"capture", "job"} or filename not in {"dry.wav", "wet.wav", "prediction.wav", "reference.wav", "model.nam"}:
            raise HTTPException(404)
        path = store.path("captures" if kind == "capture" else "jobs", record_id) / filename
        key = f"{kind}-{record_id}-{filename}"

        async def copy(destination):
            async with storage_lock:
                try:
                    await asyncio.to_thread(artifacts.copy, path, destination)
                except FileNotFoundError:
                    # Only reload on a miss, rather than on every range request.
                    await reload_store()
                    try:
                        await asyncio.to_thread(artifacts.copy, path, destination)
                    except FileNotFoundError as exc:
                        raise HTTPException(404) from exc

        local = await artifacts.acquire(key, copy)
        return ArtifactResponse(local, release=lambda: artifacts.release(key), media_type="application/json" if filename.endswith(".nam") else "audio/wav", filename=filename, content_disposition_type="inline", headers={"Cache-Control": "private, max-age=86400, immutable"})

    static = Path(__file__).parent / "static"
    static.mkdir(exist_ok=True)

    @studio.get("/")
    async def index():
        path = static / "index.html"
        if not path.exists():
            return JSONResponse({"name": "Tone Twin", "status": "Studio design in progress"})
        return FileResponse(path)

    studio.mount("/static", StaticFiles(directory=static), name="static")
    studio.state.store = store
    studio.state.catalog = catalog
    studio.state.artifacts = artifacts
    return studio


app = create_app()
