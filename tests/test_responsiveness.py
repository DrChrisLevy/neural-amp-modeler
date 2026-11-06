"""Simulate slow cloud I/O while independent browser requests keep moving."""

import asyncio
import io
import threading
import time
from contextlib import asynccontextmanager

import httpx
import numpy as np
import pytest
import soundfile as sf

from nam_studio.artifacts import Artifacts
from nam_studio.server import create_app
from nam_studio.storage import atomic_json


@asynccontextmanager
async def studio(tmp_path, **kwargs):
    app = create_app(tmp_path, **kwargs)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            yield app, client


async def responsive(client, path="/api/config", **kwargs):
    started = time.monotonic()
    response = await asyncio.wait_for(client.get(path, **kwargs), timeout=0.5)
    assert time.monotonic() - started < 0.5
    assert response.status_code == 200
    return response


def test_initial_scan_is_shared_and_does_not_block_other_requests(tmp_path, monkeypatch):
    async def run():
        async with studio(tmp_path) as (app, client):
            started, release = threading.Event(), threading.Event()
            original, calls = app.state.store.snapshot, []

            def slow_scan():
                calls.append(1)
                started.set()
                assert release.wait(5)
                return original()

            monkeypatch.setattr(app.state.store, "snapshot", slow_scan)
            requests = [asyncio.create_task(client.get("/api/state")) for _ in range(16)]
            try:
                assert await asyncio.to_thread(started.wait, 1)
                await responsive(client)
                assert len(calls) == 1
                requests[0].cancel()
                with pytest.raises(asyncio.CancelledError):
                    await requests[0]
            finally:
                release.set()
            responses = await asyncio.gather(*requests[1:])
            assert all(r.status_code == 200 for r in responses)
            revision = responses[0].json()["revision"]
            for _ in range(3):
                unchanged = (await responsive(client, "/api/state", params={"revision": revision})).json()
                assert "captures" not in unchanged and "jobs" not in unchanged
            assert len(calls) == 1

    asyncio.run(run())


def test_live_progress_refresh_is_nonblocking_and_completion_keeps_curves(tmp_path, monkeypatch):
    async def run():
        async with studio(tmp_path) as (app, client):
            take = (await client.post("/api/demo")).json()
            job_id = "a" * 32
            path = app.state.store.path("jobs", job_id) / "status.json"
            job = {"id": job_id, "capture_id": take["id"], "status": "training", "epoch": 1, "metrics": [{"epoch": 1, "loss": 0.4}]}
            atomic_json(path, job)
            initial = (await client.get("/api/state")).json()
            app.state.catalog.active_interval = 0
            started, release = threading.Event(), threading.Event()
            original, calls = app.state.store.job, []

            def slow_read(job_id):
                calls.append(job_id)
                started.set()
                assert release.wait(5)
                return original(job_id)

            monkeypatch.setattr(app.state.store, "job", slow_read)
            try:
                response = await responsive(client, "/api/state", params={"revision": initial["revision"]})
                assert "jobs" not in response.json()
                assert await asyncio.to_thread(started.wait, 1)
                await asyncio.gather(*(responsive(client, "/api/state") for _ in range(16)))
                assert calls == [job_id]
                job.update(status="completed", epoch=100, metrics=[*job["metrics"], {"epoch": 100, "loss": 0.004}])
                atomic_json(path, job)
            finally:
                release.set()
            await app.state.catalog.task
            latest = (await client.get("/api/state", params={"revision": initial["revision"]})).json()
            assert latest["revision"] != initial["revision"]
            assert latest["jobs"][0]["status"] == "completed"
            assert latest["jobs"][0]["metrics"] == job["metrics"]
            assert latest["jobs"][0]["prediction_url"].endswith("prediction.wav")
            await client.get("/api/state")
            assert calls == [job_id]  # Terminal models are not reread every second.

    asyncio.run(run())


@pytest.mark.parametrize("cancel_request", [False, True])
def test_slow_upload_does_not_block_controls_and_survives_disconnect(tmp_path, monkeypatch, cancel_request):
    async def run():
        async with studio(tmp_path) as (app, client):
            revision = (await client.get("/api/state")).json()["revision"]
            stream = io.BytesIO()
            sf.write(stream, 0.1 * np.sin(np.arange(48000) * 0.03), 48000, format="WAV", subtype="PCM_24")
            started, release = threading.Event(), threading.Event()
            original = app.state.store.capture

            def slow_save(*args, **kwargs):
                started.set()
                assert release.wait(5)
                return original(*args, **kwargs)

            monkeypatch.setattr(app.state.store, "capture", slow_save)
            upload = asyncio.create_task(client.post("/api/captures/upload", files={name: (f"{name}.wav", stream.getvalue()) for name in ("dry", "wet")}))
            try:
                assert await asyncio.to_thread(started.wait, 1)
                await responsive(client)
                await responsive(client, "/api/state")
                if cancel_request:
                    upload.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await upload
            finally:
                release.set()
            take = None if cancel_request else (await upload).json()
            async with asyncio.timeout(1):
                while True:
                    state = (await client.get("/api/state", params={"revision": revision})).json()
                    if state.get("captures"):
                        break
                    await asyncio.sleep(0.005)
            assert len(state["captures"]) == 1
            assert take is None or state["captures"][0]["id"] == take["id"]
    asyncio.run(run())


def test_audio_ranges_share_one_copy_and_do_not_block_state(tmp_path, monkeypatch):
    async def run():
        async with studio(tmp_path, key="test-key") as (app, client):
            client.headers["Authorization"] = "Bearer test-key"
            take = (await client.post("/api/demo")).json()
            await client.get("/api/state")
            source = app.state.store.path("captures", take["id"]) / "dry.wav"
            expected = source.read_bytes()
            started, release = threading.Event(), threading.Event()
            original, calls = app.state.artifacts.copy, []

            def slow_copy(source, destination):
                calls.append(1)
                started.set()
                assert release.wait(5)
                original(source, destination)

            monkeypatch.setattr(app.state.artifacts, "copy", slow_copy)
            requests = [asyncio.create_task(client.get(take["dry_url"], headers={"Range": "bytes=0-63"})) for _ in range(2)]
            try:
                assert await asyncio.to_thread(started.wait, 1)
                await responsive(client, "/api/state")
                await responsive(client)
            finally:
                release.set()
            for response in await asyncio.gather(*requests):
                assert response.status_code == 206
                assert response.content == expected[:64]
                assert "content-encoding" not in response.headers
                assert response.headers["content-range"] == f"bytes 0-63/{len(expected)}"
            source.unlink()  # Further seeks must use the detached local copy.
            response = await client.get(take["dry_url"], headers={"Range": "bytes=64-127"})
            assert response.status_code == 206 and response.content == expected[64:128]
            assert calls == [1]
            assert (await client.get(take["dry_url"], headers={"Authorization": "Bearer wrong"})).status_code == 401
            assert all(entry["users"] == 0 for entry in app.state.artifacts.entries.values())
    asyncio.run(run())


def test_training_dispatch_allows_parallel_runs_without_blocking_controls(tmp_path):
    async def run():
        started, both_started, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        submitted = []

        async def submit(job_id):
            submitted.append(job_id)
            started.set()
            if len(submitted) == 2:
                both_started.set()
            await release.wait()

        async with studio(tmp_path, submit=submit) as (app, client):
            take = (await client.post("/api/demo")).json()
            await client.get("/api/state")
            first = asyncio.create_task(client.post("/api/train", json={"capture_id": take["id"]}))
            second = None
            try:
                await asyncio.wait_for(started.wait(), 1)
                status = (await responsive(client, "/api/state")).json()
                assert status["jobs"][0]["status"] == "queued"
                second = asyncio.create_task(client.post("/api/train", json={"capture_id": take["id"], "model_size": "full", "epochs": 50}))
                await asyncio.wait_for(both_started.wait(), 1)
                status = (await responsive(client, "/api/state")).json()
                assert len(status["jobs"]) == 2
                assert {job["id"] for job in status["jobs"]} == set(submitted)
                assert all(job["status"] == "queued" for job in status["jobs"])
                await responsive(client)
            finally:
                release.set()
            assert (await first).status_code == 200
            assert (await second).status_code == 200
    asyncio.run(run())


def test_audio_cache_evicts_without_removing_files_being_played():
    async def run():
        cache = Artifacts(max_bytes=4)
        async def copy(path):
            path.write_bytes(b"test")
        try:
            playing = await cache.acquire("playing", copy)
            other = await cache.acquire("other", copy)
            await cache.release("other")
            assert playing.exists() and not other.exists()
            await cache.release("playing")
            assert cache.entries["playing"]["users"] == 0
        finally:
            await cache.close()
    asyncio.run(run())


@pytest.mark.parametrize("cancel_waiters", [False, True])
def test_cache_eviction_respects_waiters_and_cleans_up_abandoned_copies(cancel_waiters):
    async def run():
        cache = Artifacts(max_bytes=0)
        started, release = asyncio.Event(), asyncio.Event()
        async def copy(path):
            started.set()
            await release.wait()
            path.write_bytes(b"test")
        async def use():
            path = await cache.acquire("shared", copy)
            assert path.read_bytes() == b"test"
            # Immediate release must not evict before other waiters can pin it.
            await cache.release("shared")
        requests = [asyncio.create_task(use()) for _ in range(3)]
        try:
            await started.wait()
            if cancel_waiters:
                for request in requests:
                    request.cancel()
                await asyncio.gather(*requests, return_exceptions=True)
            release.set()
            if cancel_waiters:
                await asyncio.gather(*list(cache.pending.values()))
            else:
                await asyncio.gather(*requests)
            assert not cache.entries and not cache.waiters
        finally:
            release.set()
            await cache.close()
    asyncio.run(run())
