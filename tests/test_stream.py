"""Exercise bridge packets against the real server without opening audio hardware."""

import time
import asyncio
import threading
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tone_twin.bridge import BLOCK_SIZE, pack_frames, synthetic_frames
from tone_twin.server import create_app
from tone_twin.storage import SAMPLE_RATE


@contextmanager
def studio(tmp_path, submit=None, warm=None):
    app = create_app(tmp_path, key="test-stream-key", submit=submit, warm=warm)
    with TestClient(app) as client:
        client.headers["Authorization"] = "Bearer test-stream-key"
        yield app, client


def connect(ws, name="Synthetic test source"):
    ws.send_json({
        "type": "hello", "token": "test-stream-key", "source": "demo",
        "device": name, "sample_rate": SAMPLE_RATE, "channels": 2,
    })
    assert ws.receive_json() == {"type": "ready"}


def send_samples(ws, sample_start, count):
    end = sample_start + count
    while sample_start < end:
        size = min(BLOCK_SIZE, end - sample_start)
        ws.send_bytes(pack_frames(sample_start, synthetic_frames(sample_start, size)))
        sample_start += size
    return sample_start


def wait_state(client, condition):
    deadline = time.monotonic() + 5
    state = None
    while time.monotonic() < deadline:
        response = client.get("/api/state")
        assert response.status_code == 200
        state = response.json()
        if condition(state):
            return state
        time.sleep(0.005)
    pytest.fail(f"Stream state did not settle: {state}")


def start_record(client, ws):
    response = client.post("/api/captures/record", json={"action": "start", "name": "Synthetic stream test"})
    assert response.status_code == 200
    assert ws.receive_json() == {"type": "record", "action": "start"}
    return response.json()["id"]


def assert_saved_pair(app, capture, start, count):
    expected = synthetic_frames(start, count)
    folder = app.state.store.path("captures", capture["id"])
    for column, name in enumerate(("dry", "wet")):
        actual, rate = sf.read(folder / f"{name}.wav", dtype="float32")
        assert rate == SAMPLE_RATE
        assert len(actual) == count
        # Stored deliverables are PCM24; allow exactly its quantization precision.
        np.testing.assert_allclose(actual, expected[:, column], atol=2 ** -23, rtol=0)
    assert capture["duration"] == count / SAMPLE_RATE


def test_stream_start_stop_preserves_all_recorded_samples(tmp_path):
    with studio(tmp_path) as (app, client):
        with client.websocket_connect("/ws/bridge") as ws:
            connect(ws)
            # Monitoring precedes recording; the stream sequence never resets.
            offset = send_samples(ws, 0, BLOCK_SIZE)
            start_record(client, ws)
            count = 60_123
            offset = send_samples(ws, offset, count)
            wait_state(client, lambda state: state["bridge"]["samples"] == count)
            response = client.post("/api/captures/record", json={"action": "stop"})
            assert response.status_code == 200
            capture = response.json()["capture"]
            assert ws.receive_json() == {"type": "record", "action": "stop"}
            assert capture["status"] == "ready"
            assert capture["source"] == "demo"
            assert_saved_pair(app, capture, BLOCK_SIZE, count)
            assert client.get(capture["dry_url"]).status_code == 200

            # A second take must follow the same monotonically advancing stream.
            start_record(client, ws)
            second_start = offset
            send_samples(ws, offset, SAMPLE_RATE)
            wait_state(client, lambda state: state["bridge"]["samples"] == SAMPLE_RATE)
            second = client.post("/api/captures/record", json={"action": "stop"}).json()["capture"]
            assert ws.receive_json()["action"] == "stop"
            assert_saved_pair(app, second, second_start, SAMPLE_RATE)


@pytest.mark.parametrize("fault", ["sequence_skip", "gap_notice", "invalid_levels", "disconnect"])
def test_interrupted_takes_are_saved_but_cannot_train(tmp_path, fault):
    submitted = []

    async def submit(job_id):
        submitted.append(job_id)

    with studio(tmp_path, submit=submit) as (app, client):
        with client.websocket_connect("/ws/bridge") as ws:
            connect(ws)
            capture_id = start_record(client, ws)
            count = 12 * SAMPLE_RATE
            offset = send_samples(ws, 0, count)
            wait_state(client, lambda state: state["bridge"]["samples"] == count)
            if fault == "sequence_skip":
                ws.send_bytes(pack_frames(offset + BLOCK_SIZE, synthetic_frames(offset + BLOCK_SIZE)))
                assert ws.receive_json()["type"] == "error"
            elif fault == "gap_notice":
                ws.send_json({"type": "gap", "dropped_frames": BLOCK_SIZE})
                assert ws.receive_json()["type"] == "error"
            elif fault == "invalid_levels":
                invalid = synthetic_frames(offset)
                invalid[0, 0] = 1.0005  # Previously accepted by ingestion but rejected on save.
                ws.send_bytes(pack_frames(offset, invalid))
                assert ws.receive_json()["type"] == "error"
            # Leaving the context is an unplanned disconnect for the third case.
        state = wait_state(client, lambda state: not state["bridge"]["connected"] and bool(state["captures"]))
        capture = next(item for item in state["captures"] if item["id"] == capture_id)
        assert capture["status"] == "interrupted"
        assert_saved_pair(app, capture, 0, count)
        response = client.post("/api/train", json={"capture_id": capture_id})
        assert response.status_code == 400
        assert "uninterrupted" in response.json()["detail"]
        assert submitted == []


def test_duration_limit_trims_only_the_final_frame_and_keeps_bridge_connected(tmp_path):
    with patch("tone_twin.server.MAX_SECONDS", 1):
        with studio(tmp_path) as (app, client):
            with client.websocket_connect("/ws/bridge") as ws:
                connect(ws)
                start_record(client, ws)
                # 48,000 is not a multiple of the native bridge's 2,048-frame block.
                count = 24 * BLOCK_SIZE
                send_samples(ws, 0, count)
                assert ws.receive_json() == {"type": "record", "action": "stop"}
                state = wait_state(client, lambda state: bool(state["captures"]))
                assert state["bridge"]["connected"]
                assert not state["bridge"]["recording"]
                capture = state["captures"][0]
                assert capture["status"] == "ready"
                assert_saved_pair(app, capture, 0, SAMPLE_RATE)

                start_record(client, ws)
                send_samples(ws, count, 24 * BLOCK_SIZE)
                assert ws.receive_json()["action"] == "stop"
                state = wait_state(client, lambda state: len(state["captures"]) == 2)
                latest = next(item for item in state["captures"] if item["id"] != capture["id"])
                assert_saved_pair(app, latest, count, SAMPLE_RATE)


def test_second_bridge_is_rejected_and_disconnect_allows_replacement(tmp_path):
    with studio(tmp_path) as (app, client):
        with client.websocket_connect("/ws/bridge") as first:
            connect(first, "First synthetic bridge")
            with client.websocket_connect("/ws/bridge") as second:
                second.send_json({
                    "type": "hello", "token": "test-stream-key", "source": "demo",
                    "device": "Rejected bridge", "sample_rate": SAMPLE_RATE, "channels": 2,
                })
                with pytest.raises(WebSocketDisconnect) as closed:
                    second.receive_json()
                assert closed.value.code == 1008
            state = client.get("/api/state").json()
            assert state["bridge"]["device"] == "First synthetic bridge"
            assert state["bridge"]["connected"]
        wait_state(client, lambda state: not state["bridge"]["connected"])
        with client.websocket_connect("/ws/bridge") as replacement:
            connect(replacement, "Replacement bridge")
            start_record(client, replacement)
            send_samples(replacement, 0, SAMPLE_RATE)
            wait_state(client, lambda state: state["bridge"]["samples"] == SAMPLE_RATE)
            capture = client.post("/api/captures/record", json={"action": "stop"}).json()["capture"]
            assert replacement.receive_json()["action"] == "stop"
            assert_saved_pair(app, capture, 0, SAMPLE_RATE)


def test_finalization_error_releases_single_bridge_slot(tmp_path):
    with studio(tmp_path) as (app, client):
        with patch.object(app.state.store, "capture", side_effect=ValueError("Simulated finalization failure")):
            try:
                with client.websocket_connect("/ws/bridge") as ws:
                    connect(ws)
                    start_record(client, ws)
                    send_samples(ws, 0, SAMPLE_RATE)
                    wait_state(client, lambda state: state["bridge"]["samples"] == SAMPLE_RATE)
            except ValueError:
                # An error can propagate to the caller, but must not strand ownership.
                pass
        state = wait_state(client, lambda state: not state["bridge"]["connected"])
        assert not state["bridge"]["recording"]
        with client.websocket_connect("/ws/bridge") as replacement:
            connect(replacement)


@pytest.mark.parametrize("stop_mode", ["stop", "disconnect"])
def test_worker_warmup_does_not_block_capture_and_is_cancelled_when_it_ends(tmp_path, stop_mode):
    started, cancelled = threading.Event(), threading.Event()

    async def warm():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    with studio(tmp_path, warm=warm) as (app, client):
        with client.websocket_connect("/ws/bridge") as ws:
            connect(ws)
            assert not started.is_set()  # Merely monitoring never warms a GPU.
            start_record(client, ws)
            assert started.wait(1)
            send_samples(ws, 0, SAMPLE_RATE)
            wait_state(client, lambda state: state["bridge"]["samples"] == SAMPLE_RATE)
            if stop_mode == "stop":
                assert client.post("/api/captures/record", json={"action": "stop"}).status_code == 200
                assert ws.receive_json()["action"] == "stop"
        assert cancelled.wait(1)
        state = wait_state(client, lambda state: len(state["captures"]) == 1)
        assert_saved_pair(app, state["captures"][0], 0, SAMPLE_RATE)


def test_warmup_failure_does_not_fail_recording(tmp_path):
    attempted = threading.Event()

    async def warm():
        attempted.set()
        raise RuntimeError("GPU capacity temporarily unavailable")

    with studio(tmp_path, warm=warm) as (app, client):
        with client.websocket_connect("/ws/bridge") as ws:
            connect(ws)
            start_record(client, ws)
            assert attempted.wait(1)
            send_samples(ws, 0, SAMPLE_RATE)
            wait_state(client, lambda state: state["bridge"]["samples"] == SAMPLE_RATE)
            response = client.post("/api/captures/record", json={"action": "stop"})
            assert response.status_code == 200
            assert_saved_pair(app, response.json()["capture"], 0, SAMPLE_RATE)
