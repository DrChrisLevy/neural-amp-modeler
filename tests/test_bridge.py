import asyncio
import json
from unittest.mock import patch

import numpy as np
import pytest
from websockets.asyncio.server import serve

from nam_studio.bridge import (
    BLOCK_SIZE,
    FRAME_HEADER,
    BridgeError,
    CaptureBuffer,
    InputDevice,
    _send_capture,
    pack_frames,
    resolve_token,
    run_bridge,
    select_device,
    synthetic_frames,
    validate_channels,
    websocket_url,
)


def test_origin_validation_and_secure_mapping():
    assert websocket_url("https://example.com/") == "wss://example.com/ws/bridge"
    assert websocket_url("http://localhost:8000") == "ws://localhost:8000/ws/bridge"
    assert websocket_url("http://[::1]:8000") == "ws://[::1]:8000/ws/bridge"
    for value in (
        "file:///tmp/server", "https://example.com/path", "https://example.com?token=secret",
        "https://user:secret@example.com", "https://example.com#fragment", "http://localhost:bad",
        "http://localhost:0", "example.com", " https://example.com",
    ):
        with pytest.raises(BridgeError):
            websocket_url(value)


def test_auto_selection_never_falls_back_to_microphone():
    microphone = InputDevice(0, "Laptop microphone", 1, 48000)
    cortex = InputDevice(3, "Quad Cortex mini", 8, 48000)
    with pytest.raises(BridgeError, match="No Quad Cortex"):
        select_device([microphone], None)
    assert select_device([microphone, cortex], None) == cortex
    assert select_device([microphone, cortex], "3") == cortex
    assert select_device([microphone, cortex], "cortex") == cortex
    with pytest.raises(BridgeError, match="More than one"):
        select_device([cortex, InputDevice(4, "Quad Cortex", 8, 48000)], None)
    with pytest.raises(BridgeError):
        select_device([microphone], "missing")


def test_channels_and_environment_token_validation():
    cortex = InputDevice(3, "Quad Cortex mini", 8, 48000)
    assert validate_channels(cortex, 1, 3) == 3
    for dry, wet in [(0, 3), (1, 1), (1, 9)]:
        with pytest.raises(BridgeError):
            validate_channels(cortex, dry, wet)
    assert resolve_token(None, {"NAM_STUDIO_KEY": "example-key"}) == "example-key"
    assert resolve_token("env:CUSTOM_KEY", {"CUSTOM_KEY": "another-key"}) == "another-key"
    assert resolve_token(None, {}) is None
    with pytest.raises(BridgeError):
        resolve_token("env:NOT_SET", {})


def test_packet_framing_channel_order_and_sample_continuity():
    buffer = CaptureBuffer(1, 3)
    source = np.arange(16, dtype=np.float32).reshape(4, 4)
    buffer.callback(source, 4, None, False)
    buffer.callback(source + 16, 4, None, False)
    packet = buffer.queue.get_nowait()
    assert FRAME_HEADER.unpack_from(packet) == (0, 4)
    np.testing.assert_array_equal(np.frombuffer(packet, dtype="<f4", offset=12).reshape(-1, 2), source[:, [0, 2]])
    assert FRAME_HEADER.unpack_from(buffer.queue.get_nowait()) == (4, 4)
    with pytest.raises(ValueError):
        pack_frames(0, np.zeros((4, 1)))


def test_synthetic_chunks_are_deterministic_and_contiguous():
    whole = synthetic_frames(0, BLOCK_SIZE * 2)
    split = np.concatenate((synthetic_frames(0), synthetic_frames(BLOCK_SIZE)))
    np.testing.assert_array_equal(whole, split)
    assert whole.shape == (BLOCK_SIZE * 2, 2)
    assert np.isfinite(whole).all()
    assert not np.array_equal(whole[:, 0], whole[:, 1])


def test_queue_overflow_reports_gap_and_stops():
    buffer = CaptureBuffer(1, 2, capacity=1)
    block = np.zeros((8, 2), dtype=np.float32)
    buffer.callback(block, 8, None, False)
    buffer.callback(block, 8, None, False)
    assert buffer.failed.is_set()

    class Socket:
        def __init__(self):
            self.messages = []

        async def send(self, value):
            self.messages.append(json.loads(value))

    socket = Socket()
    with pytest.raises(BridgeError, match="incomplete"):
        asyncio.run(_send_capture(socket, buffer))
    assert socket.messages == [{
        "type": "gap", "dropped_frames": 8,
        "reason": "capture_queue_overflow", "dropped_frames_unknown": False,
    }]


def test_native_overflow_is_not_treated_as_complete_audio():
    class Status:
        input_overflow = True

    buffer = CaptureBuffer(1, 2)
    buffer.callback(np.zeros((8, 2), dtype=np.float32), 8, None, Status())
    assert buffer.failed.is_set()
    assert buffer.queue.empty()
    assert buffer.gap_message()["dropped_frames_unknown"] is True


def test_stalled_audio_device_reports_incomplete_recording():
    class Socket:
        messages = []

        async def send(self, value):
            self.messages.append(json.loads(value))

    socket = Socket()
    with patch("nam_studio.bridge.CAPTURE_STALL_TIMEOUT", 0.01):
        with pytest.raises(BridgeError, match="stopped delivering"):
            asyncio.run(_send_capture(socket, CaptureBuffer(1, 2)))
    assert socket.messages[0]["type"] == "gap"
    assert socket.messages[0]["reason"] == "audio_device_stopped"


def test_synthetic_websocket_handshake_framing_and_record_status():
    async def exercise():
        received = []
        statuses = []

        async def handler(websocket):
            hello = json.loads(await websocket.recv())
            received.append(hello)
            assert websocket.request.path == "/ws/bridge"
            await websocket.send(json.dumps({"type": "ready"}))
            await websocket.send(json.dumps({"type": "record", "action": "start"}))
            received.append(await websocket.recv())
            received.append(await websocket.recv())
            await websocket.send(json.dumps({"type": "record", "action": "stop"}))
            await websocket.close()

        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            with patch("nam_studio.bridge.input_devices", side_effect=AssertionError("Hardware must not be touched")):
                with pytest.raises(BridgeError, match="disconnected"):
                    await asyncio.wait_for(run_bridge(
                        f"http://127.0.0.1:{port}", synthetic=True, token="integration-key", emit=statuses.append,
                    ), timeout=5)

        assert received[0]["source"] == "demo"
        assert received[0]["device"] == "Synthetic test source"
        assert received[0]["sample_rate"] == 48000
        assert received[0]["channels"] == 2
        assert received[0]["token"] == "integration-key"
        for sample_start, packet in zip((0, BLOCK_SIZE), received[1:]):
            assert FRAME_HEADER.unpack_from(packet) == (sample_start, BLOCK_SIZE)
            assert len(packet) == 12 + BLOCK_SIZE * 2 * 4
        assert any("Recording started" in message for message in statuses)
        assert any("Recording stopped" in message for message in statuses)
        assert all("integration-key" not in message for message in statuses)

    asyncio.run(exercise())
