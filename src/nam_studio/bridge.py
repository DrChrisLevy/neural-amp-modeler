"""Native, sample-indexed audio capture for the Neural Amp Modeler web application."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import struct
import threading
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from websockets.asyncio.client import connect

SAMPLE_RATE = 48_000
BLOCK_SIZE = 2_048
FRAME_HEADER = struct.Struct("<QI")
QUEUE_BLOCKS = 16
CAPTURE_STALL_TIMEOUT = 3.0


class BridgeError(RuntimeError):
    """A capture or connection problem that can be shown to the user."""


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    channels: int
    default_sample_rate: float


def websocket_url(origin: str) -> str:
    """Accept an origin, never a credential-bearing URL or arbitrary endpoint."""
    try:
        parsed = urlsplit(origin)
        port = parsed.port  # Also validates malformed port values.
    except ValueError as exc:
        raise BridgeError("Invalid server URL. Use an origin such as http://localhost:8000.") from exc
    if (
        parsed.scheme not in {"http", "https", "ws", "wss"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or any(char.isspace() for char in origin)
        or port == 0
    ):
        raise BridgeError(
            "Use the server origin only, with no path, query, fragment, or credentials "
            "(for example, https://your-server.example)."
        )
    scheme = "wss" if parsed.scheme in {"https", "wss"} else "ws"
    return urlunsplit((scheme, parsed.netloc, "/ws/bridge", "", ""))


def resolve_token(value: str | None, environ: Mapping[str, str] | None = None) -> str | None:
    """Read secrets without including their value in status messages."""
    env = os.environ if environ is None else environ
    if value is None:
        return env.get("NAM_STUDIO_KEY") or None
    if value.startswith("env:"):
        name = value[4:]
        if not name or not env.get(name):
            raise BridgeError("The environment variable requested by --token is not set.")
        return env[name]
    return value or None


def input_devices() -> list[InputDevice]:
    try:
        import sounddevice as sd
    except (ImportError, OSError, RuntimeError) as exc:
        raise BridgeError(
            "Could not enumerate audio devices. Check that PortAudio is available "
            "and your Quad Cortex Mini is connected over USB."
        ) from exc
    try:
        devices = sd.query_devices()
    except sd.PortAudioError as exc:
        raise BridgeError(
            "Could not enumerate audio inputs. Check the Quad Cortex USB connection "
            "and your operating system's audio settings."
        ) from exc
    return [
        InputDevice(index, str(device["name"]), int(device["max_input_channels"]),
                    float(device["default_samplerate"]))
        for index, device in enumerate(devices)
        if int(device["max_input_channels"]) > 0
    ]


def select_device(devices: Sequence[InputDevice], requested: str | int | None) -> InputDevice:
    if requested is None:
        matches = [device for device in devices if "quad cortex" in device.name.casefold()]
        if not matches:
            raise BridgeError(
                "No Quad Cortex input found. Connect it by USB, run 'neural-amp-modeler devices', "
                "then select an input explicitly with --device. No microphone was opened."
            )
    else:
        query = str(requested).strip()
        if query.isdecimal():
            matches = [device for device in devices if device.index == int(query)]
        else:
            matches = [device for device in devices if device.name.casefold() == query.casefold()]
            if not matches and query:
                matches = [device for device in devices if query.casefold() in device.name.casefold()]
        if not matches:
            raise BridgeError("That input device was not found. Run 'neural-amp-modeler devices' to list inputs.")
    if len(matches) != 1:
        raise BridgeError(
            "More than one matching input was found. Run 'neural-amp-modeler devices' "
            "and choose its numeric index with --device."
        )
    return matches[0]


def validate_channels(device: InputDevice, dry_channel: int, wet_channel: int) -> int:
    if min(dry_channel, wet_channel) < 1:
        raise BridgeError("Channel numbers start at 1.")
    if dry_channel == wet_channel:
        raise BridgeError("Dry and wet must use different input channels.")
    required = max(dry_channel, wet_channel)
    if required > device.channels:
        raise BridgeError(
            f"Selected input has {device.channels} channels, but channel {required} was requested. "
            "Check the USB input device and channel mapping."
        )
    return required


def pack_frames(sample_start: int, frames: np.ndarray) -> bytes:
    """Wire format: uint64 sample offset, uint32 frame count, interleaved f32 LE."""
    audio = np.asarray(frames, dtype="<f4", order="C")
    if audio.ndim != 2 or audio.shape[1] != 2 or audio.shape[0] == 0:
        raise ValueError("Audio must have shape (frames, 2).")
    return FRAME_HEADER.pack(sample_start, len(audio)) + audio.tobytes(order="C")


def synthetic_frames(sample_start: int, count: int = BLOCK_SIZE) -> np.ndarray:
    """A reproducible test pair, deliberately distinct from real guitar capture."""
    samples = np.arange(sample_start, sample_start + count, dtype=np.float64)

    def signal(index: np.ndarray) -> np.ndarray:
        time = index / SAMPLE_RATE
        envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 0.37 * time) ** 2
        return envelope * (
            0.12 * np.sin(2 * np.pi * 110 * time)
            + 0.05 * np.sin(2 * np.pi * 220 * time)
            + 0.03 * np.sin(2 * np.pi * 330 * time)
        )

    dry = signal(samples)
    wet = 0.3 * np.tanh(3.5 * signal(np.maximum(samples - 64, 0)))
    return np.stack((dry, wet), axis=1).astype("<f4")


class CaptureBuffer:
    """Bounded callback-thread queue which permanently fails on the first gap."""

    def __init__(self, dry_channel: int, wet_channel: int, capacity: int = QUEUE_BLOCKS):
        self.columns = [dry_channel - 1, wet_channel - 1]
        self.queue: queue.Queue[bytes] = queue.Queue(maxsize=capacity)
        self.failed = threading.Event()
        self.lock = threading.Lock()
        self.sample_start = 0
        self.dropped_frames = 0
        self.reason = "capture_queue_overflow"
        self.unknown_drop_count = False

    def callback(self, indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        del time_info
        start = self.sample_start
        self.sample_start += frames
        if self.failed.is_set():
            with self.lock:
                self.dropped_frames += frames
            return
        if status:
            with self.lock:
                self.dropped_frames += frames
                self.reason = "audio_device_overflow" if getattr(status, "input_overflow", False) else "audio_device_error"
                self.unknown_drop_count = True
            self.failed.set()
            return
        packet = pack_frames(start, indata[:, self.columns])
        try:
            self.queue.put_nowait(packet)
        except queue.Full:
            with self.lock:
                self.dropped_frames += frames
            self.failed.set()

    def gap_message(self) -> dict[str, object]:
        with self.lock:
            return {
                "type": "gap",
                "dropped_frames": self.dropped_frames,
                "reason": self.reason,
                "dropped_frames_unknown": self.unknown_drop_count,
            }


async def _send_capture(websocket: object, buffer: CaptureBuffer) -> None:
    last_frame = asyncio.get_running_loop().time()
    while True:
        if buffer.failed.is_set():
            await asyncio.wait_for(websocket.send(json.dumps(buffer.gap_message())), timeout=3)
            raise BridgeError("Audio capture dropped frames. Stream stopped; any active recording is incomplete.")
        try:
            packet = await asyncio.to_thread(buffer.queue.get, True, 0.1)
        except queue.Empty:
            if asyncio.get_running_loop().time() - last_frame > CAPTURE_STALL_TIMEOUT:
                await asyncio.wait_for(websocket.send(json.dumps({
                    "type": "gap", "dropped_frames": 0,
                    "dropped_frames_unknown": True, "reason": "audio_device_stopped",
                })), timeout=3)
                raise BridgeError(
                    "Audio input stopped delivering samples. Check its USB connection "
                    "and microphone permission; any active recording is incomplete."
                )
            continue
        if buffer.failed.is_set():
            continue
        last_frame = asyncio.get_running_loop().time()
        await asyncio.wait_for(websocket.send(packet), timeout=5)


async def _send_synthetic(websocket: object) -> None:
    sample_start = 0
    started = asyncio.get_running_loop().time()
    while True:
        await asyncio.wait_for(websocket.send(pack_frames(sample_start, synthetic_frames(sample_start))), timeout=5)
        sample_start += BLOCK_SIZE
        due = started + sample_start / SAMPLE_RATE
        await asyncio.sleep(max(0, due - asyncio.get_running_loop().time()))


async def _receive_status(websocket: object, emit: Callable[[str], None]) -> None:
    async for raw in websocket:
        if not isinstance(raw, str):
            continue
        try:
            message = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if not isinstance(message, dict):
            continue
        if message.get("type") == "record":
            if message.get("action") == "start":
                emit("Recording started in Neural Amp Modeler.")
            elif message.get("action") == "stop":
                emit("Recording stopped. Live monitoring continues.")
        elif message.get("type") == "error":
            raise BridgeError("The server reported an error and stopped the stream. Check the Neural Amp Modeler page.")
    raise BridgeError("Server disconnected. Capture stopped; restart the bridge to reconnect.")


async def _run_tasks(sender: object, receiver: object) -> None:
    tasks = [asyncio.create_task(sender), asyncio.create_task(receiver)]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def run_bridge(
    url: str,
    *,
    device: str | int | None = None,
    dry_channel: int = 1,
    wet_channel: int = 3,
    token: str | None = None,
    synthetic: bool = False,
    emit: Callable[[str], None] = print,
) -> None:
    endpoint = websocket_url(url)
    sd = None
    selected = None
    required = 0
    if synthetic:
        if device is not None:
            raise BridgeError("Use --synthetic without --device; demo mode never opens audio hardware.")
        name = "Synthetic test source"
    else:
        selected = select_device(input_devices(), device)
        required = validate_channels(selected, dry_channel, wet_channel)
        import sounddevice as sd

        try:
            sd.check_input_settings(device=selected.index, channels=required, dtype="float32", samplerate=SAMPLE_RATE)
        except sd.PortAudioError as exc:
            raise BridgeError(
                "This input cannot capture the requested channels at 48 kHz. "
                "Check the Quad Cortex USB connection and audio-device settings."
            ) from exc
        name = selected.name

    async with connect(endpoint, compression=None, max_size=65_536, max_queue=8, close_timeout=3) as websocket:
        await websocket.send(json.dumps({
            "type": "hello",
            "token": token,
            "device": name,
            "name": name,
            "device_index": selected.index if selected else None,
            "source": "demo" if synthetic else "bridge",
            "sample_rate": SAMPLE_RATE,
            "channels": 2,
            "dry_channel": dry_channel,
            "wet_channel": wet_channel,
        }))
        try:
            ready = json.loads(await asyncio.wait_for(websocket.recv(), timeout=15))
        except (ValueError, TypeError, asyncio.TimeoutError) as exc:
            raise BridgeError("Server did not acknowledge the audio bridge. Check its URL and access key.") from exc
        if not isinstance(ready, dict) or ready.get("type") != "ready":
            raise BridgeError("Server refused the audio bridge. Check the access key and close any other bridge.")

        emit(f"Connected: {name} · 48 kHz · dry/wet pair. Press Ctrl+C to stop.")
        if synthetic:
            emit("DEMO audio only. No microphone or guitar input is being captured.")
            await _run_tasks(_send_synthetic(websocket), _receive_status(websocket, emit))
        else:
            emit(f"Capturing USB channels {dry_channel} (dry) and {wet_channel} (wet).")
            buffer = CaptureBuffer(dry_channel, wet_channel)
            try:
                with sd.InputStream(
                    device=selected.index,
                    samplerate=SAMPLE_RATE,
                    blocksize=BLOCK_SIZE,
                    channels=required,
                    dtype="float32",
                    callback=buffer.callback,
                ):
                    await _run_tasks(_send_capture(websocket, buffer), _receive_status(websocket, emit))
            except sd.PortAudioError as exc:
                raise BridgeError(
                    "Could not capture audio. Check that the Quad Cortex is still connected "
                    "and that your terminal has microphone permission."
                ) from exc
