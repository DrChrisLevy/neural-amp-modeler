"""Exercise edited-preset capture without connecting to audio or USB hardware."""

import copy
from contextlib import contextmanager
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from tone_twin.capture import capture_pair, capture_lock, download_input, load_input
from tone_twin.cli import parser
from tone_twin.storage import SAMPLE_RATE


class Audio:
    def __init__(self, status=False):
        self.status, self.stopped = status, False

    def query_devices(self):
        return [{"name": "Quad Cortex mini", "max_input_channels": 8, "max_output_channels": 8}]

    def playrec(self, audio, **kwargs):
        assert kwargs["input_mapping"] == [3, 4]
        assert kwargs["output_mapping"] == [5]
        assert kwargs["samplerate"] == SAMPLE_RATE
        self.played = audio.copy()
        # Distinct channels ensure the chosen return is used without summing.
        return np.column_stack((audio[:, 0] * .5, audio[:, 0] * .25))

    def get_status(self):
        return self.status

    def stop(self):
        self.stopped = True


class Controller:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def snapshot(self):
        self.state = {"preset_name": "Edited amp", "unsaved_edits": True}
        return object(), self.state

    @contextmanager
    def usb_input(self, baseline, state):
        state["routing_restored"] = False
        try:
            yield
        finally:
            state["routing_restored"] = True


def source(tmp_path):
    path = tmp_path / "source.wav"
    sf.write(path, .1 * np.sin(np.arange(12 * SAMPLE_RATE) * .013), SAMPLE_RATE, subtype="PCM_24")
    return path


def test_pair_has_identical_length_original_level_and_edit_metadata(tmp_path):
    path = source(tmp_path)
    sd, controller = Audio(), Controller()
    folder = capture_pair(path, "Crunch - dry", tmp_path / "pairs", 4, controller=controller, sd=sd, emit=lambda _: None)
    assert (folder / "input.wav").read_bytes() == path.read_bytes()
    dry, rate = sf.read(path)
    wet, wet_rate = sf.read(folder / "output.wav")
    np.testing.assert_allclose(wet, dry * .25, atol=2**-23)
    assert rate == wet_rate == SAMPLE_RATE and len(dry) == len(wet)
    assert not sd.played[:SAMPLE_RATE].any() and not sd.played[-SAMPLE_RATE:].any()
    assert sd.stopped and controller.closed
    meta = json.loads((folder / "metadata.json").read_text())
    assert meta["device"]["unsaved_edits"] and meta["device"]["routing_restored"]
    assert meta["status"] == "ready" and meta["wet_channel"] == 4
    assert meta["files"]["input.wav"] == hashlib.sha256(path.read_bytes()).hexdigest()
    again = capture_pair(path, "Crunch - dry", tmp_path / "pairs", controller=Controller(), sd=Audio(), emit=lambda _: None)
    assert again != folder


def test_audio_dropout_stops_and_restores_without_publishing_pair(tmp_path):
    sd, controller = Audio(status=True), Controller()
    with pytest.raises(RuntimeError, match="Audio dropped"):
        capture_pair(source(tmp_path), "Dropped", tmp_path / "pairs", controller=controller, sd=sd, emit=lambda _: None)
    folder = next((tmp_path / "pairs").iterdir())
    assert sd.stopped and controller.closed
    meta = json.loads((folder / "metadata.json").read_text())
    assert meta["status"] == "failed" and meta["device"]["routing_restored"]
    assert not (folder / "output.wav").exists()


def test_download_never_overwrites_a_different_input(tmp_path):
    path = tmp_path / "input.wav"
    path.write_bytes(b"my audio")
    with pytest.raises(ValueError, match="different file"):
        download_input(path)
    assert path.read_bytes() == b"my audio"


def test_only_one_local_capture_can_own_the_device():
    with capture_lock():
        with pytest.raises(RuntimeError, match="Another capture"):
            with capture_lock():
                pytest.fail("Second capture acquired the same device")
    with capture_lock():
        pass


def test_input_rejects_stereo_or_wrong_rate_before_device_access(tmp_path):
    for shape, rate in (((48000, 2), 48000), ((48000,), 44100)):
        path = tmp_path / "invalid.wav"
        sf.write(path, np.zeros(shape), rate)
        with pytest.raises(ValueError, match="mono, 48 kHz"):
            load_input(path)
    assert parser().parse_args(["capture", "--inspect"]).inspect


@pytest.fixture
def protocol():
    pytest.importorskip("qc_mcp", reason="Run with uv run --extra capture pytest")
    from qc_mcp import protocol as P
    return P


@pytest.fixture
def edited_mini(protocol):
    from tone_twin.quad_cortex import Mini, fingerprint
    baseline = protocol.message_class("RecallPreset")().preset
    baseline.name = "Unsaved crunch"
    baseline.volume = -3.5
    baseline.tempo = 123
    for port in (1, 0, 7, 0):
        chain = baseline.chains.add()
        chain.in_portid, chain.out_portid = port, 19 if port else 0
    current = copy.deepcopy(baseline)
    mini, calls = Mini(), []
    def route(row, in_portid):
        calls.append((row, in_portid))
        current.chains[row].in_portid = in_portid
    mini.qc = SimpleNamespace(set_routing=route, get_current_preset=lambda: copy.deepcopy(current), write_errors=[])
    mini.identity = lambda: {"position": {"position": 3}, "scene": 2}
    mini.controls = lambda: {"gain": 1}
    state = {**mini.identity(), "controls": mini.controls(), "fingerprint": fingerprint(baseline), "unsaved_edits": True}
    return mini, baseline, current, state, calls


@pytest.mark.parametrize("failure", [None, RuntimeError("audio failed"), KeyboardInterrupt()])
def test_routing_preserves_unsaved_tone_on_success_error_and_interrupt(edited_mini, failure):
    mini, baseline, current, state, calls = edited_mini
    def run():
        with mini.usb_input(baseline, state):
            assert current.chains[0].in_portid == 8
            assert current.volume == baseline.volume and current.tempo == baseline.tempo
            if failure is not None:
                raise failure
    if failure is None:
        run()
    else:
        with pytest.raises(type(failure)):
            run()
    assert current == baseline
    assert state["routing_restored"] is True
    assert calls == [(0, 8), (0, 1)]


def test_partial_routing_failure_still_restores_attempted_row(edited_mini):
    mini, baseline, current, state, calls = edited_mini
    route = mini.qc.set_routing
    def fail(row, in_portid):
        route(row, in_portid)
        if in_portid == 8:
            raise RuntimeError("USB write failed after changing route")
    mini.qc.set_routing = fail
    with pytest.raises(RuntimeError, match="USB write"):
        with mini.usb_input(baseline, state):
            pytest.fail("Must not play after failed routing")
    assert current == baseline and state["routing_restored"]


def test_user_edits_during_capture_are_not_undone(edited_mini):
    mini, baseline, current, state, calls = edited_mini
    with pytest.raises(RuntimeError, match="other preset settings changed"):
        with mini.usb_input(baseline, state):
            current.volume = -12
    assert current.chains[0].in_portid == 1 and current.volume == -12
    assert not state["routing_restored"]


def test_never_rewires_another_selected_preset(edited_mini):
    mini, baseline, current, state, calls = edited_mini
    with pytest.raises(RuntimeError, match="Preset changed"):
        with mini.usb_input(baseline, state):
            mini.identity = lambda: {"position": {"position": 4}, "scene": 2}
    assert calls == [(0, 8)]
    assert not state["routing_restored"]


def test_device_guard_blocks_save_recall_scene_and_other_edits(protocol):
    from tone_twin.quad_cortex import Transport
    transport = object.__new__(Transport)
    for command in ("File", "RecallPreset", "Scene", "SetlistPosition", "UndoRedo"):
        m = protocol.message_class(command)(action=protocol.ACTION["UPDATE"])
        with pytest.raises(RuntimeError, match="disabled"):
            transport.send(command, m)
    g = protocol.message_class("Grid")(action=protocol.ACTION["UPDATE"], request_id=1)
    g.preset.volume = -.5
    with pytest.raises(RuntimeError, match="Only input routing"):
        transport.send("Grid", g)
