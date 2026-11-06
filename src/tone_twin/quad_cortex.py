"""Current-preset USB routing, adapted from the decision-model capture project.

No preset recall, scene change, save, or undo commands are permitted. The live
edit buffer is the source of truth, including a preset's unsaved edits.
"""

from contextlib import contextmanager
import hashlib
import json
import subprocess
import sys
import time

from google.protobuf.json_format import MessageToDict
from qc_mcp import protocol as P
from qc_mcp.transport import QuadCortex

UPSTREAM_COMMIT = "f634ecd3dcf426090b90bc4cc176e4b500494b9e"


def plain(message):
    value = MessageToDict(message, preserving_proto_field_name=True)
    for key in ("request_id", "action"):
        value.pop(key, None)
    return value


def fingerprint(preset):
    state = type(preset)()
    state.CopyFrom(preset)
    # Routing edits can update firmware bookkeeping without changing the tone.
    state.ClearField("modified_version")
    state.ClearField("oldest_compatible_version")
    return hashlib.sha256(state.SerializeToString(deterministic=True)).hexdigest()


def input_rows(preset):
    ports = [c.in_portid for c in preset.chains]
    outputs = [c.out_portid for c in preset.chains]
    if (any(p not in (0, 1, 7, 8) for p in ports)
            or not any(p in (1, 8) for p in ports)
            or any(p not in (0, 1, 16, 17, 18, 19) for p in outputs)
            or not any(p in (1, 19) for p in outputs)):
        raise ValueError("Use an Input 1 (or USB 5) preset routed to the main outputs.")
    if any(m.hash in (27000, 27002) for c in preset.chains for m in c.models):
        raise ValueError("Choose a preset without looper blocks for this capture command.")
    return [i for i, p in enumerate(ports) if p == 1]


class Transport(QuadCortex):
    """Reads and temporary Input 1 ↔ USB 5 routing are the only tone operations."""

    def request(self, command, proto_message=None, proto_bytes=None, expect=None, timeout_ms=2000):
        # Match reply IDs; upstream can otherwise consume an old telemetry reply.
        with self._io_lock:
            target = expect if expect is not None else command
            want = target if isinstance(target, int) else P.NAME_TO_CMD[target]
            self._collect(.1)
            self._pending.clear()
            self.send(command, proto_message, proto_bytes)
            request_id = getattr(proto_message, "request_id", 0)
            deadline = time.monotonic() + timeout_ms / 1000
            while time.monotonic() < deadline:
                self._collect(.1)
                pending, self._pending = self._pending, []
                for cmd, obj, raw, pb in pending:
                    if cmd == want and obj is not None and len(pb) > 2:
                        reply_id = getattr(obj, "request_id", 0)
                        if reply_id == request_id or (command == "Version" and reply_id == 0):
                            return cmd, obj, raw
            raise RuntimeError(f"No matching response to {command} within {timeout_ms} ms")

    def send(self, command, proto_message=None, proto_bytes=None):
        if proto_bytes is not None:
            raise RuntimeError("Raw device writes are disabled")
        name = command if isinstance(command, str) else P.COMMANDS[command]
        m = proto_message
        if name == "Grid":
            if (m.action != P.ACTION["UPDATE"]
                    or {f.name for f, _ in m.ListFields()} != {"action", "request_id", "preset"}
                    or {f.name for f, _ in m.preset.ListFields()} != {"chains"}
                    or len(m.preset.chains) != 1):
                raise RuntimeError("Only input routing edits are permitted")
            c = m.preset.chains[0]
            if ({f.name for f, _ in c.ListFields()} != {"row", "in_portid"}
                    or c.row not in range(4) or c.in_portid not in (1, 8)):
                raise RuntimeError("Unsupported routing edit")
        elif name not in {"KeepAlive", "ResetCommsBuffers", "Connection", "Version"}:
            readable = {"ModelRepo", "ModuleStats", "UndoRedo", "IOSettings", "GeneralSettings",
                        "Mode", "GlobalEQ", "MasterVolume", "GlobalTempo", "Scene", "PresetDirty",
                        "SetlistPosition", "RecallPreset"}
            if name not in readable or m.action != P.ACTION["READ"]:
                raise RuntimeError(f"Device write is disabled: {name}")
        return super().send(command, m)


class Mini:
    def __enter__(self):
        if sys.platform != "darwin":
            raise RuntimeError("Automatic routing is currently supported on macOS.")
        processes = subprocess.check_output(["ps", "-axo", "comm="], text=True)
        if any(p.strip().endswith("/Cortex Control") for p in processes.splitlines()):
            raise RuntimeError("Quit Cortex Control, keeping the edited preset on the Mini, then retry.")
        self.qc = Transport(session_id="tonetwincapture00000000000000000")
        set_report = self.qc.io.set_report
        self.qc.io.set_report = lambda *a, **kw: set_report(*a, **kw) & 0xffffffff
        try:
            self.qc.open()
            if self.qc.device_type != "ATMA" or self.qc.firmware != "4.1.1":
                raise RuntimeError("Automatic routing is validated for Quad Cortex Mini CorOS 4.1.1 only.")
            if self.qc.write_errors:
                raise RuntimeError("USB control transport reported write errors")
            return self
        except BaseException:
            self.qc.close()
            raise

    def __exit__(self, *args):
        self.qc.close()

    def identity(self):
        return {"position": plain(self.qc.read_state("SetlistPosition")),
                "scene": self.qc.read_state("Scene").selected_scene}

    def preset(self, ports=None):
        for _ in range(8):
            p = self.qc.get_current_preset()
            if p is not None and (ports is None or all(p.chains[r].in_portid == port for r, port in ports.items())):
                return p
            time.sleep(.25)
        raise RuntimeError("Could not read the current preset/input routing")

    def controls(self):
        result = {name: plain(self.qc.read_state(name))
                  for name in ("MasterVolume", "IOSettings", "GlobalEQ", "GlobalTempo")}
        usb = result["IOSettings"].get("settings", {}).get("usb_port", {})
        if usb.get("dry_wet") != 0.0:
            raise RuntimeError("Set the Mini USB assignment to 1/2 dry, 3/4 wet before capturing.")
        hp = result["IOSettings"].get("settings", {}).get("hp_port", {})
        if "hp_feed" in hp:
            hp["hp_feed"] = [json.loads(s) for s in sorted({json.dumps(v, sort_keys=True) for v in hp["hp_feed"]})]
        return result

    def snapshot(self):
        p = self.preset()
        state = {**self.identity(), "preset_name": p.name,
                 "unsaved_edits": bool(self.qc.read_state("PresetDirty").is_dirty),
                 "preset": plain(p), "fingerprint": fingerprint(p), "controls": self.controls(),
                 "firmware": self.qc.firmware, "qc_mcp_commit": UPSTREAM_COMMIT}
        return p, state

    @contextmanager
    def usb_input(self, baseline, state):
        rows = input_rows(baseline)
        identity = {key: state[key] for key in ("position", "scene")}
        attempted = []
        state["routing_restored"] = False
        try:
            if self.identity() != identity:
                raise RuntimeError("Preset or scene changed before capture")
            for row in rows:
                attempted.append(row)
                self.qc.set_routing(row, in_portid=8)
            routed = self.preset({row: 8 for row in rows})
            check = type(routed)()
            check.CopyFrom(routed)
            for row in rows:
                check.chains[row].in_portid = 1
            if fingerprint(check) != state["fingerprint"]:
                raise RuntimeError("Temporary routing changed other preset settings")
            yield
            if (self.identity() != identity or fingerprint(self.preset()) != fingerprint(routed)
                    or self.controls() != state["controls"] or self.qc.write_errors):
                raise RuntimeError("Preset, scene, or global controls changed during capture")
        finally:
            # Never recall a preset: doing so would discard the user's live edits.
            if attempted:
                if self.identity() != identity:
                    raise RuntimeError("Preset changed; inspect its input routing manually. No preset was recalled.")
                for row in attempted:
                    self.qc.set_routing(row, in_portid=1)
                restored = self.preset({row: 1 for row in attempted})
                if fingerprint(restored) != state["fingerprint"]:
                    raise RuntimeError("Input routing restored, but other preset settings changed during capture")
            state["routing_restored"] = True

