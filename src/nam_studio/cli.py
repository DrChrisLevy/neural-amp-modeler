"""Command-line entry points for local serving and explicit audio capture."""

from __future__ import annotations

import argparse
import asyncio
import sys

from .bridge import BridgeError, input_devices, run_bridge


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="neural-amp-modeler", description="Capture paired guitar audio for Neural Amp Modeler.")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("devices", help="List audio inputs without starting capture.")

    source = commands.add_parser("input", help="Download and verify NAM's standard 3:10 input WAV.")
    source.add_argument("--output", default=".data/input.wav")
    capture = commands.add_parser("capture", help="Reamp a WAV through the currently edited Quad Cortex preset.")
    capture.add_argument("--input", default=".data/input.wav", help="Mono 48 kHz WAV (default: .data/input.wav).")
    capture.add_argument("--name", help="Name for this version of your edited preset.")
    capture.add_argument("--output", default=".data/pairs", help="Parent folder for separately saved pairs.")
    capture.add_argument("--wet-channel", type=int, choices=(3, 4), default=3, help="USB return: 3 left, 4 right.")
    capture.add_argument("--inspect", action="store_true", help="Read the current preset; no routing changes or audio playback.")

    repair = commands.add_parser("repair-capture", help="Save an aligned copy of a captured pair using its archived raw.wav; no device access.")
    repair.add_argument("source", help="Existing capture folder containing input.wav, raw.wav, and metadata.json.")
    repair.add_argument("--output", required=True, help="New folder for the corrected pair; must not already exist.")

    bridge = commands.add_parser("bridge", help="Stream a dry/wet pair from your Quad Cortex over USB.")
    bridge.add_argument("--url", default="http://localhost:8000", help="Neural Amp Modeler server origin (default: http://localhost:8000).")
    bridge.add_argument("--device", help="Input device name or numeric index; otherwise find a unique Quad Cortex.")
    bridge.add_argument("--dry-channel", type=int, default=1, help="Dry USB channel, numbered from 1 (default: 1).")
    bridge.add_argument("--wet-channel", type=int, default=3, help="Wet USB channel, numbered from 1 (default: 3).")
    bridge.add_argument("--synthetic", action="store_true", help="Explicit demo source for testing; opens no audio hardware.")

    serve = commands.add_parser("serve", help="Serve the local Neural Amp Modeler app.")
    serve.add_argument("--host", default="127.0.0.1", help="Listen address (default: 127.0.0.1).")
    serve.add_argument("--port", default=8000, type=int, help="Listen port (default: 8000).")
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    try:
        if args.command == "devices":
            devices = input_devices()
            if not devices:
                print("No audio inputs found. Connect the Quad Cortex Mini over USB and try again.")
                return
            print("Index  Input channels  Default rate  Device")
            for device in devices:
                print(f"{device.index:>5}  {device.channels:>14}  {device.default_sample_rate:>9.0f} Hz  {device.name}")
        elif args.command == "input":
            from .capture import download_input
            print(f"NAM v3 input (3:10, mono, 48 kHz): {download_input(args.output).resolve()}")
        elif args.command == "capture":
            try:
                from .quad_cortex import Mini
            except ModuleNotFoundError as exc:
                raise BridgeError("Install capture support with: uv sync --extra capture") from exc
            if args.inspect:
                import json
                with Mini() as mini:
                    _, snapshot = mini.snapshot()
                    print(json.dumps({k: snapshot[k] for k in ("preset_name", "unsaved_edits", "position", "scene", "firmware")}, indent=2))
            else:
                if not args.name or not args.name.strip():
                    raise BridgeError('Give this sound a name with --name "Brit crunch - no reverb".')
                from .capture import capture_pair
                capture_pair(args.input, args.name.strip(), args.output, args.wet_channel)
        elif args.command == "repair-capture":
            from .capture import repair_capture
            repair_capture(args.source, args.output)
        elif args.command == "bridge":
            asyncio.run(run_bridge(
                args.url,
                device=args.device,
                dry_channel=args.dry_channel,
                wet_channel=args.wet_channel,
                synthetic=args.synthetic,
            ))
        else:
            if not 1 <= args.port <= 65535:
                raise BridgeError("Port must be between 1 and 65535.")
            import uvicorn

            uvicorn.run("nam_studio.server:app", host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\nAudio bridge stopped." if args.command == "bridge" else "\nStopped.")
    except BridgeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except Exception as exc:
        if args.command in {"capture", "input", "repair-capture"}:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1) from None
        # Do not display server replies, connection headers, or potential secrets.
        print(
            f"Error: {type(exc).__name__}. Check the server connection "
            "and audio device; the stream has stopped.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


def cli() -> None:
    """Compatibility entry point used by the package script."""
    main()


if __name__ == "__main__":
    main()
