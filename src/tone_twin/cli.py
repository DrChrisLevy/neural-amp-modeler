"""Command-line entry points for local serving and explicit audio capture."""

from __future__ import annotations

import argparse
import asyncio
import sys

from .bridge import BridgeError, input_devices, resolve_token, run_bridge


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="tone-twin", description="Capture paired guitar audio for Tone Twin.")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("devices", help="List audio inputs without starting capture.")

    bridge = commands.add_parser("bridge", help="Stream a dry/wet pair from your Quad Cortex over USB.")
    bridge.add_argument("--url", default="http://localhost:8000", help="Tone Twin server origin (default: http://localhost:8000).")
    bridge.add_argument("--device", help="Input device name or numeric index; otherwise find a unique Quad Cortex.")
    bridge.add_argument("--dry-channel", type=int, default=1, help="Dry USB channel, numbered from 1 (default: 1).")
    bridge.add_argument("--wet-channel", type=int, default=3, help="Wet USB channel, numbered from 1 (default: 3).")
    bridge.add_argument("--token", help="Access key or env:VARIABLE. Prefer the environment; omitted reads TONE_TWIN_KEY.")
    bridge.add_argument("--synthetic", action="store_true", help="Explicit demo source for testing; opens no audio hardware.")

    serve = commands.add_parser("serve", help="Serve the local Tone Twin app.")
    serve.add_argument("--host", default="127.0.0.1", help="Listen address (default: 127.0.0.1).")
    serve.add_argument("--port", default=8000, type=int, help="Listen port (default: 8000).")
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    token = None
    try:
        if args.command == "devices":
            devices = input_devices()
            if not devices:
                print("No audio inputs found. Connect the Quad Cortex Mini over USB and try again.")
                return
            print("Index  Input channels  Default rate  Device")
            for device in devices:
                print(f"{device.index:>5}  {device.channels:>14}  {device.default_sample_rate:>9.0f} Hz  {device.name}")
        elif args.command == "bridge":
            token = resolve_token(args.token)
            asyncio.run(run_bridge(
                args.url,
                device=args.device,
                dry_channel=args.dry_channel,
                wet_channel=args.wet_channel,
                token=token,
                synthetic=args.synthetic,
            ))
        else:
            if not 1 <= args.port <= 65535:
                raise BridgeError("Port must be between 1 and 65535.")
            import uvicorn

            uvicorn.run("tone_twin.server:app", host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\nAudio bridge stopped." if args.command == "bridge" else "\nStopped.")
    except BridgeError as exc:
        message = str(exc)
        if token:
            message = message.replace(token, "[redacted]")
        print(f"Error: {message}", file=sys.stderr)
        raise SystemExit(1) from None
    except Exception as exc:
        # Do not display server replies, connection headers, or potential secrets.
        print(
            f"Error: {type(exc).__name__}. Check the server connection, access key, "
            "and audio device; the stream has stopped.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


def cli() -> None:
    """Compatibility entry point used by the package script."""
    main()


if __name__ == "__main__":
    main()
