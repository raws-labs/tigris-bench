#!/usr/bin/env python3
"""Capture a benchmark run from the board's virtual COM port.

The NUCLEO ST-LINK VCP (/dev/ttyACM0) stays enumerated across target resets, so
the intended flow is: start this in the background, then flash (cp .bin to the
drive). The H7 resets, runs once, and prints; this reads until the BENCH_DONE
sentinel. It reconnects if the port briefly drops (e.g. mass-storage remount).

Usage:
    python capture_serial.py -o results/raw/h753_cmsis_nn.log [--timeout 180]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import serial  # pyserial


def main() -> None:
    ap = argparse.ArgumentParser(description="Capture board serial output until BENCH_DONE")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--sentinel", default="BENCH_DONE")
    ap.add_argument("--timeout", type=float, default=180.0, help="overall capture timeout (s)")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.timeout
    lines: list[str] = []
    done = False

    while not done and time.monotonic() < deadline:
        try:
            ser = serial.Serial(args.port, args.baud, timeout=1)
        except (serial.SerialException, OSError):
            time.sleep(0.3)
            continue
        try:
            with ser:
                while time.monotonic() < deadline:
                    raw = ser.readline()
                    if not raw:
                        continue
                    line = raw.replace(b"\r\n", b"\n").decode("utf-8", "replace").rstrip("\n")
                    print(line, flush=True)
                    lines.append(line)
                    if args.sentinel in line:
                        done = True
                        break
        except (serial.SerialException, OSError):
            time.sleep(0.3)  # port dropped (remount); reconnect

    args.output.write_text("\n".join(lines) + "\n")
    print(f"\n[capture] wrote {len(lines)} lines to {args.output}"
          f" ({'BENCH_DONE seen' if done else 'TIMEOUT'})", file=sys.stderr)
    sys.exit(0 if done else 1)


if __name__ == "__main__":
    main()
