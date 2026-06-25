#!/usr/bin/env bash
# Flash a firmware .bin to a NUCLEO board via ST-LINK USB mass-storage drag-drop.
#
# The ST-LINK exposes the board as a USB drive; copying a raw .bin programs it
# to 0x08000000 and the board resets and runs. No external tool needed.
#
# Usage:
#   scripts/flash.sh build/h753_cmsis_nn/tigris_bench.bin [mount_point]
#
# Default mount point is the NUCLEO-H753ZI drive.
set -euo pipefail

BIN="${1:?usage: flash.sh <firmware.bin> [mount_point]}"
MOUNT="${2:-/media/armin/NOD_H753ZI}"

[ -f "$BIN" ] || { echo "error: no such file: $BIN" >&2; exit 1; }
[ -d "$MOUNT" ] || {
    echo "error: board drive not mounted at $MOUNT" >&2
    echo "       plug in the board / check 'ls /media/$USER'." >&2
    exit 1
}

echo "Flashing $(basename "$BIN") ($(stat -c%s "$BIN") bytes) -> $MOUNT"
cp "$BIN" "$MOUNT/"
sync
echo "Copied. ST-LINK is programming; the board will reset and run."
echo "Capture results with: scripts/capture_serial.py -o <log>"
