#!/bin/bash
# Run the built firmware on an emulated Cortex-M55, loading the plan into DDR.
#
# Usage: ./run.sh <plan.tgrs> [timeout_s]
# Requires qemu-system-arm (8.2+) with the mps3-an547 machine.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PLAN="$1"; TMO="${2:-180}"
[ -f "$HERE/firmware.elf" ] || { echo "build first: ./build.sh ..."; exit 2; }
[ -f "$PLAN" ] || { echo "plan not found: $PLAN"; exit 2; }
timeout "$TMO" qemu-system-arm -M mps3-an547 -cpu cortex-m55 -nographic \
  -semihosting-config enable=on,target=native \
  -kernel "$HERE/firmware.elf" \
  -device loader,file="$PLAN",addr=0x60000000
