#!/usr/bin/env bash
# Build, flash, and capture every kernel config on the H753, then collect and
# validate the results. Run from anywhere; paths are resolved relative to this
# script. Requires: arm-none-eabi-gcc, cmake, python3 (pyserial, numpy, rich),
# and the board connected (drag-drop drive mounted).
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
TOOLCHAIN="$HERE/../../tigris-runtime/cmake/arm-none-eabi.cmake"
MODELS="$HERE/../tflm-esp32s3/models/output"
RAW="$HERE/results/raw"
BOARD="nucleo_h753zi"
KERNELS=("${@:-s8_ref cmsis_nn}")

# Split a possibly space-joined first arg into an array.
read -r -a KERNELS <<< "${KERNELS[*]}"

mkdir -p "$RAW"

# 1. Vendor CMSIS if needed.
"$HERE/third_party/fetch.sh"

for K in "${KERNELS[@]}"; do
    echo ""
    echo "=== Building ${BOARD} / ${K} ==="
    BUILD="$HERE/build/${BOARD}_${K}"
    cmake -S "$HERE" -B "$BUILD" \
        -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
        -DTIGRIS_BOARD="$BOARD" \
        -DBENCH_KERNEL="$K"
    cmake --build "$BUILD" -j

    echo "=== Flashing ${BOARD} / ${K} ==="
    "$HERE/scripts/flash.sh" "$BUILD/tigris_bench.bin"

    echo "=== Capturing ${BOARD} / ${K} ==="
    python3 "$HERE/scripts/capture_serial.py" -o "$RAW/${BOARD}_${K}.log"
done

echo ""
echo "=== Results ==="
python3 "$HERE/scripts/results.py" "$RAW" -o "$HERE/results/summary.json"
python3 "$HERE/scripts/validate_accuracy.py" "$HERE/results/summary.json" "$MODELS"
