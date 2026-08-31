#!/usr/bin/env bash
# Run all 10 benchmark configurations, capture, aggregate, and validate.
#
# Usage: ./run_all.sh [port]                       # local USB
#        BENCH_TRANSPORT=siliconrig ./run_all.sh   # remote ESP32-S3
# Output: results/raw/<config>.log
set -euo pipefail

PORT="${1:-/dev/ttyUSB0}"
PYTHON="${PYTHON:-python3}"
ESPTOOL="${ESPTOOL:-esptool.py}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(dirname "$SCRIPT_DIR")"
RAW_DIR="${BENCH_RAW_DIR:-$BENCH_DIR/results/raw}"
MODELS="${BENCH_MODELS_DIR:-$BENCH_DIR/../../models/output}"
SUMMARY="${BENCH_SUMMARY:-$BENCH_DIR/results/summary.json}"
PLAN_DIR="${TIGRIS_PLAN_DIR:-$BENCH_DIR/build/plans}"
TRANSPORT="${BENCH_TRANSPORT:-local}"
MERGED_DIR="${BENCH_MERGED_DIR:-$BENCH_DIR/build/siliconrig}"
SRIG_MANIFEST=""

mkdir -p "$RAW_DIR" "$(dirname "$SUMMARY")"

flash_plan() {
    local plan_file="$1"
    echo "  Flashing plan: $plan_file"
    # Use esptool to write plan to the "plan" partition at offset 0x210000
    "$ESPTOOL" --port "$PORT" write_flash 0x210000 "$plan_file"
}

capture_until_done() {
    local log_file="$1"
    local timeout_sec="${2:-120}"
    echo "  Capturing serial to $log_file (timeout ${timeout_sec}s)..."
    if ! timeout "$timeout_sec" "$PYTHON" -c "
import serial, sys
ser = serial.Serial('$PORT', 115200, timeout=1)
with open('$log_file', 'w') as f:
    while True:
        line = ser.readline().decode('utf-8', errors='replace')
        if line:
            sys.stdout.write(line)
            f.write(line)
            if 'BENCH_DONE' in line:
                break
ser.close()
"; then
        echo "  ERROR: capture failed or timed out: $log_file" >&2
        return 1
    fi
    if ! grep -q 'BENCH_DONE' "$log_file"; then
        echo "  ERROR: capture ended without BENCH_DONE: $log_file" >&2
        return 1
    fi
}

queue_siliconrig_firmware() {
    local name="$1"
    local project_dir="$2"
    local app_name="$3"
    local plan_file="${4:-}"
    local capture_timeout="${5:-180}"
    local firmware="$MERGED_DIR/${name}.bin"
    local segments=(
        0x0 "$project_dir/build/bootloader/bootloader.bin"
        0x8000 "$project_dir/build/partition_table/partition-table.bin"
        0x10000 "$project_dir/build/${app_name}.bin"
    )

    if [ -n "$plan_file" ]; then
        # The plan partition begins at 0x210000. 0x60000 is inside the factory
        # app partition and produces a flashable image whose plan is invisible
        # to the runtime.
        segments+=(0x210000 "$plan_file")
    fi

    mkdir -p "$MERGED_DIR"
    "$ESPTOOL" --chip esp32s3 merge_bin --format raw \
        -o "$firmware" --flash_mode dio --flash_freq 80m --flash_size 16MB \
        "${segments[@]}"
    printf '%s\t%s\t%s\n' "$firmware" "$name" "$capture_timeout" >> "$SRIG_MANIFEST"
    echo "  Queued SiliconRig image: $firmware"
}

capture_siliconrig_matrix() {
    echo ""
    echo "Flashing + capturing all configurations on SiliconRig..."
    "$PYTHON" -u - "$SRIG_MANIFEST" "$RAW_DIR" <<'PY'
import pathlib
import sys

from siliconrig import Client


manifest = pathlib.Path(sys.argv[1])
raw_dir = pathlib.Path(sys.argv[2])
items = [line.rstrip("\n").split("\t") for line in manifest.read_text().splitlines()]
client = Client()
try:
    with client.session(board="esp32-s3") as session:
        info = session.info()
        print(f"-- esp32-s3 board={info.get('board_id', 'unknown')} ({len(items)} cells) --")
        for firmware, name, timeout in items:
            print(f"  {name}: flashing {pathlib.Path(firmware).stat().st_size} bytes")
            session.flash(firmware, timeout=300)
            log = session.serial.read_until("BENCH_DONE", timeout=float(timeout))
            if "BENCH_DONE" not in log:
                raise RuntimeError(f"{name}: capture ended without BENCH_DONE")
            if not log.endswith("\n"):
                log += "\n"
            (raw_dir / f"{name}.log").write_text(log)
            result = next(
                (line for line in log.splitlines() if "BENCH_RESULT:" in line),
                "missing BENCH_RESULT",
            )
            print(f"  {name}: {result[:160]}")
finally:
    client.close()
PY
}

run_tigris_config() {
    local name="$1"
    local plan_file="$2"
    local kernel="$3"  # f32, s8, esp_nn
    local capture_timeout="${4:-180}"  # slow models (U-Net prints a 512K output map) need more
    local log_file="$RAW_DIR/${name}.log"

    echo ""
    echo "Config: $name"

    # Build with the right kernel
    cd "$BENCH_DIR/tigris-esp"
    idf.py fullclean 2>/dev/null || true
    idf.py set-target esp32s3
    if [ "$kernel" = "f32" ]; then
        idf.py build
    else
        idf.py build -DBENCH_KERNEL="$kernel"
    fi
    if [ "$TRANSPORT" = "siliconrig" ]; then
        queue_siliconrig_firmware \
            "$name" "$BENCH_DIR/tigris-esp" "tigris_bench" "$plan_file" "$capture_timeout"
    else
        idf.py -p "$PORT" flash
        flash_plan "$plan_file"
        "$ESPTOOL" --port "$PORT" run 2>/dev/null || true
        sleep 1
        capture_until_done "$log_file" "$capture_timeout"
    fi

    echo "  Log: $log_file"
}

run_tflm_config() {
    local name="$1"
    local int8_flag="$2"  # "" or "-DBENCH_INT8=1"
    local log_file="$RAW_DIR/${name}.log"

    echo ""
    echo "Config: $name"

    cd "$BENCH_DIR/tflm-esp"
    idf.py fullclean 2>/dev/null || true
    idf.py set-target esp32s3
    if [ -n "$int8_flag" ]; then
        idf.py build "$int8_flag"
    else
        idf.py build
    fi
    if [ "$TRANSPORT" = "siliconrig" ]; then
        queue_siliconrig_firmware \
            "$name" "$BENCH_DIR/tflm-esp" "tflm_bench"
    else
        idf.py -p "$PORT" flash
        sleep 1
        capture_until_done "$log_file" 120
    fi

    echo "  Log: $log_file"
}

collect_and_validate() {
    echo ""
    echo "Collecting the complete matrix..."
    SUMMARY_CANDIDATE="$(mktemp "$(dirname "$SUMMARY")/.summary.XXXXXX.json")"
    trap 'rm -f "$SUMMARY_CANDIDATE"' EXIT
    "$PYTHON" "$SCRIPT_DIR/results.py" "$RAW_DIR" \
        -o "$SUMMARY_CANDIDATE"

    echo ""
    echo "Validating device outputs..."
    "$PYTHON" "$SCRIPT_DIR/validate_accuracy.py" \
        "$SUMMARY_CANDIDATE" "$MODELS"

    # Slim any megabyte-scale dense output map (e.g. the U-Net segmentation map)
    # down to a canary slice plus a length+sha256 digest, AFTER the accuracy gate
    # has already checked the full vector against the model reference. Classifier
    # cells are untouched, so a U-Net-free summary is unchanged.
    "$PYTHON" "$SCRIPT_DIR/slim_summary.py" "$SUMMARY_CANDIDATE"

    # Only replace the publishable summary after both gates pass. A failed or
    # numerically incorrect capture can never overwrite the previous good result.
    mv "$SUMMARY_CANDIDATE" "$SUMMARY"
    trap - EXIT

    echo ""
    echo "Result and accuracy gates passed."
}

# Accuracy references are checked before touching hardware so an incomplete
# model-preparation step cannot waste a full benchmark run.
for ref in \
    ds_cnn_reference_f32.bin \
    ds_cnn_matched_ref.bin \
    ds_cnn_tflite_reference_f32.bin \
    ds_cnn_tflite_reference_i8.bin \
    mobilenet_v1_matched_ref.bin \
    unet_matched_ref.bin; do
    if [ ! -f "$MODELS/$ref" ]; then
        echo "ERROR: missing $MODELS/$ref; run: python models/prepare.py" >&2
        exit 1
    fi
done

# Host-only entry point used by tests and to revalidate existing captures.
if [ "${BENCH_VALIDATE_ONLY:-0}" = 1 ]; then
    collect_and_validate
    exit 0
fi

case "$TRANSPORT" in
    local) ;;
    siliconrig)
        : "${SRIG_API_KEY:?set SRIG_API_KEY for BENCH_TRANSPORT=siliconrig}"
        "$PYTHON" -c 'import numpy, rich, siliconrig'
        command -v "$ESPTOOL" >/dev/null
        SRIG_MANIFEST="$(mktemp)"
        trap 'rm -f "$SRIG_MANIFEST"' EXIT
        ;;
    *)
        echo "ERROR: BENCH_TRANSPORT must be local or siliconrig" >&2
        exit 2
        ;;
esac

TIGRIS_COMPILER_ROOT="${TIGRIS_COMPILER_ROOT:-$(cd "$BENCH_DIR/../../../tigris" && pwd)}"
TIGRIS_RUNTIME_ROOT="${TIGRIS_RUNTIME_ROOT:-$(cd "$BENCH_DIR/../../../tigris-runtime" && pwd)}"
TIGRIS_COMPILER="${TIGRIS_COMPILER:-$TIGRIS_COMPILER_ROOT/.venv/bin/tigris}"
CORE_CHECK_ARGS=(
    --compiler-root "$TIGRIS_COMPILER_ROOT"
    --runtime-root "$TIGRIS_RUNTIME_ROOT"
)
if [ "${TIGRIS_ALLOW_UNPINNED_CORE:-0}" = 1 ]; then
    CORE_CHECK_ARGS+=(--allow-unpinned)
fi
"$PYTHON" "$BENCH_DIR/../../common/check_core_versions.py" "${CORE_CHECK_ARGS[@]}"

echo "Compiling current TiGrIS plans..."
"$PYTHON" "$SCRIPT_DIR/prepare_tigris_plans.py" \
    --compiler "$TIGRIS_COMPILER" --models-dir "$MODELS" \
    --output-dir "$PLAN_DIR"

echo "TiGrIS vs TFLM Benchmark Suite"
echo "Transport: $TRANSPORT"
if [ "$TRANSPORT" = "local" ]; then
    echo "Port: $PORT"
fi
echo "Output: $RAW_DIR/"

# Config 1: TiGrIS f32 (ref kernel)
run_tigris_config "tigris_f32_ref" "$PLAN_DIR/ds_cnn.tgrs" "f32"

# Config 2: TiGrIS i8 (ref kernel)
run_tigris_config "tigris_i8_ref" "$PLAN_DIR/ds_cnn_i8.tgrs" "s8"

# Config 3: TiGrIS i8 (ESP-NN kernel)
run_tigris_config "tigris_i8_espnn" "$PLAN_DIR/ds_cnn_i8.tgrs" "esp_nn"

# Config 4: TFLM f32
run_tflm_config "tflm_f32" ""

# Config 5: TFLM i8
run_tflm_config "tflm_i8" "-DBENCH_INT8=1"

# Case A: "Doesn't Fit" (MobileNetV1)

# Config 6: TiGrIS MobileNetV1 i8 (ESP-NN)
run_tigris_config "tigris_mbv1_i8_espnn" "$PLAN_DIR/mobilenet_v1_i8.tgrs" "esp_nn"

# Config 7: TFLM MobileNetV1 i8 (expected: ARENA_TOO_SMALL)
run_tflm_config "tflm_mbv1_i8" "-DBENCH_WIDE=1"

# Case B: Tiling overhead sweep (MobileNetV1, varied budgets)

for budget in 128k 64k 32k; do
    run_tigris_config "tigris_mbv1_i8_espnn_${budget}" "$PLAN_DIR/mobilenet_v1_i8_${budget}.tgrs" "esp_nn"
done

# Case C: "Doesn't Fit" segmentation showcase (U-Net, 2D-tiled ConvTranspose
# decoder). ESP-NN has no TransposeConv/Concat kernels, so those ops fall
# back to s8_ref; the surrounding Conv stages still dispatch to ESP-NN.

# TiGrIS U-Net i8 (ESP-NN, falls back to s8_ref per-op)
run_tigris_config "tigris_unet_i8_espnn" "$PLAN_DIR/unet.tgrs" "esp_nn" 1200

# TFLM U-Net i8 (expected: ARENA_TOO_SMALL; peak activation 1.19 MiB >>
# the 256 KB internal-SRAM arena)
run_tflm_config "tflm_unet_i8" "-DBENCH_UNET=1"

if [ "$TRANSPORT" = "siliconrig" ]; then
    capture_siliconrig_matrix
    rm -f "$SRIG_MANIFEST"
    SRIG_MANIFEST=""
    trap - EXIT
fi

collect_and_validate
echo "All configs complete; result and accuracy gates passed."
