#!/usr/bin/env bash
# Run all 10 benchmark configurations, capture, aggregate, and validate.
#
# Usage: ./run_all.sh [port]
# Output: results/raw/<config>.log
set -euo pipefail

PORT="${1:-/dev/ttyUSB0}"
PYTHON="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(dirname "$SCRIPT_DIR")"
RAW_DIR="${BENCH_RAW_DIR:-$BENCH_DIR/results/raw}"
MODELS="${BENCH_MODELS_DIR:-$BENCH_DIR/models/output}"
SUMMARY="${BENCH_SUMMARY:-$BENCH_DIR/results/summary.json}"

mkdir -p "$RAW_DIR" "$(dirname "$SUMMARY")"

flash_plan() {
    local plan_file="$1"
    echo "  Flashing plan: $plan_file"
    # Use esptool to write plan to the "plan" partition at offset 0x210000
    "$PYTHON" -m esptool --port "$PORT" write_flash 0x210000 "$plan_file"
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

run_tigris_config() {
    local name="$1"
    local plan_file="$2"
    local kernel="$3"  # f32, s8, esp_nn
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
    idf.py -p "$PORT" flash

    # Flash the plan
    flash_plan "$plan_file"

    # Reset and capture
    "$PYTHON" -m esptool --port "$PORT" run 2>/dev/null || true
    sleep 1
    capture_until_done "$log_file" 120

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
    idf.py -p "$PORT" flash

    sleep 1
    capture_until_done "$log_file" 120

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
    ds_cnn_reference_i8.bin \
    ds_cnn_tflite_reference_f32.bin \
    ds_cnn_tflite_reference_i8.bin \
    mobilenet_v1_reference_i8.bin; do
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

echo "TiGrIS vs TFLM Benchmark Suite"
echo "Port: $PORT"
echo "Output: $RAW_DIR/"

# Config 1: TiGrIS f32 (ref kernel)
run_tigris_config "tigris_f32_ref" "$MODELS/ds_cnn.tgrs" "f32"

# Config 2: TiGrIS i8 (ref kernel)
run_tigris_config "tigris_i8_ref" "$MODELS/ds_cnn_i8.tgrs" "s8"

# Config 3: TiGrIS i8 (ESP-NN kernel)
run_tigris_config "tigris_i8_espnn" "$MODELS/ds_cnn_i8.tgrs" "esp_nn"

# Config 4: TFLM f32
run_tflm_config "tflm_f32" ""

# Config 5: TFLM i8
run_tflm_config "tflm_i8" "-DBENCH_INT8=1"

# Case A: "Doesn't Fit" (MobileNetV1)

# Config 6: TiGrIS MobileNetV1 i8 (ESP-NN)
run_tigris_config "tigris_mbv1_i8_espnn" "$MODELS/mobilenet_v1_i8.tgrs" "esp_nn"

# Config 7: TFLM MobileNetV1 i8 (expected: ARENA_TOO_SMALL)
run_tflm_config "tflm_mbv1_i8" "-DBENCH_WIDE=1"

# Case B: Tiling overhead sweep (MobileNetV1, varied budgets)

for budget in 128k 64k 32k; do
    run_tigris_config "tigris_mbv1_i8_espnn_${budget}" "$MODELS/mobilenet_v1_i8_${budget}.tgrs" "esp_nn"
done

collect_and_validate
echo "All configs complete; result and accuracy gates passed."
