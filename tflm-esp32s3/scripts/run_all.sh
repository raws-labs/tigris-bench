#!/usr/bin/env bash
# Run all 5 benchmark configurations, capture serial logs.
#
# Usage: ./run_all.sh [port]
# Output: results/raw/<config>.log
set -euo pipefail

PORT="${1:-/dev/ttyUSB0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(dirname "$SCRIPT_DIR")"
RAW_DIR="$BENCH_DIR/results/raw"

mkdir -p "$RAW_DIR"

flash_plan() {
    local plan_file="$1"
    echo "  Flashing plan: $plan_file"
    # Use esptool to write plan to the "plan" partition at offset 0x210000
    python -m esptool --port "$PORT" write_flash 0x210000 "$plan_file"
}

capture_until_done() {
    local log_file="$1"
    local timeout_sec="${2:-120}"
    echo "  Capturing serial to $log_file (timeout ${timeout_sec}s)..."
    timeout "$timeout_sec" python -c "
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
" || echo "  (timed out)"
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
    python -m esptool --port "$PORT" run 2>/dev/null || true
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

MODELS="$BENCH_DIR/models/output"

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

echo ""
echo "All configs complete"
echo "View results: python scripts/results.py results/raw/ -o results/summary.json"
