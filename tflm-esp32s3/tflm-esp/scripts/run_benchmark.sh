#!/usr/bin/env bash
# Build and flash TFLM benchmark for a given dtype.
# Usage: ./run_benchmark.sh [f32|i8] [port]
set -euo pipefail

DTYPE="${1:-f32}"
PORT="${2:-/dev/ttyUSB0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=== TFLM Benchmark: dtype=$DTYPE ==="

CMAKE_ARGS=""
if [ "$DTYPE" = "i8" ]; then
    CMAKE_ARGS="-DBENCH_INT8=1"
fi

# Clean + build + flash
idf.py fullclean
idf.py set-target esp32s3
idf.py build $CMAKE_ARGS
idf.py -p "$PORT" flash

# Monitor and capture output until BENCH_DONE
echo "Monitoring serial output..."
timeout 120 idf.py -p "$PORT" monitor --no-reset 2>&1 | tee /dev/stderr | while IFS= read -r line; do
    if echo "$line" | grep -q "BENCH_DONE"; then
        kill $PPID 2>/dev/null || true
        break
    fi
done

echo "Done."
