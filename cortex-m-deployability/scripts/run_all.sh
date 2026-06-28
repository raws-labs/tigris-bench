#!/usr/bin/env bash
# Build every (board, model, config), flash + capture each on the SiliconRig HIL
# lab, then collect + validate. The Cortex-M analog of
# ../tflm-esp32s3/scripts/run_all.sh: there the device is on local USB; here the
# three boards (NUCLEO-H753ZI, NUCLEO-F446RE, Pico 2 / RP2350) live in the
# SiliconRig remote lab and the SDK abstracts the programmer (.bin via st-flash,
# .uf2 via picotool) and the serial console.
#
# Usage:   ./run_all.sh [board ...]                 # default: h753 f446 rp2350
#   env:   BENCH_MODELS="ds_cnn ad ts mbv2"         # subset of models
#          BENCH_CONFIGS="cmsis_nn s8_ref tflm"     # subset of configs
#          SRIG_API_KEY=...                         # required (rig auth)
# Output:  results/raw/<board>_<model>_<config>.log -> results/summary.json
#
# Needs: arm-none-eabi-gcc, cmake, the pico-sdk (RP2350), the release_with_logs
# TFLM libs (see BUILD.md), and python3 with the `siliconrig` SDK + numpy / rich.
# Plans + TFLM headers come from ../tflm-esp32s3/models/output (models/prepare_*.py).
set -euo pipefail

: "${SRIG_API_KEY:?set SRIG_API_KEY (SiliconRig auth) before running}"

HERE="$(cd "$(dirname "$0")/.." && pwd)"
TC="$(cd "$HERE/../../tigris-runtime" && pwd)/cmake/arm-none-eabi.cmake"
OUT="$(cd "$HERE/../tflm-esp32s3/models/output" && pwd)"
RAW="$HERE/results/raw"
PICO_SDK="${PICO_SDK_PATH:-$HOME/pico/pico-sdk}"
PICOTOOL="${PICOTOOL_DIR:-$HOME/pico/picotool/install/lib/cmake/picotool}"
NPROC="$(nproc)"

BOARDS=("$@"); [ "${#BOARDS[@]}" -eq 0 ] && BOARDS=(h753 f446 rp2350)
read -r -a MODELS  <<< "${BENCH_MODELS:-ds_cnn ad ts mbv2}"
read -r -a CONFIGS <<< "${BENCH_CONFIGS:-cmsis_nn s8_ref tflm}"

declare -A RIG=( [h753]=stm32-h753 [f446]=stm32-f446 [rp2350]=rp2350 )
declare -A TB=(  [h753]=nucleo_h753zi [f446]=nucleo_f446re )

mkdir -p "$RAW"
"$HERE/third_party/fetch.sh"          # vendor CMSIS-NN / CMSIS_6 / device headers if missing

MANIFEST="$(mktemp)"; trap 'rm -f "$MANIFEST"' EXIT
emit() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$MANIFEST"; }  # rigtype fw logname timeout

build_cell() {   # board model config
    local board=$1 model=$2 cfg=$3
    # Combinations that don't exist:
    [ "$board" = rp2350 ] && [ "$cfg" = tflm ] && return 0                       # no prebuilt M33 TFLM lib
    [ "$model" = mbv2 ] && [ "$cfg" = s8_ref ] && return 0                       # mbv2 reference is impractically slow
    [ "$model" = mbv2 ] && [ "$cfg" = tflm ] && [ "$board" != h753 ] && return 0 # OOM demo only needs the 512 KB board

    local bd="$HERE/build/${board}_${model}_${cfg}"
    local to=180; [ "$cfg" = s8_ref ] && to=300; [ "$model" = mbv2 ] && to=600
    local plan fast=32768 slow=8192
    if [ "$model" = mbv2 ]; then plan="$OUT/mbv2_a35_128k.tgrs"; fast=163840; slow=327680
    else plan="$OUT/${model}_matched_32k.tgrs"; fi

    if [ "$board" = rp2350 ]; then
        PICO_SDK_PATH="$PICO_SDK" cmake -S "$HERE/boards/pico2_rp2350" -B "$bd" \
            -Dpicotool_DIR="$PICOTOOL" -DBENCH_KERNEL="$cfg" -DTIGRIS_PLAN="$plan" \
            -DTIGRIS_FAST_ARENA_BYTES=$fast -DTIGRIS_SLOW_ARENA_BYTES=$slow >/dev/null
        PICO_SDK_PATH="$PICO_SDK" cmake --build "$bd" -j"$NPROC" >/dev/null
        emit "${RIG[$board]}" "$bd/tigris_pico_bench.uf2" "${board}_${model}_${cfg}" "$to"
        return 0
    fi

    local common=(-S "$HERE" -B "$bd" -DCMAKE_TOOLCHAIN_FILE="$TC" -DTIGRIS_BOARD="${TB[$board]}")
    if [ "$cfg" = tflm ]; then
        local arena=32768; [ "$model" = mbv2 ] && arena=491520   # ~480 KB: most of the H753 SRAM, still OOMs
        cmake "${common[@]}" -DBENCH_FRAMEWORK=tflm -DTFLM_MODEL="$model" -DTFLM_ARENA_BYTES=$arena >/dev/null
        cmake --build "$bd" -j"$NPROC" >/dev/null
        emit "${RIG[$board]}" "$bd/tflm_bench.bin" "${board}_${model}_${cfg}" "$to"
    elif cmake "${common[@]}" -DBENCH_KERNEL="$cfg" -DTIGRIS_PLAN="$plan" \
              -DTIGRIS_FAST_ARENA_BYTES=$fast -DTIGRIS_SLOW_ARENA_BYTES=$slow >/dev/null 2>&1 \
         && cmake --build "$bd" -j"$NPROC" >/dev/null 2>&1; then
        emit "${RIG[$board]}" "$bd/tigris_bench.bin" "${board}_${model}_${cfg}" "$to"
    else
        # mbv2 on F446: 591 KB plan > 512 KB flash and 301 KB working set > 128 KB
        # SRAM. The link overflow IS the result (the flash/RAM barrier), not flashed.
        echo "  BARRIER: ${board}/${model}/${cfg} does not fit (link overflow) - expected"
    fi
}

echo "Building..."
for board in "${BOARDS[@]}"; do
    for model in "${MODELS[@]}"; do
        for cfg in "${CONFIGS[@]}"; do
            echo "  build ${board}_${model}_${cfg}"
            build_cell "$board" "$model" "$cfg"
        done
    done
done

echo "Flashing + capturing on SiliconRig..."
python3 - "$MANIFEST" "$RAW" <<'PY'
import sys, collections
from siliconrig import Client
from siliconrig.serial import SerialTimeout

manifest, raw = sys.argv[1], sys.argv[2]
cells = collections.defaultdict(list)
for line in open(manifest):
    bt, fw, name, to = line.rstrip("\n").split("\t")
    cells[bt].append((fw, name, float(to)))

c = Client()
try:
    for bt, items in cells.items():
        print(f"-- {bt} ({len(items)} cells) --")
        with c.session(board=bt) as s:          # one session per board: avoids 503 on rapid realloc
            for fw, name, to in items:
                log, status = "", "ok"
                try:
                    s.flash(fw)                  # the harness boot-quiets so this races cleanly
                    log = s.serial.read_until("BENCH_DONE", timeout=to)
                except SerialTimeout:
                    status = "TIMEOUT"
                except Exception as e:
                    status = f"ERR:{type(e).__name__}"
                with open(f"{raw}/{name}.log", "w") as f:
                    f.write(log)
                rl = next((l for l in log.splitlines() if l.startswith("BENCH_RESULT")), f"status={status}")
                print(f"  {name}: {rl[:140]}")
finally:
    c.close()
PY

echo ""
echo "Collecting + validating..."
python3 "$HERE/scripts/results.py" "$RAW" -o "$HERE/results/summary.json"
python3 "$HERE/scripts/validate_accuracy.py" "$HERE/results/summary.json"
