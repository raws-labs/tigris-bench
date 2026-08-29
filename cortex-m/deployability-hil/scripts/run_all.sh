#!/usr/bin/env bash
# Build every (board, model, config), flash + capture each on the SiliconRig HIL
# lab, then collect + validate. The Cortex-M analog of
# ../esp32s3/latency-hil/scripts/run_all.sh: there the device is on local USB; here the
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
# TFLM headers come from ../esp32s3/latency-hil/models/output. TiGrIS plans are
# compiled fresh from its matched ONNX sources into build/plans for each run.
set -euo pipefail

: "${SRIG_API_KEY:?set SRIG_API_KEY (SiliconRig auth) before running}"

HERE="$(cd "$(dirname "$0")/.." && pwd)"
TIGRIS_COMPILER_ROOT="${TIGRIS_COMPILER_ROOT:-$(cd "$HERE/../../../tigris" && pwd)}"
TIGRIS_RUNTIME_ROOT="${TIGRIS_RUNTIME_ROOT:-$(cd "$HERE/../../../tigris-runtime" && pwd)}"
TC="$TIGRIS_RUNTIME_ROOT/cmake/arm-none-eabi.cmake"
MODELS_DIR="$(cd "$HERE/../../models/output" && pwd)"
PLAN_DIR="${TIGRIS_PLAN_DIR:-$HERE/build/plans}"
TIGRIS_COMPILER="${TIGRIS_COMPILER:-$TIGRIS_COMPILER_ROOT/.venv/bin/tigris}"
RAW="$HERE/results/raw"
PICO_SDK="${PICO_SDK_PATH:-$HOME/pico/pico-sdk}"
PICOTOOL="${PICOTOOL_DIR:-$HOME/pico/picotool/install/lib/cmake/picotool}"
PICO_SDK_COMMIT="a1438dff1d38bd9c65dbd693f0e5db4b9ae91779"
NPROC="$(nproc)"

CORE_CHECK_ARGS=(
    --compiler-root "$TIGRIS_COMPILER_ROOT"
    --runtime-root "$TIGRIS_RUNTIME_ROOT"
)
if [ "${TIGRIS_ALLOW_UNPINNED_CORE:-0}" = 1 ]; then
    CORE_CHECK_ARGS+=(--allow-unpinned)
fi
python3 "$HERE/../../common/check_core_versions.py" "${CORE_CHECK_ARGS[@]}"

CANONICAL_RUN=0
if [ "$#" -eq 0 ] \
        && [ -z "${BENCH_MODELS+x}" ] \
        && [ -z "${BENCH_CONFIGS+x}" ]; then
    CANONICAL_RUN=1
fi
BOARDS=("$@"); [ "${#BOARDS[@]}" -eq 0 ] && BOARDS=(h753 f446 rp2350)
read -r -a MODELS  <<< "${BENCH_MODELS:-ds_cnn ad ts mbv2}"
read -r -a CONFIGS <<< "${BENCH_CONFIGS:-cmsis_nn s8_ref tflm}"

declare -A RIG=( [h753]=stm32-h753 [f446]=stm32-f446 [rp2350]=rp2350 )
declare -A TB=(  [h753]=nucleo_h753zi [f446]=nucleo_f446re )

if [[ " ${BOARDS[*]} " == *" rp2350 "* ]]; then
    actual_pico_sdk="$(git -C "$PICO_SDK" rev-parse HEAD)"
    if [ "$actual_pico_sdk" != "$PICO_SDK_COMMIT" ]; then
        echo "[error] pico-sdk HEAD $actual_pico_sdk != pinned $PICO_SDK_COMMIT" >&2
        exit 1
    fi
fi

mkdir -p "$RAW"
"$HERE/third_party/fetch.sh"          # vendor CMSIS-NN / CMSIS_6 / device headers if missing

echo "Compiling current TiGrIS plans..."
python3 "$HERE/scripts/prepare_tigris_plans.py" \
    --compiler "$TIGRIS_COMPILER" --models-dir "$MODELS_DIR" \
    --output-dir "$PLAN_DIR" "${MODELS[@]}"

MANIFEST="$(mktemp)"
RUN_RAW="$(mktemp -d)"
COLLECTED_SUMMARY="$(mktemp)"
trap 'rm -f "$MANIFEST" "$COLLECTED_SUMMARY"; rm -rf "$RUN_RAW"' EXIT
# rigtype, firmware, log name, timeout, embedded model, exact CMake invocation
emit() { printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6" >> "$MANIFEST"; }
command_string() { printf -v REPLY '%q ' "$@"; REPLY="${REPLY% }"; }

build_cell() {   # board model config
    local board=$1 model=$2 cfg=$3
    # Combinations that don't exist:
    [ "$board" = rp2350 ] && [ "$cfg" = tflm ] && return 0                       # no prebuilt M33 TFLM lib
    [ "$model" = mbv2 ] && [ "$cfg" = s8_ref ] && return 0                       # mbv2 reference is impractically slow
    [ "$model" = mbv2 ] && [ "$cfg" = tflm ] && [ "$board" != h753 ] && return 0 # OOM demo only needs the 512 KB board

    local bd="$HERE/build/${board}_${model}_${cfg}"
    local to=180; [ "$cfg" = s8_ref ] && to=300; [ "$model" = mbv2 ] && to=600
    # The 32 KiB plan budget excludes backend scratch. Keep enough physical
    # backing for the budget plus the exact CMSIS-NN reservation computed by
    # the runtime; reported RAM uses measured high-water, not this capacity.
    local plan fast=65536 slow=8192
    if [ "$model" = mbv2 ]; then
        plan="$PLAN_DIR/mbv2_a35.tgrs"; fast=163840; slow=327680
    else
        plan="$PLAN_DIR/${model}_matched.tgrs"
    fi

    if [ "$board" = rp2350 ]; then
        local -a configure=(cmake -S "$HERE/boards/pico2_rp2350" -B "$bd"
            -Dpicotool_DIR="$PICOTOOL" -DBENCH_KERNEL="$cfg" -DTIGRIS_PLAN="$plan" \
            -DTIGRIS_CODEGEN="$TIGRIS_COMPILER" \
            -DTIGRIS_RUNTIME_ROOT="$TIGRIS_RUNTIME_ROOT" \
            -DTIGRIS_FAST_ARENA_BYTES=$fast -DTIGRIS_SLOW_ARENA_BYTES=$slow)
        command_string "${configure[@]}"; local configure_command="$REPLY"
        PICO_SDK_PATH="$PICO_SDK" "${configure[@]}" >/dev/null
        PICO_SDK_PATH="$PICO_SDK" cmake --build "$bd" -j"$NPROC" >/dev/null
        emit "${RIG[$board]}" "$bd/tigris_pico_bench.uf2" \
             "${board}_${model}_${cfg}" "$to" "$plan" "$configure_command"
        return 0
    fi

    local common=(-S "$HERE" -B "$bd" -DCMAKE_TOOLCHAIN_FILE="$TC"
                  -DTIGRIS_RUNTIME_ROOT="$TIGRIS_RUNTIME_ROOT"
                  -DTIGRIS_BOARD="${TB[$board]}")
    if [ "$cfg" = tflm ]; then
        local arena=32768; [ "$model" = mbv2 ] && arena=491520   # ~480 KB: most of the H753 SRAM, still OOMs
        local -a configure=(cmake "${common[@]}" -DBENCH_FRAMEWORK=tflm
                            -DTFLM_MODEL="$model" -DTFLM_ARENA_BYTES=$arena)
        command_string "${configure[@]}"; local configure_command="$REPLY"
        "${configure[@]}" >/dev/null
        cmake --build "$bd" -j"$NPROC" >/dev/null
        emit "${RIG[$board]}" "$bd/tflm_bench.bin" \
             "${board}_${model}_${cfg}" "$to" \
             "$MODELS_DIR/${model}_tflite_i8.h" "$configure_command"
    else
        local -a configure=(cmake "${common[@]}" -DBENCH_KERNEL="$cfg"
                            -DTIGRIS_PLAN="$plan" -DTIGRIS_CODEGEN="$TIGRIS_COMPILER"
                            -DTIGRIS_FAST_ARENA_BYTES=$fast
                            -DTIGRIS_SLOW_ARENA_BYTES=$slow)
        command_string "${configure[@]}"; local configure_command="$REPLY"
        if "${configure[@]}" >/dev/null 2>&1 \
         && cmake --build "$bd" -j"$NPROC" >/dev/null 2>&1; then
            emit "${RIG[$board]}" "$bd/tigris_bench.bin" \
                 "${board}_${model}_${cfg}" "$to" "$plan" "$configure_command"
        else
            # mbv2 on F446: 591 KB plan > 512 KB flash and 301 KB working set > 128 KB
            # SRAM. The link overflow IS the result (the flash/RAM barrier), not flashed.
            echo "  BARRIER: ${board}/${model}/${cfg} does not fit (link overflow) - expected"
        fi
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
python3 - "$MANIFEST" "$RUN_RAW" "$HERE" "$TIGRIS_COMPILER_ROOT" \
    "$TIGRIS_RUNTIME_ROOT" "$PICO_SDK" "$HERE/../../esp32s3/latency-hil/requirements.txt" <<'PY'
import collections
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from siliconrig import Client
from siliconrig.serial import SerialTimeout

manifest, raw, bench_root, compiler_root, runtime_root, pico_sdk, requirements = map(
    Path, sys.argv[1:])


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(*args):
    return subprocess.run(
        args, check=True, capture_output=True, text=True).stdout.strip()


def git_state(path):
    if not (path / ".git").exists():
        return None
    return {
        "revision": command("git", "-C", str(path), "rev-parse", "HEAD"),
        "dirty": bool(command(
            "git", "-C", str(path), "status", "--short",
            "--untracked-files=no")),
    }


def artifact(path):
    path = Path(path)
    return {
        "name": path.name,
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def checked_python_environment(path):
    model_packages = {"onnx", "onnxruntime", "numpy", "tensorflow"}
    packages = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise RuntimeError(f"mutable Python requirement is not allowed: {line}")
        name, expected = line.split("==", 1)
        if name not in model_packages:
            continue
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(
                f"Python environment mismatch: {name}=={actual}, expected {expected}")
        packages[name] = actual
    return {"requirements_sha256": sha256(path), "packages": packages}


repositories = {
    "benchmark": git_state(bench_root.parent),
    "tigris_compiler": git_state(compiler_root),
    "tigris_runtime": git_state(runtime_root),
    "tflite_micro": git_state(bench_root / "third_party/tflite-micro"),
}
for name in ("benchmark", "tigris_compiler", "tigris_runtime"):
    if repositories[name] is None:
        raise RuntimeError(f"{name} is not a Git checkout")

dependency_dirs = {
    "CMSIS-NN": "CMSIS-NN",
    "CMSIS-Core": "CMSIS_6",
    "cmsis-device-f4": "cmsis-device-f4",
    "cmsis-device-h7": "cmsis-device-h7",
}
dependencies = {
    name: command(
        "git", "-C", str(bench_root / "third_party" / directory),
        "rev-parse", "HEAD")
    for name, directory in dependency_dirs.items()
}
tools = {
    "arm_none_eabi_gcc": command("arm-none-eabi-gcc", "--version").splitlines()[0],
    "cmake": command("cmake", "--version").splitlines()[0],
    "pico_sdk_revision": (
        git_state(pico_sdk)["revision"] if git_state(pico_sdk) else None),
}
common = {
    "repositories": repositories,
    "dependencies": dependencies,
    "tools": tools,
    "host_model_environment": checked_python_environment(requirements),
    "siliconrig_sdk_version": importlib.metadata.version("siliconrig"),
}

cells = collections.defaultdict(list)
for line in open(manifest):
    bt, fw, name, to, model, configure = line.rstrip("\n").split("\t")
    cells[bt].append((fw, name, float(to), model, configure))

c = Client()
try:
    for bt, items in cells.items():
        print(f"-- {bt} ({len(items)} cells) --")
        with c.session(board=bt) as s:          # one session per board: avoids 503 on rapid realloc
            info = s.info()
            specs = info.get("board_specs")
            if isinstance(specs, str):
                try:
                    specs = json.loads(specs)
                except json.JSONDecodeError:
                    pass
            board = {
                "siliconrig_board_id": info.get("board_id"),
                "board_type": info.get("board_type", bt),
                "specs": specs,
            }
            if not board["siliconrig_board_id"]:
                raise RuntimeError("SiliconRig did not report the allocated board ID")
            for fw, name, to, model, configure in items:
                log, status = "", "ok"
                try:
                    s.flash(fw, timeout=300)     # bridge uploads can take ~2.5 minutes
                    log = s.serial.read_until("BENCH_DONE", timeout=to)
                except SerialTimeout:
                    status = "TIMEOUT"
                except Exception as e:
                    status = f"ERR:{type(e).__name__}"
                provenance = {
                    **common,
                    "captured_at_utc": datetime.now(timezone.utc).isoformat().replace(
                        "+00:00", "Z"),
                    "build": {"configure_command": configure},
                    "artifacts": {
                        "model": artifact(model),
                        "firmware": artifact(fw),
                    },
                    "board": board,
                }
                if log and not log.endswith("\n"):
                    log += "\n"
                log += "BENCH_PROVENANCE:" + json.dumps(
                    provenance, sort_keys=True, separators=(",", ":")) + "\n"
                with open(f"{raw}/{name}.log", "w") as f:
                    f.write(log)
                rl = next((l for l in log.splitlines() if l.startswith("BENCH_RESULT")), f"status={status}")
                print(f"  {name}: {rl[:140]}")
finally:
    c.close()
PY

echo ""
echo "Collecting + validating..."
python3 "$HERE/scripts/results.py" "$RUN_RAW" \
    -o "$COLLECTED_SUMMARY" --require-provenance

if [[ " ${CONFIGS[*]} " == *" cmsis_nn "* ]] \
        && [[ " ${CONFIGS[*]} " == *" tflm "* ]]; then
    python3 "$HERE/scripts/validate_accuracy.py" "$COLLECTED_SUMMARY"
else
    echo "Skipping cross-framework parity: this subset has no TFLM/CMSIS pair."
fi

# Promote only a completely collected invocation. A subset updates its selected
# raw logs but cannot silently replace the canonical 27-cell summary.
cp "$RUN_RAW"/*.log "$RAW"/
SUMMARY_OUTPUT="${BENCH_SUMMARY_OUTPUT:-}"
if [ "$CANONICAL_RUN" -eq 1 ]; then
    SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-$HERE/results/summary.json}"
fi
if [ -n "$SUMMARY_OUTPUT" ]; then
    mkdir -p "$(dirname "$SUMMARY_OUTPUT")"
    cp "$COLLECTED_SUMMARY" "$SUMMARY_OUTPUT"
    echo "Promoted summary to $SUMMARY_OUTPUT"
else
    echo "Subset run complete; canonical summary left unchanged."
fi
