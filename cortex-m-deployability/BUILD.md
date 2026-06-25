# Building and running the Cortex-M deployability benchmark

This is the how-to. For the why and the experiment design, see `README.md`.

Status: runs on hardware. NUCLEO-H753ZI (Cortex-M7 @ 480 MHz), NUCLEO-F446RE
(Cortex-M4F @ 180 MHz), and Raspberry Pi Pico 2 / RP2350 (Cortex-M33 @ 150 MHz),
DS-CNN / anomaly-detection / timeseries models, TiGrIS (cmsis_nn + s8_ref) vs
TFLite Micro, bit-exact device-to-device. See `README.md` for the results table.

## Prerequisites (host)

```bash
# ARM cross toolchain + flashing/serial tools (need sudo - run yourself)
sudo apt install gcc-arm-none-eabi binutils-arm-none-eabi
# (openocd / stlink-tools are optional; drag-drop flashing needs no tool)

# Python deps for the result scripts
pip install pyserial numpy rich
```

You are already in the `dialout` group, so `/dev/ttyACM0` is readable.

## One-shot: build, flash, capture, validate all configs

```bash
scripts/run_all.sh                 # s8_ref and cmsis_nn
scripts/run_all.sh cmsis_nn        # a single kernel
```

This vendors CMSIS (first run), builds each config, flashes over the ST-LINK
drag-drop drive, captures serial until `BENCH_DONE`, then writes
`results/summary.json` and validates against `ds_cnn_reference_i8.bin`.

## Manual steps

```bash
# 1. Vendor CMSIS-NN v7.0.0 + CMSIS-Core v6.3.0 + ST H7 device pack
third_party/fetch.sh

# 2. Configure + build (CMSIS-NN backend, DS-CNN INT8 plan)
cmake -S . -B build/h753_cmsis_nn \
  -DCMAKE_TOOLCHAIN_FILE=../../tigris-runtime/cmake/arm-none-eabi.cmake \
  -DTIGRIS_BOARD=nucleo_h753zi \
  -DBENCH_KERNEL=cmsis_nn
cmake --build build/h753_cmsis_nn -j
# -> build/h753_cmsis_nn/tigris_bench.bin  (+ a size report: code/rodata vs RAM)

# 3. Flash (copies the .bin to the NOD_H753ZI drive)
scripts/flash.sh build/h753_cmsis_nn/tigris_bench.bin

# 4. Capture the run (board resets and prints over USART3 -> VCP)
python3 scripts/capture_serial.py -o results/raw/h753_cmsis_nn.log

# 5. Table + parity
python3 scripts/results.py results/raw/ -o results/summary.json
python3 scripts/validate_accuracy.py results/summary.json ../tflm-esp32s3/models/output
```

## TFLM baseline lib (per core)

The TFLM firmware links a prebuilt `libtensorflow-microlite.a`, one per core. Build
it from the vendored `third_party/tflite-micro` (uses the same pinned CMSIS-NN as
TiGrIS, `6d9d61d8`, so the kernels are byte-identical). The CMake picks the lib for
the board's core via `BOARD_TFLM_ARCH` in `board.cmake`.

Build it with `BUILD_TYPE=release_with_logs` (TFLM's recommended benchmark build:
`-DNDEBUG`, error strings kept). The `default` build leaves debug asserts/checks on,
which makes it ~5-10% slower and would unfairly handicap the TFLM latency baseline.

```bash
cd third_party/tflite-micro
PATH="$PWD/tensorflow/lite/micro/tools/make/downloads/gcc_embedded/bin:$PATH" \
  make -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=cortex_m_generic TARGET_ARCH=cortex-m7+fp \   # H753: m7 ; F446: cortex-m4+fp
  BUILD_TYPE=release_with_logs OPTIMIZED_KERNEL_DIR=cmsis_nn microlite -j
# -> gen/cortex_m_generic_<arch>_release_with_logs_cmsis_nn_gcc/lib/libtensorflow-microlite.a
```

Then build the TFLM firmware: `-DBENCH_FRAMEWORK=tflm -DTFLM_MODEL=<model>`
(e.g. `ds_cnn` / `ad` / `ts`), and on the F446 add `-DTFLM_ARENA_BYTES=32768`
(the 128 KB default would eat the whole SRAM). The CMake finds the lib via
`TFLM_BUILD_TYPE` (default `release_with_logs`); set it to match what you built.

## RP2350 (Pico 2) firmware

The RP2350 has no on-board debugger, so it is a SEPARATE pico-sdk project
(`boards/pico2_rp2350/`) that links the same TiGrIS runtime + CMSIS-NN. It needs
`PICO_SDK_PATH` and a built `picotool`.

```bash
PICO_SDK_PATH=~/pico/pico-sdk cmake -S boards/pico2_rp2350 -B build/pico_ts_cmsis \
  -Dpicotool_DIR=~/pico/picotool/install/lib/cmake/picotool \
  -DBENCH_KERNEL=cmsis_nn -DTIGRIS_PLAN=<abs path to a .tgrs>
cmake --build build/pico_ts_cmsis -j            # -> tigris_pico_bench.uf2
python3 scripts/flash_pico.py build/pico_ts_cmsis/tigris_pico_bench.uf2 \
  results/raw/pico_ts_cmsis.log                 # 1200-baud-touch -> BOOTSEL -> UF2 -> capture
```

## Knobs

| CMake var | Default | Meaning |
|---|---|---|
| `TIGRIS_BOARD` | `nucleo_h753zi` | board under `boards/<name>/` (`nucleo_h753zi` \| `nucleo_f446re`; RP2350 is a separate pico-sdk build) |
| `BENCH_FRAMEWORK` | `tigris` | `tigris` or `tflm` |
| `BENCH_KERNEL` | `cmsis_nn` | TiGrIS backend: `cmsis_nn` or `s8_ref` |
| `BENCH_OPT_LEVEL` | `-O2` | optimization for ALL benchmark code; consistent across boards + matches TFLM's `-O2` kernels (the clean-CMake default would otherwise be `-O0`) |
| `TIGRIS_PLAN` | `ds_cnn_i8.tgrs` | the `.tgrs` plan embedded in flash (compile at a budget that fits the board) |
| `TIGRIS_FAST_ARENA_BYTES` | `131072` | static fast arena backing store; the harness provisions only a tight slice of it (`peak + align`) so the compactor engages and the measured RAM is the true minimum |
| `TIGRIS_SLOW_ARENA_BYTES` | `262144` | slow arena backing store; use `8192` on the F446 |
| `TFLM_MODEL` | `ds_cnn` | TFLM model: `<name>_tflite_i8.h` in `MODELS_DIR` |
| `TFLM_ARENA_BYTES` | `131072` | TFLM tensor arena; use `32768` on the F446 |
| `TFLM_BUILD_TYPE` | `release_with_logs` | TFLM microlite build type (must match the `gen/` lib you built) |

## Notes / open items

- **Clock:** each board runs at its rated speed (H753 480 MHz, F446 180 MHz,
  RP2350 150 MHz), with a safe fallback. The harness prints `CLOCK_DIAG: stage=5`
  on success, and `results.py` **rejects** any run whose `cpu_mhz` is off the
  per-board rated clock (`EXPECTED_MHZ`), so a silent fallback can't be published.
  Cycle counts are the clock-independent metric; ms is at the rated clock. (The
  STM32s read core cycles via DWT CYCCNT; the RP2350 uses its hardware us timer,
  as its DWT under-counts XIP-stall cycles.)
- **RAM metric:** `sram_peak_bytes` is the MEASURED runtime working set, directly
  comparable to TFLM's `arena_used_bytes`. For TiGrIS it is the fast + slow arena
  high-water + CMSIS-NN scratch + the runtime tensor table; the harness provisions
  a tight arena (`peak + alignment`) so the executor's reactive compactor engages
  and the figure is the true minimum, not a lazy bump high-water. (Provision a
  generous arena and the same model reports far more RAM - that is a measurement
  artifact, not a real cost.)
- **Vendor pins** (`fetch.sh`): CMSIS-NN `6d9d61d8` (full SHA, asserted after
  checkout; matches TFLM's bundled pin), cmsis-device-h7 `master`, cmsis-device-f4
  `3c77349`. Override via `DEV_*_REF`.
- **Parity** is checked device-to-device by `scripts/validate_accuracy.py
  results/raw/`: it groups the captured logs by (board, model) and compares the
  int8 `OUTPUT_I8` vectors of `tigris/cmsis_nn` vs `tflm/cmsis_nn` (bit-exact) and
  `tigris/s8_ref` vs `tigris/cmsis_nn` (+-1 LSB requant nudge). A (board, model)
  with no TFLM baseline is reported INCOMPLETE, never silently passed. The host
  ORT/tflite interpreter is NOT a valid baseline (different kernels).
