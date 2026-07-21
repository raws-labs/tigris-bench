# tflm-esp32s3

Reproducible benchmarks: **TiGrIS vs TFLite Micro** on ESP32-S3. DS-CNN and MobileNetV1, 10 configurations, machine-parseable output.

This is the suite that backs the numbers quoted in the [Introducing TiGrIS](https://tigris-ml.dev/blog/introducing-tigris) post.

## Benchmark matrix

### Case A: DS-CNN (keyword spotting)

Model fits in SRAM, so tiling is not needed. The INT8 TiGrIS source is
reconstructed from the exact TFLite model embedded in the TFLM firmware; those
two cells therefore compare matched weights, quantization, input, and ESP-NN
kernels. The float models are independently initialized diagnostic cells and
must not be presented as a framework-overhead comparison.

| # | Config | Framework | dtype | Kernel |
|---|--------|-----------|-------|--------|
| 1 | TiGrIS f32 (ref) | TiGrIS | f32 | `tigris_dispatch_kernel` |
| 2 | TiGrIS i8 (ref) | TiGrIS | int8 | `tigris_dispatch_kernel_s8` |
| 3 | TiGrIS i8 (ESP-NN) | TiGrIS | int8 | `tigris_dispatch_kernel_esp_nn` |
| 4 | TFLM f32 | TFLite Micro | f32 | default |
| 5 | TFLM i8 | TFLite Micro | int8 | ESP-NN |

### Case B: MobileNetV1 (image classification)

Model is close to the SRAM ceiling. Measures tiling overhead as the budget shrinks, and shows the point at which TFLM can no longer fit the model.

| # | Config | Framework | Budget | Expected |
|---|--------|-----------|--------|----------|
| 6 | TiGrIS MBV1 i8 ESP-NN | TiGrIS | 256K | Runs, no tiling needed |
| 7 | TFLM MBV1 i8 | TFLite Micro | 256K | Fails (arena too small) |
| 8 | TiGrIS MBV1 i8 ESP-NN | TiGrIS | 128K | Runs, chain-tiled |
| 9 | TiGrIS MBV1 i8 ESP-NN | TiGrIS | 64K | Runs, spatially tiled |
| 10 | TiGrIS MBV1 i8 ESP-NN | TiGrIS | 32K | Runs, spatially tiled |

## Hardware

ESP32-S3-DevKitC-1 (N16R8): dual Xtensa LX7 at 240 MHz, 512 KB SRAM, 8 MB PSRAM, 16 MB flash.

## Prerequisites

Before running the device builds, you also need the TiGrIS C runtime source tree next to this repo. The ESP-IDF components pull headers and source files from it at build time:

```bash
# From the directory that contains tigris-bench, check out the exact compiler
# and runtime commits recorded in ../tigris-bench/core-versions.json
python tigris-bench/scripts/check_core_versions.py
```

By default the build looks for `tigris-runtime/` as a sibling of `tigris-bench/`. Override with `-DTIGRIS_RUNTIME_DIR=/path/to/tigris-runtime` on the `idf.py build` invocation if you keep it elsewhere.

The orchestration script refuses mismatched or modified compiler/runtime
checkouts before flashing. `TIGRIS_ALLOW_UNPINNED_CORE=1` is available only for
non-canonical development runs.

## Quick start

### 1. Prepare models (host)

```bash
pip install -r requirements.txt
pip install tigris-ml
python models/prepare.py
```

This generates ONNX source models, TFLite models, C headers, and ONNX Runtime/TFLite reference outputs under `models/output/`. `run_all.sh` compiles the TiGrIS plans afresh into ignored `build/plans/`, using the active local compiler, before it touches hardware. It never benchmarks a pre-existing `.tgrs` artifact.

### 2. Run all benchmarks (device)

Local USB:

```bash
./scripts/run_all.sh /dev/ttyUSB0
```

SiliconRig (one remote board session for the complete matrix):

```bash
host_python="$(command -v python3)"
source /path/to/esp-idf/export.sh
SRIG_API_KEY=... PYTHON="$host_python" BENCH_TRANSPORT=siliconrig \
  ./scripts/run_all.sh
```

The host Python environment needs this directory's requirements plus the
SiliconRig SDK. Capturing its path before sourcing ESP-IDF avoids accidentally
using ESP-IDF's private Python environment, which does not carry the benchmark
model or SiliconRig packages.

For TiGrIS cells the SiliconRig path merges the bootloader, partition table,
application, and freshly compiled plan into one raw image. The plan is placed
at the partition-table offset `0x210000`; the runner has a regression test that
prevents the old, incorrect `0x60000` app-partition offset from returning.

`run_all.sh` now accepts a result only after all 10 canonical cells were
captured through `BENCH_DONE`, aggregated, and checked against their
model-specific references. A timeout, missing or extra cell, unexpected status,
or numerical mismatch makes the command fail.

Or run individual configs:

```bash
# TiGrIS f32
cd tigris-esp && idf.py set-target esp32s3 && idf.py build && idf.py flash
# Compile the current plan, then flash it to the "plan" partition
python ../scripts/prepare_tigris_plans.py --compiler ../../../tigris/.venv/bin/tigris \
  --models-dir ../models/output --output-dir ../build/plans
python -m esptool --port /dev/ttyUSB0 write_flash 0x210000 ../build/plans/ds_cnn.tgrs

# TiGrIS i8 + ESP-NN
cd tigris-esp && idf.py fullclean && idf.py set-target esp32s3
idf.py build -DBENCH_KERNEL=esp_nn && idf.py flash
python -m esptool --port /dev/ttyUSB0 write_flash 0x210000 ../build/plans/ds_cnn_i8.tgrs

# TFLM f32
cd tflm-esp && idf.py set-target esp32s3 && idf.py build && idf.py flash

# TFLM i8
cd tflm-esp && idf.py fullclean && idf.py set-target esp32s3
idf.py build -DBENCH_INT8=1 && idf.py flash
```

### 3. Re-run the result gates manually

```bash
python scripts/results.py results/raw/ -o results/summary.json
python scripts/validate_accuracy.py results/summary.json models/output/
```

`results.py` parses the serial logs and prints a pretty table; with `-o` it also
writes the tracked, machine-parseable `summary.json`. `validate_accuracy.py`
compares device outputs against the corresponding ONNX Runtime or TFLite
reference and rejects the run if the numbers drift beyond the matrix cell's
explicit tolerance. The TFLM DS-CNN ESP-NN cell permits at most 4 INT8 output
units relative to the host TFLite interpreter; TiGrIS cells permit at most 1.
Raw logs remain gitignored.

INT8 references are raw int8 vectors. The TiGrIS INT8 models are reconstructed
from the exact TFLite models embedded in the firmware, so their device outputs
and the corresponding TFLM cells share weights and quantization. The TFLM
MobileNet cell is required to report the expected `ARENA_TOO_SMALL` status.
The independently initialized float cells retain model-specific references and
are not used for cross-framework claims.

For an intentional development run containing only some canonical cells, pass
`--allow-partial` to `results.py`. This never permits unknown filenames,
malformed/truncated logs, or invalid cell contents.

## Project structure

```
models/prepare.py             # Build and quantize ONNX/TFLite source artifacts
scripts/prepare_tigris_plans.py # Compile current TiGrIS plans into ignored build output
tigris-esp/                   # ESP-IDF project: TiGrIS benchmark harness
tflm-esp/                     # ESP-IDF project: TFLite Micro benchmark harness
scripts/run_all.sh            # Orchestrate all configs end-to-end
scripts/results.py            # Parse logs, print table, optionally emit JSON
scripts/validate_accuracy.py  # Compare device output against ORT reference
scripts/benchmark_matrix.py   # Canonical 10-cell matrix and structural gate
tests/test_validation.py      # Self-contained result/accuracy fixtures
```

Run the host-only gate tests with:

```bash
python -m unittest discover -s tests -v
```

To run the same post-capture gates without building or touching hardware:

```bash
BENCH_VALIDATE_ONLY=1 ./scripts/run_all.sh
```

## Dependencies

- Host: Python 3.10+, `onnx`, `onnxruntime`, `tensorflow`, `tigris-ml`, `pyserial`, `esptool`
- Device: ESP-IDF 5.x, `esp-tflite-micro ~1.3.1`
- Hardware: ESP32-S3 board with 16 MB flash and 8 MB PSRAM
- Runtime: a local checkout of [tigris-runtime](https://github.com/raws-labs/tigris-runtime) (see Prerequisites)
