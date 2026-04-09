# tflm-esp32s3

Reproducible benchmarks: **TiGrIS vs TFLite Micro** on ESP32-S3. DS-CNN and MobileNetV1, 10 configurations, machine-parseable output.

This is the suite that backs the numbers quoted in the [Introducing TiGrIS](https://tigris-ml.dev/blog/introducing-tigris) post.

## Benchmark matrix

### Case A: DS-CNN (keyword spotting)

Model fits in SRAM, so tiling is not needed. Measures framework overhead at parity on the same kernels.

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
| 7 | TiGrIS MBV1 i8 ESP-NN | TiGrIS | 128K | Runs, chain-tiled |
| 8 | TiGrIS MBV1 i8 ESP-NN | TiGrIS | 64K | Runs, spatially tiled |
| 9 | TiGrIS MBV1 i8 ESP-NN | TiGrIS | 32K | Runs, spatially tiled |
| 10 | TFLM MBV1 i8 | TFLite Micro | 256K | Fails (arena too small) |

## Hardware

ESP32-S3-DevKitC-1 (N16R8): dual Xtensa LX7 at 240 MHz, 512 KB SRAM, 8 MB PSRAM, 16 MB flash.

## Prerequisites

Before running the device builds, you also need the TiGrIS C runtime source tree next to this repo. The ESP-IDF components pull headers and source files from it at build time:

```bash
# From the directory that contains tigris-bench
git clone https://github.com/raws-labs/tigris-runtime
```

By default the build looks for `tigris-runtime/` as a sibling of `tigris-bench/`. Override with `-DTIGRIS_RUNTIME_DIR=/path/to/tigris-runtime` on the `idf.py build` invocation if you keep it elsewhere.

## Quick start

### 1. Prepare models (host)

```bash
pip install -r requirements.txt
pip install tigris-ml
python models/prepare.py
```

This generates ONNX, TiGrIS plans, TFLite models, C headers, and ORT reference outputs under `models/output/`.

### 2. Run all benchmarks (device)

```bash
./scripts/run_all.sh /dev/ttyUSB0
```

Or run individual configs:

```bash
# TiGrIS f32
cd tigris-esp && idf.py set-target esp32s3 && idf.py build && idf.py flash
# Flash the .tgrs plan to the "plan" partition
python -m esptool --port /dev/ttyUSB0 write_flash 0x210000 ../models/output/ds_cnn.tgrs

# TiGrIS i8 + ESP-NN
cd tigris-esp && idf.py fullclean && idf.py set-target esp32s3
idf.py build -DBENCH_KERNEL=esp_nn && idf.py flash
python -m esptool --port /dev/ttyUSB0 write_flash 0x210000 ../models/output/ds_cnn_i8.tgrs

# TFLM f32
cd tflm-esp && idf.py set-target esp32s3 && idf.py build && idf.py flash

# TFLM i8
cd tflm-esp && idf.py fullclean && idf.py set-target esp32s3
idf.py build -DBENCH_INT8=1 && idf.py flash
```

### 3. Collect results

```bash
python scripts/results.py results/raw/ -o results/summary.json
python scripts/validate_accuracy.py results/summary.json models/output/
```

`results.py` parses the serial logs and prints a pretty table; with `-o` it also writes a machine-parseable `summary.json`. `validate_accuracy.py` compares device outputs against the ORT reference and rejects the run if the numbers drift beyond tolerance.

## Project structure

```
models/prepare.py             # Build, quantize, compile all model variants
tigris-esp/                   # ESP-IDF project: TiGrIS benchmark harness
tflm-esp/                     # ESP-IDF project: TFLite Micro benchmark harness
scripts/run_all.sh            # Orchestrate all configs end-to-end
scripts/results.py            # Parse logs, print table, optionally emit JSON
scripts/validate_accuracy.py  # Compare device output against ORT reference
```

## Dependencies

- Host: Python 3.10+, `onnx`, `onnxruntime`, `tensorflow`, `tigris-ml`, `pyserial`, `esptool`
- Device: ESP-IDF 5.x, `esp-tflite-micro ~1.3.1`
- Hardware: ESP32-S3 board with 16 MB flash and 8 MB PSRAM
- Runtime: a local checkout of [tigris-runtime](https://github.com/raws-labs/tigris-runtime) (see Prerequisites)
