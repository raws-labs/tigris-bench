# tigris-bench

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-tigris--ml.dev-green)](https://tigris-ml.dev/docs)

**Reproducible benchmarks for TiGrIS.** The source of truth behind every number quoted in the [TiGrIS](https://github.com/raws-labs/tigris) docs and blog.

Each benchmark suite is self-contained and lives in its own subdirectory: model preparation on the host, a device harness, and scripts that collect machine-parseable results. Anyone with the matching hardware should be able to reproduce any number end-to-end.

## How it is organized

Every suite follows the same three-step shape:

1. **Prepare** models on the host: ONNX, quantization, TiGrIS compilation, and reference outputs from ONNX Runtime.
2. **Run** the device harness for each configuration under test.
3. **Collect** results into JSON, format them as tables, and validate device outputs against the reference to catch numerical drift.

Suites are grouped by what they are measuring (e.g. latency against a peer framework, tiling overhead across memory budgets, end-to-end demos). See each suite's own README for its benchmark matrix, hardware, and exact commands.

## Quick start

```bash
# Install the TiGrIS toolchain (used by every suite's model preparation)
pip install tigris-ml

# Enter the suite you want to run, then follow its README
cd <suite>/
pip install -r requirements.txt
python models/prepare.py
./scripts/run_all.sh /dev/ttyUSB0
python scripts/results.py results/raw/ -o results/summary.json
```

Accuracy validation is part of the pipeline: `validate_accuracy.py` compares device outputs against the ORT reference, so a run is rejected if the numbers do not match within tolerance.

## Common dependencies

- Host: Python 3.10+, `onnx`, `onnxruntime`, `tigris-ml`, plus whatever a specific suite needs
- Device: ESP-IDF 5.x for ESP32 suites; toolchains for other targets as applicable
- Hardware: varies per suite, listed in each suite's README

## Further reading

- [Introducing TiGrIS](https://tigris-ml.dev/blog/introducing-tigris): context for the numbers and how tiling works
- [TiGrIS](https://github.com/raws-labs/tigris) and [tigris-runtime](https://github.com/raws-labs/tigris-runtime): the toolchain and runtime being benchmarked
