# tigris-bench

Reproducible benchmarks of TiGrIS (tiling ahead-of-time compiler plus INT8 runtime) against TFLite Micro on microcontroller targets; every tracked number comes from a pinned on-device or emulated run recorded in this repo. Python host tooling, ESP-IDF projects, CMake with arm-none-eabi, and bare-metal QEMU firmware. Public, github.com/raws-labs/tigris-bench.

## Build, test, run
- Host validation (what CI runs, no hardware): `pip install -r .github/requirements-host-validation.txt`, then `python -m unittest discover -s esp32s3/latency-hil/tests`, `python -m unittest discover -s common -p 'test_*.py'`, `python common/validate_tracked_json.py`, `python common/check_core_versions.py --manifest-only`, and from `cortex-m/deployability-hil/`: `python scripts/validate_accuracy.py results/summary.json`.
- Model prep, shared by all targets, from `models/`: `python prepare.py` (also `prepare_ad.py`, `prepare_ts.py`, `prepare_matched.py`, `prepare_mobilenetv2.py`); artifacts go to the gitignored `models/output/`.
- ESP32-S3 (`esp32s3/latency-hil/`): `pip install -r requirements.txt`; `./scripts/run_all.sh /dev/ttyUSB0` on a local board, or `SRIG_API_KEY=... PYTHON=<host python> BENCH_TRANSPORT=siliconrig ./scripts/run_all.sh` on the remote lab; then `python scripts/results.py results/raw/ -o results/summary.json` and `python scripts/validate_accuracy.py results/summary.json models/output/`.
- Cortex-M (`cortex-m/deployability-hil/`): `third_party/fetch.sh` once (vendors CMSIS-NN, CMSIS-Core, ST device pack), then `scripts/run_all.sh [h753|f446|rp2350 ...]` (needs `SRIG_API_KEY`; `BENCH_MODELS=... BENCH_CONFIGS=...` narrow it to one cell). Manual CMake and local-board flow in `BUILD.md`.
- Emulated Cortex-M55 (`cortex-m/m55-qemu/`): `./build.sh <plan> <name> -DFAST_KB=... -DSLOW_KB=...` (or `-DSLOW_DDR`), then `./run.sh <plan>`; needs arm-none-eabi-gcc 13.2+ and qemu-system-arm 8.2+ with `mps3-an547`.

## Layout
- `common/`: core-version pin checks and provenance validation shared by every suite.
- `core-versions.json`: exact compiler, runtime and tigris-cortex-m commits the tracked results were produced with.
- `<target>/results/summary.json` is the tracked artifact; `results/raw/` serial logs are gitignored.
- `cortex-m/deployability-hil/tools/tflite_to_qdq_onnx.py`: reconstructs the TiGrIS ONNX model from the TFLite file TFLM runs, so both sides share weights.

## Conventions
- Cells are weight-matched: TiGrIS INT8 models come from the exact TFLite file embedded in the TFLM firmware. Float cells are independently initialized and are never used for cross-framework claims.
- RAM figures are the measured working set (TiGrIS `sram_peak` vs TFLM `arena_used`), not a provisioned arena; weights and stack are excluded.
- Plans are always recompiled into gitignored `build/plans/` with the active local compiler; a pre-existing `.tgrs` is never benchmarked.
- The compiler and runtime checkouts sit next to this repo at the commits in `core-versions.json`; `run_all.sh` refuses anything else. `TIGRIS_ALLOW_UNPINNED_CORE=1` is for non-canonical development runs only, and such results are not promoted to a tracked summary.
- Tracked summaries keep the commits that produced them. Re-pinning `core-versions.json` without rerunning hardware is only valid for CI-safe maintenance (host CMSIS parity uses a synthetic fixture, not tracked plans); the compatibility-manifest and compiler pins move together.
- Every raw capture carries a provenance record (repos, dependencies, tools, build invocation, model, firmware, board); collection fails without it.

## Gotchas
- A run is accepted only after every canonical cell is captured through the `BENCH_DONE` sentinel and passes its per-cell tolerance; `--allow-partial` on `results.py` is for intentional development runs.
- ESP32-S3 TiGrIS cells flash the plan to the partition at offset `0x210000`; a regression test guards against the old `0x60000` app-partition offset.
- Capture the host Python path before sourcing ESP-IDF `export.sh`; ESP-IDF's private Python lacks the benchmark and SiliconRig packages.
- The device resets after `idf.py flash`; open the serial port right away and read until `BENCH_DONE`.
- TFLM baseline libraries are built with `BUILD_TYPE=release_with_logs`; the `default` build keeps debug asserts on and slows TFLM by 5-10%.
- The TFLM MobileNet cells are required to report `ARENA_TOO_SMALL`; that status is the expected result, not a broken run.
- QEMU's Cortex-M55 is functional, not cycle-accurate: correctness and memory-fit only, never latency claims.
- Cortex-M boards must run at their rated clock; `results.py` has a clock guard that rejects captures otherwise.
