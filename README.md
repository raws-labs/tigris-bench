# tigris-bench

Reproducible benchmarks for **TiGrIS**, a tiling ahead-of-time compiler plus INT8 runtime for
embedded devices, against **TFLite Micro**, across MCU targets. Every number below comes from a
pinned on-device run recorded in this repo.

> Looking for the *story* behind the numbers? Read the [TiGrIS blog and docs](https://tigris-ml.dev).
> This repo is the reproducible evidence; the site is the narrative.

## Headline results

### Cortex-M: deployability (does it fit and run?)

NUCLEO-H753ZI (Cortex-M7, on-chip SRAM only), INT8, CMSIS-NN. Same weights on both sides: TiGrIS
compiles its plan from the TFLite file TFLM runs. SRAM is the measured peak from
`cortex-m/deployability-hil/results/summary.json`, in KB of 1000 bytes.

| Model | TFLite Micro | TiGrIS |
|---|---|---|
| **MobileNetV2** (alpha 0.35, 224x224) | **OOM**, `AllocateTensors` -> `ARENA_TOO_SMALL` | **runs**, tiled to 307 KB SRAM, 0.37 s |
| DS-CNN (keyword spotting) | 22.7 KB SRAM, 12.8 ms | 16.4 KB, 11.1 ms |
| Anomaly detection | 15.8 KB SRAM, 1.2 ms | 1.0 KB, 1.2 ms |
| Time-series forecast | 3.0 KB SRAM, 0.3 ms | 1.5 KB, 0.3 ms |

TiGrIS tiles activations so the working set fits on-chip SRAM. TFLM's single contiguous arena
can't, so **MobileNetV2 does not run at all**. Tiling turns "won't fit" into "runs."
`tigris export` rewrites the same MobileNetV2 so that stock TFLM runs it on the H753 in 424 KB,
with output identical to TiGrIS.

Both MobileNetV2 cells, recorded on a NUCLEO-H753ZI with the binaries this suite builds at the
pinned releases; each page plays back the board's serial output and lists the firmware hash:

- [![Run on SiliconRig](https://siliconrig.dev/badge/run-on-siliconrig.svg)](https://siliconrig.dev/record/rec_56ae7e9325b750ac2282d359)
  TFLite Micro: `AllocateTensors` fails, `ARENA_TOO_SMALL`
- [![Run on SiliconRig](https://siliconrig.dev/badge/run-on-siliconrig.svg)](https://siliconrig.dev/record/rec_b91a83d0da9b4080be23b379)
  TiGrIS: 374.62 ms median over 30 runs, 307023 B SRAM peak

### ESP32-S3: latency (how fast, with acceleration?)

ESP32-S3, INT8. TiGrIS dispatches to ESP-NN; TFLM uses its optimized kernels.

| Model | TFLite Micro | TiGrIS (ESP-NN) | TiGrIS (portable ref) |
|---|---|---|---|
| DS-CNN | 23.3 ms | **21.0 ms** | 535 ms |

The ESP-NN path is **25x faster** than TiGrIS's own portable reference kernel, and 1.11x faster
than TFLM on DS-CNN.

The three DS-CNN cells, recorded on an ESP32-S3 with the images behind this table; each page
plays back the board's serial output and lists the firmware hash:

- [![Run on SiliconRig](https://siliconrig.dev/badge/run-on-siliconrig.svg)](https://siliconrig.dev/record/rec_ed72e50dd34627e4a36b1d38)
  TFLite Micro: 23.31 ms mean over 10 runs
- [![Run on SiliconRig](https://siliconrig.dev/badge/run-on-siliconrig.svg)](https://siliconrig.dev/record/rec_b8a16a02789ad7e3e9c545c2)
  TiGrIS (ESP-NN): 21.02 ms mean over 10 runs
- [![Run on SiliconRig](https://siliconrig.dev/badge/run-on-siliconrig.svg)](https://siliconrig.dev/record/rec_9047b005e94312530a871e36)
  TiGrIS (portable reference): 535.11 ms mean over 10 runs

### Emulated Cortex-M55: capability (functional)

Larger models on an emulated Arm Cortex-M55 (QEMU Corstone-300). QEMU is functional, **not
cycle-accurate**, so these are correctness + memory-fit results, not latency.

| Model | Result |
|---|---|
| ResNet-50 (25 M params) | runs, 508 KB fast-SRAM working set, bit-exact vs host |
| MobileNetV2-0.35 | runs SRAM-only in 127 KB (tiled 5.8x from a 735 KB peak), bit-exact |

## Reproduce

Each target is a self-contained "clone, build, flash, same numbers" unit:

| Path | Target | How |
|---|---|---|
| [`cortex-m/deployability-hil/`](cortex-m/deployability-hil) | H753, F446, RP2350 | real hardware (SiliconRig HIL) |
| [`esp32s3/latency-hil/`](esp32s3/latency-hil) | ESP32-S3 | real hardware (SiliconRig HIL) |
| [`cortex-m/m55-qemu/`](cortex-m/m55-qemu) | emulated Cortex-M55 | QEMU, no hardware needed |

Models are prepared once in [`models/`](models) and shared by every target. Compiler/runtime
releases are pinned per suite in `core-versions.json`, fetched and enforced by the tooling in `common/`.

## Methodology

- **Weight-matched.** TiGrIS compiles the same TFLite file TFLM runs, so both sides execute the
  *identical* INT8 weights and every cell is apples-to-apples.
- **Honest RAM.** SRAM figures are the measured working set, not a provisioned arena.
- **Provenance.** Every result carries the compiler/runtime commit and plan schema, validated in CI.

## Layout

```
cortex-m/
  deployability-hil/   # H753/F446/RP2350 on real hardware
  m55-qemu/            # emulated Cortex-M55
esp32s3/
  latency-hil/         # ESP32-S3 on real hardware
models/                # shared model prep (one source of truth)
common/                # version + provenance tooling
core-versions.json     # per-suite TiGrIS release pins
```
