# tigris-bench

Reproducible benchmarks for **TiGrIS** — a tiling ahead-of-time compiler + INT8 runtime for
microcontrollers — against **TFLite Micro**, across MCU targets. Every number below comes from a
pinned on-device run recorded in this repo.

> Looking for the *story* behind the numbers? Read the [TiGrIS blog and docs](https://tigris-ml.dev).
> This repo is the reproducible evidence; the site is the narrative.

## Headline results

### Cortex-M — deployability (does it fit and run?)

NUCLEO-H753ZI (Cortex-M7, on-chip SRAM only), INT8, CMSIS-NN. Same weights on both sides — TiGrIS
runs the exact model reconstructed from TFLM's own TFLite file.

| Model | TFLite Micro | TiGrIS |
|---|---|---|
| **MobileNetV2** (α0.35, 224²) | **OOM** — `AllocateTensors` → `ARENA_TOO_SMALL` | **runs** — tiled to 299 KB SRAM, 1.43 s |
| DS-CNN (keyword spotting) | 22.7 KB SRAM · 12.8 ms | 17.0 KB · 11.1 ms |
| Anomaly detection | 15.8 KB SRAM · 1.2 ms | 2.9 KB · 1.2 ms |
| Time-series forecast | 3.0 KB SRAM · 0.3 ms | 2.4 KB · 0.3 ms |

TiGrIS tiles activations so the working set fits on-chip SRAM. TFLM's single contiguous arena
can't, so **MobileNetV2 does not run at all** — the headline: tiling turns "won't fit" into "runs."

### ESP32-S3 — latency (how fast, with acceleration?)

ESP32-S3, INT8. TiGrIS dispatches to ESP-NN; TFLM uses its optimized kernels.

| Model | TFLite Micro | TiGrIS (ESP-NN) | TiGrIS (portable ref) |
|---|---|---|---|
| DS-CNN | 30.4 ms | **29.4 ms** | 629 ms |

The ESP-NN path brings TiGrIS to parity with TFLM — and is **21× faster** than its own portable
reference kernel, showing the accelerated dispatch works.

### Emulated Cortex-M55 — capability (functional)

Larger models on an emulated Arm Cortex-M55 (QEMU Corstone-300). QEMU is functional, **not
cycle-accurate**, so these are correctness + memory-fit results, not latency.

| Model | Result |
|---|---|
| ResNet-50 (25 M params) | runs, 508 KB fast-SRAM working set, bit-exact vs host |
| MobileNetV2-0.35 | runs SRAM-only in 127 KB (tiled 5.8× from a 735 KB peak), bit-exact |

## Reproduce

Each target is a self-contained "clone → build → flash → same numbers" unit:

| Path | Target | How |
|---|---|---|
| [`cortex-m/deployability-hil/`](cortex-m/deployability-hil) | H753 · F446 · RP2350 | real hardware (SiliconRig HIL) |
| [`esp32s3/latency-hil/`](esp32s3/latency-hil) | ESP32-S3 | real hardware (SiliconRig HIL) |
| [`cortex-m/m55-qemu/`](cortex-m/m55-qemu) | emulated Cortex-M55 | QEMU, no hardware needed |

Models are prepared once in [`models/`](models) and shared by every target. Compiler/runtime
versions are pinned in `core-versions.json` and enforced by the tooling in `common/`.

## Methodology

- **Weight-matched.** TiGrIS runs the *identical* INT8 weights as TFLM, reconstructed from the same
  TFLite file, so every cell is apples-to-apples.
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
core-versions.json     # pinned compiler/runtime
```
