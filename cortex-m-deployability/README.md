# Cortex-M Deployability Benchmark: TiGrIS vs TFLite Micro

Measures inference latency, activation-memory working set, and bit-exact
device-to-device output parity for the TiGrIS runtime against TFLite Micro on
three Cortex-M architectures, on INT8 models. Sibling to `../tflm-esp32s3/` (the
same comparison on ESP32-S3). Build and run: see `BUILD.md`.

## Boards

| Board | MCU | Core | Clock | Flash | RAM |
|---|---|---|---|---|---|
| NUCLEO-H753ZI | STM32H753ZI | Cortex-M7 | 480 MHz | 2 MB | 1 MB (+128 KB DTCM / 64 KB ITCM) |
| NUCLEO-F446RE | STM32F446RE | Cortex-M4F | 180 MHz | 512 KB | 128 KB |
| Raspberry Pi Pico 2 W | RP2350 | dual Cortex-M33 | 150 MHz | 4 MB (ext QSPI) | 520 KB |

## Method

TiGrIS and TFLite Micro run the same INT8 models: identical weights,
reconstructed from TFLM's `.tflite` via `tools/tflite_to_qdq_onnx.py`, both
linking the same pinned CMSIS-NN commit. The comparison is TiGrIS vs TFLM, both
on cmsis_nn, device-to-device. The x86 host tflite interpreter rounds gemmlowp
ties differently from on-device CMSIS-NN, so it is not used as a baseline. All
boards build at `-O2`; each runs at its rated clock, verified per run by a clock
guard in `results.py`. Latency is the median of 30 runs (DWT cycle counter on
M7/M4, hardware `time_us` on M33). RAM is the activation working set: TiGrIS
`sram_peak` (fast-arena peak + slow-pool spill + scratch + tensor table +
executor workspace) against TFLM `arena_used`; weights live in flash and are
excluded, as is stack. Captured remotely on a SiliconRig hardware-in-the-loop
lab via `scripts/run_all.sh`.

For each TiGrIS cell, CMake runs `tigris codegen <plan> --backend
{cmsis-nn|reference} --format core`. The generated, backend-specific deployment
core loads the embedded plan, resets runtime memory, prepares the backend, and
selects/executes its dispatcher. The benchmark harness supplies only the
board entry point, statically sized arenas, benchmark input pattern, timing, and
machine-parseable reporting.

Three INT8 models are used: DS-CNN (keyword spotting, conv/depthwise, ~23K
params / 24 KB weights), an anomaly-detection dense autoencoder (~266K params /
265 KB weights, 10 FC layers), and a 1D-signal timeseries CNN (3 strided convs +
global-average + Dense, ~5.4K params / 5.5 KB weights). Parameter counts are
int8 weights plus biases, measured from the committed plans.

## NUCLEO-H753ZI (Cortex-M7 @ 480 MHz)

**DS-CNN:**

| Framework | Kernel | Latency | Cycles | RAM (work. set) | Flash (firmware) |
|---|---|---|---|---|---|
| TiGrIS | cmsis_nn | 11.14 ms | 5.35 M | 25.5 KB | 118 KB |
| TFLM | cmsis_nn | 12.80 ms | 6.14 M | 22.2 KB | 176 KB |
| TiGrIS | s8_ref | 81.96 ms | 39.34 M | 25.3 KB | 93 KB |

**Anomaly detection:**

| Framework | Kernel | Latency | Cycles | RAM (work. set) | Flash (firmware) |
|---|---|---|---|---|---|
| TiGrIS | cmsis_nn | 1.19 ms | 570 K | 11.6 KB | 370 KB |
| TFLM | cmsis_nn | 1.16 ms | 558 K | 15.5 KB | 417 KB |
| TiGrIS | s8_ref | 3.05 ms | 1.46 M | 11.4 KB | 346 KB |

**Timeseries:**

| Framework | Kernel | Latency | Cycles | RAM (work. set) | Flash (firmware) |
|---|---|---|---|---|---|
| TiGrIS | cmsis_nn | 0.298 ms | 143 K | 11.4 KB | 87 KB |
| TFLM | cmsis_nn | 0.345 ms | 166 K | 2.9 KB | 145 KB |
| TiGrIS | s8_ref | 1.61 ms | 774 K | 10.7 KB | 62 KB |

- Output is bit-exact device-to-device: every (model, framework, kernel) cell
  emits the identical INT8 vector (max abs diff 0), checked by
  `scripts/validate_accuracy.py`.
- CMSIS-NN vs the portable reference kernels (same model, both `-O2`): 7.4x
  (DS-CNN) / 2.6x (AD) / 5.4x (TS) on the M7.
- Cycles are clock-independent; ms is at 480 MHz.
- Latency, cycles, RAM, and firmware sizes come from the same tracked,
  provenance-bearing run in `results/summary.json`.

## NUCLEO-F446RE (Cortex-M4F @ 180 MHz)

| Model | TiGrIS cmsis | TFLM cmsis | TiGrIS s8 | RAM (TiGrIS / TFLM) |
|---|---|---|---|---|
| TS | 1.56 ms | 1.80 ms | 8.41 ms | 11.4 / 2.9 KB |
| AD | 4.97 ms | 4.82 ms | 16.53 ms | 11.6 / 15.5 KB |
| DS-CNN | 63.53 ms | 68.19 ms | 467.58 ms | 25.5 / 22.2 KB |

Output is byte-identical to the H753 (same weights, two architectures). The
128 KB SRAM holds every model.

## Raspberry Pi Pico 2 W / RP2350 (Cortex-M33 @ 150 MHz)

A separate pico-sdk build (UF2 flash, `time_us` timing), same runtime and
compiler, no TFLM baseline (TFLM ships no prebuilt M33 CMSIS-NN lib). Output is
byte-identical to the H753 and F446. Weights are read from QSPI flash via XIP.

| Model | TiGrIS cmsis | TiGrIS s8 | RAM |
|---|---|---|---|
| TS | 2.68 ms | 8.37 ms | 11.4 KB |
| AD | 35.01 ms | 44.56 ms | 11.6 KB |
| DS-CNN | 67.83 ms | 412.06 ms | 25.5 KB |

The FC-heavy AD is slower here (35.01 ms vs 4.97 ms on the F446): each of its
265 KB of weights is read once per inference from XIP flash with no reuse, so it
is QSPI-bandwidth-bound. The conv models reuse weights across spatial positions
and stay fast.

## MobileNetV2 (tiling)

MobileNetV2 (alpha 0.35, 224x224, INT8, 591 KB weights, 52 convs with
inverted-residual ADD skips) has a naive activation peak of 735 KB, larger than
any of these boards' SRAM. TiGrIS tiles it to a 307.7 KB working set (127.1 KB
fast + 171.5 KB slow-pool spill + 9.1 KB scratch/runtime metadata, 2 tiled
stages), with bit-exact output across boards.

| Board (SRAM) | TiGrIS (tiled) | TFLM (no tiling) |
|---|---|---|
| H753ZI (512 KB) | runs, 1.43 s, 307.7 KB | OOM at AllocateTensors |
| RP2350 (520 KB) | runs, 7.03 s, 307.7 KB | n/a (no M33 lib) |
| F446RE (128 KB) | does not fit | does not fit |

- On the H753, TFLM given a 480 KB arena (nearly all of the 512 KB SRAM) fails
  `AllocateTensors` with `ARENA_TOO_SMALL`: with no tiling it needs the full
  735 KB. TiGrIS runs the identical model on the same board.
- The F446 cannot hold MobileNetV2: the 591 KB weight blob exceeds its 512 KB
  flash, and the 300 KB tiled working set exceeds its 128 KB SRAM.
- RP2350 is ~4.9x slower than the H753 on this model (XIP-bound: 591 KB of
  weights streamed from QSPI flash each inference, plus the lower clock).

## Reproduce

`SRIG_API_KEY=... ./scripts/run_all.sh` builds every cell, flashes and captures
on the SiliconRig lab, aggregates, and runs the parity gate. Raw serial logs land
in `results/raw/` (gitignored); the tracked artifact is `results/summary.json`,
which carries each cell's numbers, `OUTPUT_I8` vector, and deduplicated execution
provenance. See `BUILD.md` for the build knobs and a locally-attached-board
(no-rig) path.

`results/provenance.json` binds that summary and its collector/validator to
SHA-256 digests and records the paths and hashes of all 27 source captures. The
summary embeds the actual compiler, runtime, TFLM, CMSIS and SDK revisions;
tool and Python-environment versions; build invocations; model and firmware
hashes/sizes; capture timestamps; and SiliconRig board identities. The captures
are not shipped in a clone, but their hashes are retained so an obtained capture
set can be checked. Host validation always checks tracked artifact/tool hashes
and also checks source-capture hashes when those gitignored files are present.
When the complete capture set is available, it reruns the collector in a
temporary directory and requires byte-identical `summary.json` output; a clean
clone skips that reconstruction and does not report it as verified.
