# TiGrIS on an emulated Arm Cortex-M55

Runs TiGrIS-compiled INT8 plans on an emulated **Arm Cortex-M55** (QEMU
`mps3-an547` / Corstone-300) as a bare-metal firmware, and reports the output
checksum + on-chip SRAM usage over ARM semihosting. Weights (the `.tgrs` plan)
are loaded into DDR at `0x60000000` via QEMU `-device loader`, modelling
external XIP flash; activations are tiled into on-chip SRAM.

Two demonstrations, both verified bit-exact against a host reference:

| Model | What it shows | On-chip SRAM | Output |
|-------|---------------|--------------|--------|
| **ResNet-50** backbone @224 | A 25M-param model runs on an MCU | 508 KB fast working set (slow tier in DDR) | checksum `-1942099862`, bit-exact |
| **MobileNetV2-0.35** @224 (full model) | Clean SRAM-only feasibility barrier | **127 KB total** (fast+slow both on-chip), tiled 5.8× from the 735 KB naive peak | checksum `5740`, bit-exact |

The MobileNetV2-0.35 run is the interesting one: its whole working set tiles into
127 KB of on-chip SRAM, whereas a non-tiling runtime (TFLM) needs the full
735 KB arena contiguous and OOMs on the chip's SRAM. The bench already measures
that OOM on real hardware (NUCLEO-H753ZI: `AllocateTensors` → `ARENA_TOO_SMALL`).

## Honest caveats

- **No latency numbers.** QEMU's Cortex-M55 is functional, not cycle-accurate.
  These runs prove *correctness, tiling, and memory placement*, not speed.
- **The barrier needs a chain-friendly model.** MobileNetV2-0.35's large tensors
  co-tile into line-buffered chains, so the slow tier stays below the naive peak.
  Full-width / big residual CNNs (ResNet-50, VGG-16) keep their large feature
  maps cross-stage, so they need a multi-MB slow tier (external memory) and give
  an *efficiency* story (small fast-SRAM footprint), not a feasibility barrier.
- **Kernel is `s8_ref`** (portable reference). No CMSIS-NN / Helium here.

## Layout

```
firmware/   main.c, startup.c, link.ld  - bare-metal Cortex-M55 firmware
tools/      host_probe.c                - host runner: fast/slow arena requirement of a plan
            build_backbone.py           - torchvision backbone -> int8 QDQ ONNX (truncated before the head)
build.sh    build firmware.elf for a given plan (+ arena config)
run.sh      run a plan on QEMU mps3-an547
```

## Reproduce

Prerequisites: `arm-none-eabi-gcc` (13.2+), `qemu-system-arm` (8.2+, has
`mps3-an547`), and the TiGrIS compiler + runtime (sibling repos).

### MobileNetV2-0.35 (SRAM-only barrier)

Uses the bench flagship plan directly:

```bash
PLAN=../deployability-hil/build/plans/mbv2_a35.tgrs
./build.sh "$PLAN" MobileNetV2-0.35 -DFAST_KB=192 -DSLOW_KB=256
./run.sh   "$PLAN"
# -> OUTPUT n=10 checksum=5740 ... total_sram=130192 bytes
```

### ResNet-50 (capability demo)

```bash
# 1. Build the int8 backbone and compile a plan (needs the tigris venv).
python3 tools/build_backbone.py resnet50 224 /tmp/r50
tigris compile /tmp/r50/resnet50_bb_224.onnx -m 512K -m 4M -o /tmp/r50/resnet50.tgrs
# 2. Fast arena in on-chip SRAM, slow tier in DDR (the backbone's intermediates are large).
./build.sh /tmp/r50/resnet50.tgrs ResNet-50 -DFAST_KB=640 -DSLOW_DDR
./run.sh   /tmp/r50/resnet50.tgrs
# -> OUTPUT n=100352 checksum=-1942099862 ... fast_peak=~508K
```

### Check a plan's memory profile on the host

`tools/host_probe.c` reports the fast/slow arena a plan actually needs (build it
against the runtime, then `./host_probe plan.tgrs <fast_bytes> <slow_bytes>`).
Useful for finding whether a model tiles to an SRAM-only total.
