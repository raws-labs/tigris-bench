#!/bin/bash
# Build the bare-metal Cortex-M55 firmware for a given .tgrs plan.
#
# Usage: ./build.sh <plan.tgrs> <MODEL_NAME> [extra -D flags...]
#   e.g. ./build.sh mbv2_a35.tgrs MobileNetV2-0.35 -DFAST_KB=192 -DSLOW_KB=256
#        ./build.sh resnet50.tgrs ResNet-50 -DFAST_KB=640 -DSLOW_DDR
#
# Requires arm-none-eabi-gcc (13.2+). Produces firmware.elf next to this script.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
RT="${TIGRIS_RUNTIME:-$HERE/../../../tigris-runtime}"   # sibling repo by default
PLAN="$1"; NAME="$2"; shift 2 || { echo "usage: build.sh <plan.tgrs> <MODEL_NAME> [-D...]"; exit 2; }
[ -f "$PLAN" ] || { echo "plan not found: $PLAN"; exit 2; }
[ -d "$RT/include" ] || { echo "runtime not found at $RT (set TIGRIS_RUNTIME)"; exit 2; }
PLAN_LEN=$(stat -c%s "$PLAN")
echo "plan: $PLAN  ($PLAN_LEN bytes)   model: $NAME"

SRCS=(
  "$RT/src/tigris_executor.c" "$RT/src/tigris_executor_compat.c"
  "$RT/src/tigris_kernels.c"  "$RT/src/tigris_kernels_s8.c"
  "$RT/src/tigris_loader.c"   "$RT/src/tigris_lz4.c" "$RT/src/tigris_mem.c"
)
arm-none-eabi-gcc \
  -mcpu=cortex-m55+nomve -mthumb -mfloat-abi=hard \
  -O2 -ffunction-sections -fdata-sections -ffreestanding \
  -I"$RT/include" \
  -DPLAN_LEN="${PLAN_LEN}u" -DMODEL_NAME="\"$NAME\"" "$@" \
  "$HERE/firmware/main.c" "$HERE/firmware/startup.c" "${SRCS[@]}" \
  -T "$HERE/firmware/link.ld" -nostartfiles -Wl,--gc-sections \
  -specs=nano.specs -specs=nosys.specs -lm \
  -o "$HERE/firmware.elf"
arm-none-eabi-size "$HERE/firmware.elf"
