#!/usr/bin/env bash
# Vendor the ARM + ST sources the cross-build needs, pinned for reproducibility.
#
#   - ARM-software/CMSIS-NN  v7.0.0  : the int8 kernels (standalone since v7,
#                                      no CMSIS-Core / CMSIS-DSP build dep)
#   - ARM-software/CMSIS_6   v6.3.0  : CMSIS-Core (core_cm7.h) for the BSP
#   - STMicroelectronics/cmsis-device-h7 : stm32h753xx.h, system file, startup
#   - STMicroelectronics/cmsis-device-f4 : stm32f446xx.h, system file, startup
#
# Run once before configuring CMake. Clones are git-ignored.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# cmsis-device-h7 publishes no release tags; pin a commit for reproducibility.
# Override with DEV_H7_REF=<sha> to bump it.
DEV_H7_REF="${DEV_H7_REF:-master}"

clone_tag() {  # url tag dir
    local url="$1" ref="$2" dir="$3"
    if [ -d "$HERE/$dir/.git" ]; then
        echo "[skip] $dir already present"
        return
    fi
    echo "[fetch] $dir @ $ref"
    git clone --depth 1 --branch "$ref" "$url" "$HERE/$dir"
}

# CMSIS-NN is pinned to the EXACT commit TFLite Micro bundles
# (tflite-micro .../ext_libs/cmsis_nn_download.sh ZIP_PREFIX_NN) so the
# TiGrIS-vs-TFLM benchmark links byte-identical kernels - the latency
# difference is then the framework, not the CMSIS-NN version.
CMSIS_NN_COMMIT="6d9d61d8a586c39160d0c1ba58f6948e4cf61ad0"
if [ ! -d "$HERE/CMSIS-NN/.git" ]; then
    echo "[fetch] CMSIS-NN @ ${CMSIS_NN_COMMIT} (matches TFLM's pin)"
    git clone https://github.com/ARM-software/CMSIS-NN.git "$HERE/CMSIS-NN"
    git -C "$HERE/CMSIS-NN" checkout "$CMSIS_NN_COMMIT"
    # A full-SHA checkout IS git's content-integrity guarantee (object hashes are
    # verified), so this is at least as strong as TFLM's MD5 check. Assert it
    # landed exactly, so a future edit to a mutable branch/tag fails loudly.
    got="$(git -C "$HERE/CMSIS-NN" rev-parse HEAD)"
    if [ "$got" != "$CMSIS_NN_COMMIT" ]; then
        echo "[error] CMSIS-NN HEAD $got != pinned $CMSIS_NN_COMMIT" >&2
        exit 1
    fi
    echo "[pin] CMSIS-NN at $(git -C "$HERE/CMSIS-NN" rev-parse --short HEAD)"
else
    echo "[skip] CMSIS-NN already present ($(git -C "$HERE/CMSIS-NN" rev-parse --short HEAD))"
fi
clone_tag https://github.com/ARM-software/CMSIS_6.git   v6.3.0 CMSIS_6

if [ ! -d "$HERE/cmsis-device-h7/.git" ]; then
    echo "[fetch] cmsis-device-h7 @ ${DEV_H7_REF}"
    git clone https://github.com/STMicroelectronics/cmsis-device-h7.git "$HERE/cmsis-device-h7"
    if [ "$DEV_H7_REF" != "master" ]; then
        git -C "$HERE/cmsis-device-h7" checkout "$DEV_H7_REF"
    fi
    echo "[pin] cmsis-device-h7 at $(git -C "$HERE/cmsis-device-h7" rev-parse --short HEAD)"
else
    echo "[skip] cmsis-device-h7 already present"
fi

# cmsis-device-f4 (NUCLEO-F446RE). Pin a commit for reproducibility.
DEV_F4_REF="${DEV_F4_REF:-3c77349ce04c8af401454cc51f85ea9a50e34fc1}"
if [ ! -d "$HERE/cmsis-device-f4/.git" ]; then
    echo "[fetch] cmsis-device-f4 @ ${DEV_F4_REF}"
    git clone https://github.com/STMicroelectronics/cmsis-device-f4.git "$HERE/cmsis-device-f4"
    git -C "$HERE/cmsis-device-f4" checkout "$DEV_F4_REF"
    echo "[pin] cmsis-device-f4 at $(git -C "$HERE/cmsis-device-f4" rev-parse --short HEAD)"
else
    echo "[skip] cmsis-device-f4 already present"
fi

echo "Done. Vendored under $HERE/"
