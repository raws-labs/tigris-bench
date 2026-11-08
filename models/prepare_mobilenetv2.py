#!/usr/bin/env python3
"""Prepare the MobileNetV2 model for the Cortex-M deployability benchmark.

Its largest activation (112x112x48 int8, 588 KiB) is larger than the largest
contiguous SRAM of every benchmark board (F446 128 KiB, H753 512 KiB AXI SRAM,
RP2350 520 KiB), and TFLite Micro plans at least 735 KiB of activations for it,
because two of its tensors are live at once. TFLite Micro fails AllocateTensors
on the H753 with a 480 KiB arena; TiGrIS tiles the model and runs it on the H753
and the RP2350. It also exercises inverted-residual Adds and stride-2 SAME
asymmetric padding inside tiled stages.

Variant: MobileNetV2 alpha=0.35 @ 224x224, the Keras ImageNet backbone weights +
a fresh Dense(10) head (tf seed 42, never trained), NO softmax (compare logits,
like DS-CNN). int8 calibration uses 100 random-normal samples (convert_tflite_i8
in prepare.py). 401,888 parameters, 413 KiB of int8 weights and int32 biases:
the firmware fits the H753 (2 MiB) and RP2350 (4 MiB XIP) flash, not the F446's
512 KiB. The ImageNet backbone keeps the int8 output non-degenerate (a
random-weight deep net collapses to all-zeros through the global pool, making
parity checks meaningless); the untrained head gives the logits no class meaning.

  1. Build the model (ImageNet backbone, random head).
  2. Convert to INT8 TFLite (the TFLM-native model) + emit its C header.

The Cortex-M run compiles its TiGrIS plan from that .tflite
(cortex-m/deployability-hil/scripts/prepare_tigris_plans.py).

Usage: python prepare_mobilenetv2.py   (downloads imagenet weights once, cached)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE
OUT = MODELS / "output"
sys.path.insert(0, str(MODELS))
from prepare import convert_tflite_i8, generate_c_header  # noqa: E402

ALPHA, RES, CLASSES = 0.35, 224, 10


def build_keras_mobilenetv2():
    """imagenet-pretrained MobileNetV2 backbone + GlobalAveragePool + Dense head,
    no softmax. Real backbone weights keep the int8 output non-degenerate."""
    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(42)   # reproducible Dense-head init
    base = keras.applications.MobileNetV2(
        input_shape=(RES, RES, 3), alpha=ALPHA, weights="imagenet", include_top=False)
    x = keras.layers.GlobalAveragePooling2D()(base.output)
    out = keras.layers.Dense(CLASSES, name="head")(x)
    return keras.Model(base.input, out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tflite = OUT / "mbv2_a35_r224_i8.tflite"

    print("[1/2] Build MobileNetV2 (imagenet backbone + Dense head, no softmax)")
    model = build_keras_mobilenetv2()

    print("[2/2] Convert to INT8 TFLite + emit the TFLM C header")
    convert_tflite_i8(model, tflite)
    generate_c_header(tflite, OUT / "mbv2_tflite_i8.h", "mbv2_tflite_i8")

    print("\nDone. The Cortex-M run compiles its TiGrIS plan from the .tflite;")
    print("TFLM (-DTFLM_MODEL=mbv2) fails AllocateTensors on the H753 at 480 KiB.")


if __name__ == "__main__":
    main()
