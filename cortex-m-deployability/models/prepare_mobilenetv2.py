#!/usr/bin/env python3
"""Prepare the MobileNetV2 model for the Cortex-M deployability benchmark.

This is the model with the largest activation (112x112x48 =
588 KiB, naive co-resident peak 735 KiB) exceeds the SRAM of ALL THREE boards, so
TFLite Micro OOMs at AllocateTensors everywhere, while TiGrIS tiles it to fit
(down to 32 KiB if asked) and runs. It also exercises the executor's tiled-stage
fix end to end: inverted-residual Adds and stride-2 SAME asymmetric padding inside
tiled stages (verified bit-exact tiled-vs-non-tiled, see below).

Variant: MobileNetV2 alpha=0.35 @ 224x224, imagenet backbone + a fresh Dense(10)
head, NO softmax (compare logits, like DS-CNN). ~423 K params, ~413 KiB int8
weights -> fits H753 (2 MB) / RP2350 (4 MB XIP) flash; the ~605 KiB plan does NOT
fit F446's 512 KiB flash (the flash-barrier board). The imagenet backbone makes
the output non-degenerate (a random-weight deep net collapses to all-zeros through
the global pool, making parity checks meaningless).

  1. Build the model (imagenet backbone, random head).
  2. Convert to INT8 TFLite (the TFLM-native model) + emit its C header.
  3. Reconstruct a TiGrIS-runnable QDQ ONNX from that exact .tflite (the
     reconstructor is DAG-aware: residual Adds read two branches by tensor index).
  4. Compile per-board plans + a non-tiled reference plan.

Usage: python prepare_mobilenetv2.py   (downloads imagenet weights once, cached)
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = (HERE / "../../tflm-esp32s3/models").resolve()
OUT = MODELS / "output"
sys.path.insert(0, str(MODELS))
from prepare import convert_tflite_i8, generate_c_header, compile_tigris_plan  # noqa: E402

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
    onnx = OUT / "mbv2_a35_r224_matched.onnx"

    print("[1/4] Build MobileNetV2 (imagenet backbone + Dense head, no softmax)")
    model = build_keras_mobilenetv2()

    print("[2/4] Convert to INT8 TFLite + emit the TFLM C header")
    convert_tflite_i8(model, tflite)
    generate_c_header(tflite, OUT / "mbv2_tflite_i8.h", "mbv2_tflite_i8")

    print("[3/4] Reconstruct the TiGrIS QDQ ONNX (DAG: residual Adds by tensor index)")
    tool = (HERE / "../tools/tflite_to_qdq_onnx.py").resolve()
    subprocess.run([sys.executable, str(tool), str(tflite), str(onnx)], check=True)

    print("[4/4] Compile per-board plans + a non-tiled reference")
    # Budget == the board's usable SRAM, so the plan tiles to fit it. The non-tiled
    # 2M plan is the host correctness reference (tiled output must match it).
    compile_tigris_plan(onnx, OUT / "mbv2_a35_128k.tgrs", "128K")   # F446 SRAM (flash barrier)
    compile_tigris_plan(onnx, OUT / "mbv2_a35_480k.tgrs", "480K")   # H753 / RP2350 SRAM
    compile_tigris_plan(onnx, OUT / "mbv2_a35_2m.tgrs", "2M")       # non-tiled reference

    print("\nDone. TFLM (-DTFLM_MODEL=mbv2) OOMs at AllocateTensors on all 3 boards;")
    print("TiGrIS tiles it (-DTIGRIS_PLAN=.../mbv2_a35_480k.tgrs on H753/RP2350).")


if __name__ == "__main__":
    main()
