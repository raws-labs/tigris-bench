#!/usr/bin/env python3
"""Prepare the MLPerf Tiny Anomaly Detection (ToyADMOS) FC autoencoder for the
Cortex-M deployability benchmark, mirroring the DS-CNN matched flow:

  1. Build the reference dense-autoencoder architecture in Keras.
  2. Convert to INT8 TFLite (the TFLM-native model) + emit its C header.
  3. Reconstruct a TiGrIS-runnable QDQ ONNX from that exact .tflite (bit-exact
     weight parity), compile it to a .tgrs plan.
  4. Generate the golden reference output from the TFLite interpreter.

Deployability/parity benchmark: weights need not be trained-to-accuracy (we
measure latency/memory + that TiGrIS and TFLM agree bit-for-bit, as with DS-CNN).

Usage: python prepare_ad.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = (HERE / "../../tflm-esp32s3/models").resolve()
OUT = MODELS / "output"
sys.path.insert(0, str(MODELS))
from prepare import convert_tflite_i8, generate_c_header  # noqa: E402


def build_keras_ad():
    """MLPerf Tiny AD reference: 640 -> (128 x4) -> 8 -> (128 x4) -> 640 dense
    autoencoder, BatchNorm + ReLU after every hidden Dense."""
    import tensorflow as tf
    from tensorflow import keras

    np.random.seed(42)
    tf.random.set_seed(42)

    inp = keras.Input(shape=(640,), name="input")
    x = inp
    for _ in range(4):
        x = keras.layers.Dense(128)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
    x = keras.layers.Dense(8)(x)            # bottleneck
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    for _ in range(4):
        x = keras.layers.Dense(128)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
    out = keras.layers.Dense(640, name="output")(x)
    return keras.Model(inputs=inp, outputs=out)


def dump_tflite_ops(path: Path):
    """Print the op list of an INT8 tflite so we know what the reconstructor must
    handle (fused activation, bias, etc.)."""
    from tflite.Model import Model
    from tflite.BuiltinOperator import BuiltinOperator
    from tflite.ActivationFunctionType import ActivationFunctionType
    from tflite.FullyConnectedOptions import FullyConnectedOptions

    buf = path.read_bytes()
    M = Model.GetRootAs(buf, 0)
    G = M.Subgraphs(0)
    BOP = {getattr(BuiltinOperator, k): k for k in dir(BuiltinOperator) if not k.startswith("_")}
    AF = {getattr(ActivationFunctionType, k): k for k in dir(ActivationFunctionType) if not k.startswith("_")}
    print(f"  {path.name}: {G.OperatorsLength()} ops")
    for oi in range(G.OperatorsLength()):
        op = G.Operators(oi)
        code = max(M.OperatorCodes(op.OpcodeIndex()).BuiltinCode(),
                   M.OperatorCodes(op.OpcodeIndex()).DeprecatedBuiltinCode())
        name = BOP[code]
        fused = "?"
        nin = op.InputsLength()
        if name == "FULLY_CONNECTED" and op.BuiltinOptions() is not None:
            o = FullyConnectedOptions()
            o.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
            fused = AF.get(o.FusedActivationFunction(), "?")
        print(f"    [{oi}] {name:18} inputs={nin} fused_act={fused}")


def main():
    import subprocess
    from prepare import compile_tigris_plan

    OUT.mkdir(parents=True, exist_ok=True)
    print("[1/5] Build Keras AD autoencoder")
    model = build_keras_ad()

    print("[2/5] Convert to INT8 TFLite (TFLM-native model)")
    convert_tflite_i8(model, OUT / "ad_i8.tflite")
    dump_tflite_ops(OUT / "ad_i8.tflite")

    print("[3/5] Generate the TFLM C header (embedded by the tflm firmware)")
    generate_c_header(OUT / "ad_i8.tflite", OUT / "ad_tflite_i8.h", "ad_tflite_i8")

    print("[4/5] Reconstruct the TiGrIS QDQ ONNX from that .tflite (weight parity)")
    tool = (HERE / "../tools/tflite_to_qdq_onnx.py").resolve()
    subprocess.run([sys.executable, str(tool),
                    str(OUT / "ad_i8.tflite"), str(OUT / "ad_matched.onnx")], check=True)

    print("[5/5] Compile the TiGrIS plan @ 64K (H753 fast-arena budget)")
    compile_tigris_plan(OUT / "ad_matched.onnx", OUT / "ad_matched_64k.tgrs", "64K")
    print("\nDone. Build with: -DTFLM_MODEL=ad (TFLM) / -DTIGRIS_PLAN=.../ad_matched_64k.tgrs (TiGrIS)")


if __name__ == "__main__":
    main()
