#!/usr/bin/env python3
"""Prepare a Conv1D timeseries model for the Cortex-M deployability benchmark,
mirroring the DS-CNN / AD matched flow (Keras -> INT8 tflite -> reconstruct QDQ
ONNX -> .tgrs). Exercises the int8 Conv1D kernel + a timeseries use-case.

Usage: python prepare_ts.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = (HERE / "../../tflm-esp32s3/models").resolve()
OUT = MODELS / "output"
sys.path.insert(0, str(MODELS))
from prepare import convert_tflite_i8, generate_c_header  # noqa: E402


def build_keras_ts():
    """Small 1D-signal CNN, built 2D-native so it stays a clean CONV_2D op set
    (CMSIS-NN/TFLM have no Conv1D kernel - they run 1D conv as 2D with a singleton
    width, so this keeps the TiGrIS-vs-TFLM comparison apples-to-apples on
    cmsis_nn). The signal is H=64 timesteps x C=8 channels, width W=1; each conv
    is a (k,1) kernel along time. 3 strided blocks -> global-avg -> Dense."""
    import tensorflow as tf
    from tensorflow import keras

    np.random.seed(7)
    tf.random.set_seed(7)

    inp = keras.Input(shape=(64, 1, 8), name="input")  # H=time, W=1, C=channels
    x = inp
    for ch, k in [(16, 3), (32, 3), (32, 3)]:
        x = keras.layers.Conv2D(ch, (k, 1), strides=(2, 1), padding="same", use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    out = keras.layers.Dense(10, name="output")(x)
    return keras.Model(inputs=inp, outputs=out)


def dump_tflite_ops(path):
    from tflite.Model import Model
    from tflite.BuiltinOperator import BuiltinOperator
    buf = path.read_bytes()
    M = Model.GetRootAs(buf, 0)
    G = M.Subgraphs(0)
    BOP = {getattr(BuiltinOperator, k): k for k in dir(BuiltinOperator) if not k.startswith("_")}
    print(f"  {path.name}: {G.OperatorsLength()} ops")
    for oi in range(G.OperatorsLength()):
        op = G.Operators(oi)
        code = max(M.OperatorCodes(op.OpcodeIndex()).BuiltinCode(),
                   M.OperatorCodes(op.OpcodeIndex()).DeprecatedBuiltinCode())
        name = BOP[code]
        outs = [op.Outputs(j) for j in range(op.OutputsLength())]
        osh = [G.Tensors(outs[0]).Shape(k) for k in range(G.Tensors(outs[0]).ShapeLength())]
        wsh = "-"
        if op.InputsLength() > 1:
            wti = op.Inputs(1)
            wsh = [G.Tensors(wti).Shape(k) for k in range(G.Tensors(wti).ShapeLength())]
        print(f"    [{oi}] {name:18} out={osh}  filter={wsh}")


def main():
    import subprocess
    from prepare import compile_tigris_plan

    OUT.mkdir(parents=True, exist_ok=True)
    print("[1/5] Build Keras 1D-signal CNN (2D-native)")
    model = build_keras_ts()

    print("[2/5] Convert to INT8 TFLite (TFLM-native model)")
    convert_tflite_i8(model, OUT / "ts_i8.tflite")
    dump_tflite_ops(OUT / "ts_i8.tflite")

    print("[3/5] Generate the TFLM C header")
    generate_c_header(OUT / "ts_i8.tflite", OUT / "ts_tflite_i8.h", "ts_tflite_i8")

    print("[4/5] Reconstruct the TiGrIS QDQ ONNX from that .tflite (weight parity)")
    tool = (HERE / "../tools/tflite_to_qdq_onnx.py").resolve()
    subprocess.run([sys.executable, str(tool),
                    str(OUT / "ts_i8.tflite"), str(OUT / "ts_matched.onnx")], check=True)

    print("[5/5] Compile the TiGrIS plan @ 64K")
    compile_tigris_plan(OUT / "ts_matched.onnx", OUT / "ts_matched_64k.tgrs", "64K")
    print("\nDone. Build with -DTFLM_MODEL=ts / -DTIGRIS_PLAN=.../ts_matched_64k.tgrs")


if __name__ == "__main__":
    main()
