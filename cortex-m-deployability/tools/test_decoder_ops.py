#!/usr/bin/env python3
"""Acceptance test for the U-Net decoder op handlers in tflite_to_qdq_onnx.py.

Builds a tiny INT8 .tflite containing TRANSPOSE_CONV, CONCATENATION, and
RESIZE_NEAREST_NEIGHBOR, reconstructs it through the tool, and asserts:
  1. onnx.checker.check_model passes on the reconstructed graph, and
  2. onnxruntime on the reconstructed ONNX matches the tflite INT8 interpreter
     within 1 LSB (per-element int8 abs diff <= 1).

A second case asserts RESIZE_BILINEAR is rejected fail-loud: tflite's integer
bilinear kernel is not bit-exactly reproducible with a float QDQ graph, so the
tool must raise rather than emit a silently divergent model.

Run with the tigris venv python (tensorflow + onnx + onnxruntime + tflite):
    python test_decoder_ops.py
or under pytest:
    pytest cortex-m-deployability/tools/test_decoder_ops.py

Building the test fixture requires TensorFlow. Environments that run the fast
host-validation pytest pass without TensorFlow installed skip this module
(via pytest.importorskip below) instead of erroring.
"""
import os, sys, subprocess, tempfile
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import pytest

pytest.importorskip("tensorflow")

TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tflite_to_qdq_onnx.py")


def _build_int8_tflite(path, resize_method):
    import tensorflow as tf
    inp = tf.keras.Input(shape=(16, 16, 3))
    c1 = tf.keras.layers.Conv2D(6, 3, strides=1, padding="same", activation="relu")(inp)
    c2 = tf.keras.layers.Conv2D(6, 3, strides=2, padding="same", activation="relu")(c1)
    u1 = tf.keras.layers.Conv2DTranspose(
        6, 2, strides=2, padding="same",
        bias_initializer=tf.keras.initializers.RandomNormal(stddev=0.5, seed=1))(c2)
    cat = tf.keras.layers.Concatenate(axis=-1)([u1, c1])
    up = tf.keras.layers.Lambda(
        lambda t: tf.image.resize(t, [32, 32], method=resize_method))(cat)
    out = tf.keras.layers.Conv2D(3, 1, padding="same")(up)
    model = tf.keras.Model(inp, out)

    def rep():
        rng = np.random.default_rng(0)
        for _ in range(60):
            yield [rng.standard_normal((1, 16, 16, 3)).astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    open(path, "wb").write(conv.convert())


def _reconstruct(tfl_path, onnx_path):
    subprocess.run([sys.executable, TOOL, tfl_path, onnx_path], check=True)


def _run_and_compare(tfl_path, onnx_path):
    import tensorflow as tf
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(onnx_path))

    interp = tf.lite.Interpreter(model_path=tfl_path)
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    in_scale, in_zp = inp_d["quantization"]
    out_scale, out_zp = out_d["quantization"]

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    rng = np.random.default_rng(7)
    worst = 0
    for _ in range(8):
        x_i8 = rng.integers(-128, 128, size=inp_d["shape"], dtype=np.int8)

        interp.set_tensor(inp_d["index"], x_i8)
        interp.invoke()
        tfl_out = interp.get_tensor(out_d["index"]).astype(np.int32)  # NHWC int8

        x_f = (x_i8.astype(np.float32) - in_zp) * in_scale            # dequant
        x_nchw = np.transpose(x_f, (0, 3, 1, 2))                      # NHWC -> NCHW
        ort_out = sess.run(None, {in_name: x_nchw})[0]               # NCHW float
        ort_nhwc = np.transpose(ort_out, (0, 2, 3, 1))
        ort_i8 = np.clip(np.round(ort_nhwc / out_scale) + out_zp, -128, 127).astype(np.int32)

        worst = max(worst, int(np.abs(ort_i8 - tfl_out).max()))
    return worst


def test_transpose_conv_concat_resize_nearest():
    """TransposeConv + Concat + ResizeNearestNeighbor reconstruct within 1 LSB."""
    import tensorflow as tf
    with tempfile.TemporaryDirectory() as d:
        tfl = os.path.join(d, "m.tflite")
        onx = os.path.join(d, "m.onnx")
        _build_int8_tflite(tfl, tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        _reconstruct(tfl, onx)
        worst = _run_and_compare(tfl, onx)
        print(f"[nearest] worst int8 abs diff vs tflite = {worst}")
        assert worst <= 1, f"nearest reconstruction diverged by {worst} > 1 LSB"


def test_resize_bilinear_is_rejected():
    """RESIZE_BILINEAR is not bit-exactly reproducible, so the tool must fail loud."""
    import tensorflow as tf
    with tempfile.TemporaryDirectory() as d:
        tfl = os.path.join(d, "m.tflite")
        onx = os.path.join(d, "m.onnx")
        _build_int8_tflite(tfl, tf.image.ResizeMethod.BILINEAR)
        r = subprocess.run([sys.executable, TOOL, tfl, onx],
                           capture_output=True, text=True)
        assert r.returncode != 0, "expected the tool to reject RESIZE_BILINEAR"
        assert "RESIZE_BILINEAR" in r.stderr, r.stderr
        print("[bilinear] rejected fail-loud as expected")


if __name__ == "__main__":
    test_transpose_conv_concat_resize_nearest()
    test_resize_bilinear_is_rejected()
    print("PASS")
