#!/usr/bin/env python3
"""Build DS-CNN model, quantize, compile TiGrIS plans, convert TFLite.

Outputs go to models/output/:
  ds_cnn.onnx              f32 ONNX model
  ds_cnn_i8.onnx           int8 quantized ONNX model
  ds_cnn.tgrs              TiGrIS plan (f32)
  ds_cnn_i8.tgrs           TiGrIS plan (int8)
  ds_cnn.tflite            TFLite f32 model
  ds_cnn_i8.tflite         TFLite int8 model
  ds_cnn_reference_f32.bin ORT reference output (f32, np.ones input)
  ds_cnn_reference_i8.bin  ORT reference output (i8, np.ones input)
  ds_cnn_tflite.h          C array for TFLM (f32)
  ds_cnn_tflite_i8.h       C array for TFLM (int8)
"""

from __future__ import annotations

import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build_ds_cnn(channels: int = 64) -> onnx.ModelProto:
    """MLPerf Tiny DS-CNN: depthwise-separable CNN for keyword spotting.

    Conv(10x4,s2)->BN->ReLU -> 4x[DWConv(3x3)->BN->ReLU->Conv(1x1)->BN->ReLU]
    -> GlobalAvgPool -> Flatten -> FC(12).
    Input: [1, 1, 49, 10] (NCHW, mel spectrogram).
    Output: [1, 12] (12 keyword classes).
    """
    np.random.seed(42)

    def _conv_bn_relu(prefix, x_name, out_name, c_in, c_out, kh, kw, sh, sw, group=1):
        nodes, inits = [], []
        conv_out = f"{prefix}_conv_out"
        w = np.random.randn(c_out, c_in // group, kh, kw).astype(np.float32) * 0.1
        b = np.zeros(c_out, dtype=np.float32)
        inits.append(numpy_helper.from_array(w, f"{prefix}_w"))
        inits.append(numpy_helper.from_array(b, f"{prefix}_b"))
        pad_h, pad_w = max(0, kh - 1), max(0, kw - 1)
        pt, pb = pad_h // 2, pad_h - pad_h // 2
        pl, pr = pad_w // 2, pad_w - pad_w // 2
        nodes.append(helper.make_node(
            "Conv", [x_name, f"{prefix}_w", f"{prefix}_b"], [conv_out],
            name=f"{prefix}_conv", kernel_shape=[kh, kw], strides=[sh, sw],
            pads=[pt, pl, pb, pr], group=group))
        bn_out = f"{prefix}_bn_out"
        for suffix, data in [("scale", np.ones(c_out)), ("bias", np.zeros(c_out)),
                             ("mean", np.zeros(c_out)), ("var", np.ones(c_out))]:
            inits.append(numpy_helper.from_array(data.astype(np.float32), f"{prefix}_{suffix}"))
        nodes.append(helper.make_node(
            "BatchNormalization",
            [conv_out, f"{prefix}_scale", f"{prefix}_bias", f"{prefix}_mean", f"{prefix}_var"],
            [bn_out], name=f"{prefix}_bn"))
        nodes.append(helper.make_node("Relu", [bn_out], [out_name], name=f"{prefix}_relu"))
        return nodes, inits

    all_nodes, all_inits = [], []
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 49, 10])

    n, i = _conv_bn_relu("block0", "input", "b0_out", 1, channels, 10, 4, 2, 2)
    all_nodes.extend(n); all_inits.extend(i)

    prev = "b0_out"
    for idx in range(1, 5):
        n, i = _conv_bn_relu(f"dw{idx}", prev, f"dw{idx}_out", channels, channels, 3, 3, 1, 1, group=channels)
        all_nodes.extend(n); all_inits.extend(i)
        n, i = _conv_bn_relu(f"pw{idx}", f"dw{idx}_out", f"pw{idx}_out", channels, channels, 1, 1, 1, 1)
        all_nodes.extend(n); all_inits.extend(i)
        prev = f"pw{idx}_out"

    all_nodes.append(helper.make_node("GlobalAveragePool", [prev], ["pool_out"], name="gap"))
    all_nodes.append(helper.make_node("Flatten", ["pool_out"], ["flat_out"], name="flatten", axis=1))
    fc_w = np.random.randn(12, channels).astype(np.float32) * 0.1
    fc_b = np.zeros(12, dtype=np.float32)
    all_inits.append(numpy_helper.from_array(fc_w, "fc_w"))
    all_inits.append(numpy_helper.from_array(fc_b, "fc_b"))
    all_nodes.append(helper.make_node("Gemm", ["flat_out", "fc_w", "fc_b"], ["output"],
                                      name="fc", transB=1))
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 12])

    graph = helper.make_graph(all_nodes, "ds_cnn", [X], [Y], initializer=all_inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def build_mobilenet_v1(alpha: float = 1.0, input_size: int = 128,
                       num_classes: int = 10) -> onnx.ModelProto:
    """MobileNetV1: standard depthwise-separable CNN for vision.

    Architecture follows the original paper (Howard et al., 2017).
    Input: [1, 3, input_size, input_size] (NCHW, RGB image).
    Output: [1, num_classes].
    """
    np.random.seed(42)

    def _make_ch(c):
        return max(8, int(c * alpha + 0.5))

    def _conv_bn_relu(prefix, x_name, out_name, c_in, c_out, kh, kw, sh, sw, group=1):
        nodes, inits = [], []
        conv_out = f"{prefix}_conv_out"
        w = np.random.randn(c_out, c_in // group, kh, kw).astype(np.float32) * 0.02
        b = np.zeros(c_out, dtype=np.float32)
        inits.append(numpy_helper.from_array(w, f"{prefix}_w"))
        inits.append(numpy_helper.from_array(b, f"{prefix}_b"))
        pad_h, pad_w = max(0, kh - 1), max(0, kw - 1)
        pt, pb = pad_h // 2, pad_h - pad_h // 2
        pl, pr = pad_w // 2, pad_w - pad_w // 2
        nodes.append(helper.make_node(
            "Conv", [x_name, f"{prefix}_w", f"{prefix}_b"], [conv_out],
            name=f"{prefix}_conv", kernel_shape=[kh, kw], strides=[sh, sw],
            pads=[pt, pl, pb, pr], group=group))
        bn_out = f"{prefix}_bn_out"
        for suffix, data in [("scale", np.ones(c_out)), ("bias", np.zeros(c_out)),
                             ("mean", np.zeros(c_out)), ("var", np.ones(c_out))]:
            inits.append(numpy_helper.from_array(data.astype(np.float32), f"{prefix}_{suffix}"))
        nodes.append(helper.make_node(
            "BatchNormalization",
            [conv_out, f"{prefix}_scale", f"{prefix}_bias", f"{prefix}_mean", f"{prefix}_var"],
            [bn_out], name=f"{prefix}_bn"))
        nodes.append(helper.make_node("Relu", [bn_out], [out_name], name=f"{prefix}_relu"))
        return nodes, inits

    # MobileNetV1 block definitions: (dw_stride, pw_out_channels)
    blocks = [
        (1, 64), (2, 128), (1, 128), (2, 256), (1, 256), (2, 512),
        (1, 512), (1, 512), (1, 512), (1, 512), (1, 512),
        (2, 1024), (1, 1024),
    ]

    all_nodes, all_inits = [], []
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT,
                                      [1, 3, input_size, input_size])

    # Initial conv: 3x3 s2
    ch0 = _make_ch(32)
    n, i = _conv_bn_relu("conv0", "input", "conv0_out", 3, ch0, 3, 3, 2, 2)
    all_nodes.extend(n); all_inits.extend(i)

    prev = "conv0_out"
    prev_ch = ch0
    for idx, (stride, out_ch) in enumerate(blocks):
        out_ch = _make_ch(out_ch)
        tag = f"b{idx+1}"
        # Depthwise conv 3x3
        n, i = _conv_bn_relu(f"{tag}_dw", prev, f"{tag}_dw_out",
                              prev_ch, prev_ch, 3, 3, stride, stride, group=prev_ch)
        all_nodes.extend(n); all_inits.extend(i)
        # Pointwise conv 1x1
        n, i = _conv_bn_relu(f"{tag}_pw", f"{tag}_dw_out", f"{tag}_pw_out",
                              prev_ch, out_ch, 1, 1, 1, 1)
        all_nodes.extend(n); all_inits.extend(i)
        prev = f"{tag}_pw_out"
        prev_ch = out_ch

    # Global average pool -> flatten -> FC
    all_nodes.append(helper.make_node("GlobalAveragePool", [prev], ["pool_out"], name="gap"))
    all_nodes.append(helper.make_node("Flatten", ["pool_out"], ["flat_out"],
                                      name="flatten", axis=1))
    fc_w = np.random.randn(num_classes, prev_ch).astype(np.float32) * 0.02
    fc_b = np.zeros(num_classes, dtype=np.float32)
    all_inits.append(numpy_helper.from_array(fc_w, "fc_w"))
    all_inits.append(numpy_helper.from_array(fc_b, "fc_b"))
    all_nodes.append(helper.make_node("Gemm", ["flat_out", "fc_w", "fc_b"], ["output"],
                                      name="fc", transB=1))
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, num_classes])

    graph = helper.make_graph(all_nodes, "mobilenet_v1", [X], [Y], initializer=all_inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def build_keras_mobilenet_v1(alpha: float = 1.0, input_size: int = 128,
                             num_classes: int = 10):
    """Build MobileNetV1 in Keras for TFLite conversion."""
    import tensorflow as tf
    return tf.keras.applications.MobileNet(
        input_shape=(input_size, input_size, 3),
        alpha=alpha,
        include_top=True,
        weights=None,
        classes=num_classes,
    )


def generate_reference(onnx_path: Path, output_path: Path):
    """Run ONNX model and save reference output bytes.

    For quantized (i8) models: fills input with int8 value 1 (dequantized
    via the model's input QuantizeLinear params), saves raw int8 output.
    For float models: fills input with 1.0f, saves raw float32 output.
    """
    import onnx as _onnx
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path))
    inp_meta = sess.get_inputs()[0]
    shape = [1 if isinstance(d, str) else d for d in inp_meta.shape]

    # Detect quantized model: look for input QuantizeLinear node
    model = _onnx.load(str(onnx_path))
    is_quantized = False
    out_scale = None
    out_zp = 0

    # Build initializer lookup
    inits = {init.name: init for init in model.graph.initializer}

    for node in model.graph.node:
        # Find input QuantizeLinear (first QuantizeLinear whose input is the model input)
        if node.op_type == "QuantizeLinear" and node.input[0] == inp_meta.name:
            is_quantized = True
            # Get input scale/zp to dequantize our int8 value back to float
            in_scale_init = inits.get(node.input[1])
            in_zp_init = inits.get(node.input[2]) if len(node.input) > 2 else None
            in_scale = (list(in_scale_init.float_data) or
                        list(np.frombuffer(in_scale_init.raw_data, dtype=np.float32)))[0]
            in_zp = 0
            if in_zp_init is not None:
                zp_vals = list(in_zp_init.int32_data) or list(np.frombuffer(in_zp_init.raw_data, dtype=np.int8))
                if zp_vals:
                    in_zp = int(zp_vals[0])
            # int8 value 1 to float: (1 - zp) * scale
            float_val = float((1 - in_zp) * in_scale)
            break

    if is_quantized:
        # Find output DequantizeLinear (last one feeding the model output)
        model_out_name = model.graph.output[0].name
        for node in reversed(model.graph.node):
            if node.op_type == "DequantizeLinear" and node.output[0] == model_out_name:
                os_init = inits.get(node.input[1])
                oz_init = inits.get(node.input[2]) if len(node.input) > 2 else None
                out_scale = (list(os_init.float_data) or
                             list(np.frombuffer(os_init.raw_data, dtype=np.float32)))[0]
                if oz_init is not None:
                    zp_vals = list(oz_init.int32_data) or list(np.frombuffer(oz_init.raw_data, dtype=np.int8))
                    if zp_vals:
                        out_zp = int(zp_vals[0])
                break

    if is_quantized and out_scale is not None:
        inp_data = np.full(shape, float_val, dtype=np.float32)
        results = sess.run(None, {inp_meta.name: inp_data})
        float_out = results[0].flatten().astype(np.float64)
        int8_out = np.clip(np.round(float_out / out_scale) + out_zp, -128, 127).astype(np.int8)
        ref_data = int8_out.tobytes()
        output_path.write_bytes(ref_data)
        print(f"  Reference: {output_path} ({len(ref_data)} bytes, int8, "
              f"input=1 (float {float_val:.4f}), first 10: {list(int8_out[:10])})")
    else:
        inp_data = np.ones(shape, dtype=np.float32)
        results = sess.run(None, {inp_meta.name: inp_data})
        ref_data = b"".join(r.astype(np.float32).tobytes() for r in results)
        output_path.write_bytes(ref_data)
        print(f"  Reference: {output_path} ({len(ref_data)} bytes, {len(ref_data)//4} floats)")


class _CalibrationDataReader:
    """Feeds calibration data for onnxruntime quantization."""

    def __init__(self, input_name: str, shape: list[int], n_samples: int = 100):
        self.input_name = input_name
        self.shape = shape
        self.n_samples = n_samples
        self._idx = 0
        np.random.seed(42)

    def get_next(self):
        if self._idx >= self.n_samples:
            return None
        self._idx += 1
        return {self.input_name: np.random.randn(*self.shape).astype(np.float32)}


def quantize_onnx(f32_path: Path, i8_path: Path):
    """Quantize f32 ONNX model to int8 using static quantization."""
    from onnxruntime.quantization import quantize_static, CalibrationMethod, QuantType

    model = onnx.load(str(f32_path))
    inp = model.graph.input[0]
    shape = []
    for dim in inp.type.tensor_type.shape.dim:
        shape.append(dim.dim_value if dim.dim_value > 0 else 1)

    reader = _CalibrationDataReader(inp.name, shape, n_samples=100)

    quantize_static(
        str(f32_path),
        str(i8_path),
        reader,
        quant_format=None,
        calibrate_method=CalibrationMethod.MinMax,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )
    print(f"  Quantized: {i8_path} ({i8_path.stat().st_size} bytes)")


def compile_tigris_plan(onnx_path: Path, plan_path: Path, budget: str = "256K"):
    """Compile ONNX model to TiGrIS binary plan via the public CLI."""
    import subprocess

    plan_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["tigris", "compile", str(onnx_path), "-m", budget, "-o", str(plan_path)],
        check=True,
    )
    print(f"  TiGrIS plan: {plan_path} ({plan_path.stat().st_size} bytes)")


def build_keras_ds_cnn(channels: int = 64):
    """Build the same DS-CNN architecture in Keras for TFLite conversion."""
    import tensorflow as tf
    from tensorflow import keras

    np.random.seed(42)
    tf.random.set_seed(42)

    inp = keras.Input(shape=(49, 10, 1), name="input")  # NHWC

    # Block 0: Conv(10x4, s2), BN, ReLU
    x = keras.layers.Conv2D(channels, (10, 4), strides=(2, 2), padding="same", use_bias=False)(inp)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    # 4x depthwise separable blocks
    for _ in range(4):
        x = keras.layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    out = keras.layers.Dense(12, name="output")(x)

    return keras.Model(inputs=inp, outputs=out)


def convert_tflite_f32(output_path: Path):
    """Convert Keras DS-CNN to f32 TFLite."""
    import tensorflow as tf

    model = build_keras_ds_cnn()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    print(f"  TFLite f32: {output_path} ({len(tflite_model)} bytes)")
    return model


def convert_tflite_i8(keras_model, output_path: Path):
    """Convert Keras DS-CNN to int8 TFLite with full integer quantization."""
    import tensorflow as tf

    input_shape = keras_model.input_shape  # e.g. (None, 49, 10, 1)
    calib_shape = [1] + list(input_shape[1:])

    def representative_dataset():
        np.random.seed(42)
        for _ in range(100):
            yield [np.random.randn(*calib_shape).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    print(f"  TFLite i8: {output_path} ({len(tflite_model)} bytes)")


def generate_c_header(tflite_path: Path, header_path: Path, array_name: str):
    """Generate xxd -i style C header from TFLite model."""
    data = tflite_path.read_bytes()
    lines = [f"/* Auto-generated from {tflite_path.name}. Do not edit. */",
             f"const unsigned char {array_name}[] = {{"]
    for i in range(0, len(data), 12):
        chunk = data[i:i+12]
        hex_vals = ", ".join(f"0x{b:02x}" for b in chunk)
        lines.append(f"  {hex_vals},")
    lines.append("};")
    lines.append(f"const unsigned int {array_name}_len = {len(data)};")
    lines.append("")
    header_path.write_text("\n".join(lines))
    print(f"  C header: {header_path} ({len(data)} bytes model)")


def main():
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)

    print("DS-CNN Model Preparation\n")

    # 1. Build f32 ONNX
    print("[1/12] Building DS-CNN ONNX (f32)...")
    onnx_model = build_ds_cnn()
    onnx_f32 = out / "ds_cnn.onnx"
    onnx.save(onnx_model, str(onnx_f32))
    print(f"  Saved: {onnx_f32} ({onnx_f32.stat().st_size} bytes)")

    # 2. Generate f32 reference output
    print("\n[2/12] Generating ORT reference (f32, np.ones input)...")
    generate_reference(onnx_f32, out / "ds_cnn_reference_f32.bin")

    # 3. Quantize to int8
    print("\n[3/12] Quantizing ONNX to int8...")
    onnx_i8 = out / "ds_cnn_i8.onnx"
    quantize_onnx(onnx_f32, onnx_i8)

    # 4. Generate i8 reference output
    print("\n[4/12] Generating ORT reference (i8)...")
    generate_reference(onnx_i8, out / "ds_cnn_reference_i8.bin")

    # 5. Compile TiGrIS plans
    print("\n[5/12] Compiling TiGrIS plan (f32)...")
    compile_tigris_plan(onnx_f32, out / "ds_cnn.tgrs", "256K")

    print("\n[6/12] Compiling TiGrIS plan (i8)...")
    compile_tigris_plan(onnx_i8, out / "ds_cnn_i8.tgrs", "256K")

    # 7. Convert TFLite
    try:
        import tensorflow  # noqa: F401
        has_tf = True
    except ImportError:
        has_tf = False

    if has_tf:
        print("\n[7/12] Converting to TFLite...")
        keras_model = convert_tflite_f32(out / "ds_cnn.tflite")
        convert_tflite_i8(keras_model, out / "ds_cnn_i8.tflite")

        # 8. Generate C headers for TFLM (into tflm-esp/main/ for the build)
        print("\n[8/12] Generating C headers for TFLM...")
        tflm_main = Path(__file__).parent.parent / "tflm-esp" / "main"
        generate_c_header(out / "ds_cnn.tflite", tflm_main / "ds_cnn_tflite.h", "ds_cnn_tflite")
        generate_c_header(out / "ds_cnn_i8.tflite", tflm_main / "ds_cnn_tflite_i8.h", "ds_cnn_tflite_i8")
    else:
        print("\n[7/12] Skipping TFLite conversion (tensorflow not installed)")
        print("[8/12] Skipping C header generation (no .tflite files)")

    # MobileNetV1 (alpha=1.0, 128x128 input)
    print("\n[9/12] Building MobileNetV1 ONNX (f32)...")
    mbv1_model = build_mobilenet_v1()
    mbv1_f32 = out / "mobilenet_v1.onnx"
    onnx.save(mbv1_model, str(mbv1_f32))
    print(f"  Saved: {mbv1_f32} ({mbv1_f32.stat().st_size} bytes)")

    print("\n[10/12] Quantizing MobileNetV1...")
    mbv1_i8 = out / "mobilenet_v1_i8.onnx"
    quantize_onnx(mbv1_f32, mbv1_i8)

    print("\n[11/12] Compiling TiGrIS plan + reference (MobileNetV1)...")
    generate_reference(mbv1_i8, out / "mobilenet_v1_reference_i8.bin")
    compile_tigris_plan(mbv1_i8, out / "mobilenet_v1_i8.tgrs", "256K")

    # TFLite + C header for MobileNetV1 (if TF available)
    if has_tf:
        keras_mbv1 = build_keras_mobilenet_v1()
        convert_tflite_i8(keras_mbv1, out / "mobilenet_v1_i8.tflite")
        tflm_main = Path(__file__).parent.parent / "tflm-esp" / "main"
        generate_c_header(out / "mobilenet_v1_i8.tflite",
                          tflm_main / "mobilenet_v1_tflite_i8.h", "mobilenet_v1_tflite_i8")

    # Tiling sweep (DS-CNN i8, varied budgets)
    print("\n[12/13] Compiling tiling sweep plans (DS-CNN)...")
    for budget in ["128K", "64K", "32K"]:
        plan_name = f"ds_cnn_i8_{budget.lower()}.tgrs"
        compile_tigris_plan(onnx_i8, out / plan_name, budget)

    # Tiling overhead sweep (MobileNetV1 i8, varied budgets)
    print("\n[13/13] Compiling tiling sweep plans (MobileNetV1)...")
    for budget in ["128K", "64K", "32K"]:
        plan_name = f"mobilenet_v1_i8_{budget.lower()}.tgrs"
        compile_tigris_plan(mbv1_i8, out / plan_name, budget)

    print(f"\nDone. All outputs in {out}/")


if __name__ == "__main__":
    main()
