#!/usr/bin/env python3
"""Reconstruct a TiGrIS-compatible QDQ ONNX from an INT8 .tflite (DS-CNN op set).

Emits the canonical ORT-style QDQ pattern TiGrIS folds (per-channel weight
DequantizeLinear, int32 bias DQ, Q/DQ around activations, explicit fused ReLU)
in NCHW layout, populated with the tflite's EXACT int8 weights + scales. This
lets TiGrIS run the identical quantized model TFLite Micro runs, for a true
weight-for-weight parity comparison.

Verified faithful: ORT on the output matches the tflite interpreter bit-for-bit.
Handles CONV_2D, DEPTHWISE_CONV_2D, MEAN (global-avg), FULLY_CONNECTED.

Usage: python tflite_to_qdq_onnx.py model_i8.tflite out.onnx
Needs: tflite, onnx, numpy (no TensorFlow).
"""
import sys, numpy as np, onnx
from onnx import helper, numpy_helper, TensorProto
from tflite.Model import Model
from tflite.BuiltinOperator import BuiltinOperator
from tflite.TensorType import TensorType
from tflite.Conv2DOptions import Conv2DOptions
from tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
from tflite.FullyConnectedOptions import FullyConnectedOptions
from tflite.AddOptions import AddOptions
from tflite.ActivationFunctionType import ActivationFunctionType

def conv_opts(op, dw=False):
    o = (DepthwiseConv2DOptions if dw else Conv2DOptions)()
    o.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
    return o.StrideH(), o.StrideW(), o.FusedActivationFunction()

def fc_fused(op):
    o = FullyConnectedOptions()
    o.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
    return o.FusedActivationFunction()

def add_fused(op):
    if op.BuiltinOptions() is None:
        return ActivationFunctionType.NONE
    o = AddOptions()
    o.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
    return o.FusedActivationFunction()

tfl_path, onnx_path = sys.argv[1], sys.argv[2]
buf = open(tfl_path, 'rb').read()
M = Model.GetRootAs(buf, 0)
G = M.Subgraphs(0)
BOP = {getattr(BuiltinOperator, k): k for k in dir(BuiltinOperator) if not k.startswith('_')}

def tensor(ti): return G.Tensors(ti)
def shape(ti): return [tensor(ti).Shape(k) for k in range(tensor(ti).ShapeLength())]
def data(ti):
    t = tensor(ti); b = M.Buffers(t.Buffer()).DataAsNumpy()
    dt = {TensorType.INT8: np.int8, TensorType.INT32: np.int32,
          TensorType.FLOAT32: np.float32}[t.Type()]
    return np.frombuffer(b.tobytes(), dtype=dt).reshape(shape(ti))
def qparams(ti):
    q = tensor(ti).Quantization()
    s = q.ScaleAsNumpy(); z = q.ZeroPointAsNumpy()
    s = np.atleast_1d(np.asarray(s, dtype=np.float32))
    z = np.atleast_1d(np.asarray(z, dtype=np.int64))
    return s, z

nodes, inits = [], []
def add_init(arr, name):
    inits.append(numpy_helper.from_array(arr, name)); return name

def dq(name_q, arr_q, scale, zp, axis=None, zp_dtype=np.int8):
    """int8/int32 initializer + DequantizeLinear -> float tensor `name_q`_dq."""
    add_init(arr_q, name_q)
    add_init(scale.astype(np.float32) if scale.size > 1 else np.float32(scale[0]), name_q + "_s")
    add_init(zp.astype(zp_dtype) if zp.size > 1 else np.array(zp[0], dtype=zp_dtype), name_q + "_z")
    out = name_q + "_dq"
    kw = {"axis": axis} if axis is not None else {}
    nodes.append(helper.make_node("DequantizeLinear", [name_q, name_q + "_s", name_q + "_z"], [out], **kw))
    return out

def act_qdq(src, ti, tag):
    """Wrap activation `src` (float) in Q/DQ using tensor ti's scale/zp."""
    s, z = qparams(ti); s = np.float32(s[0]); z = np.array(z[0], dtype=np.int8)
    add_init(s, tag + "_as"); add_init(z, tag + "_az")
    q, d = tag + "_q", tag + "_dq"
    nodes.append(helper.make_node("QuantizeLinear", [src, tag + "_as", tag + "_az"], [q]))
    nodes.append(helper.make_node("DequantizeLinear", [q, tag + "_as", tag + "_az"], [d]))
    return d

def fused_act(src, fused, tag):
    """Emit the tflite-fused activation (ReLU/ReLU6) explicitly, before requant."""
    if fused == ActivationFunctionType.RELU:
        out = tag + "_relu"
        nodes.append(helper.make_node("Relu", [src], [out]))
        return out
    if fused == ActivationFunctionType.RELU6:
        out = tag + "_relu6"
        add_init(np.float32(0.0), tag + "_lo"); add_init(np.float32(6.0), tag + "_hi")
        nodes.append(helper.make_node("Clip", [src, tag + "_lo", tag + "_hi"], [out]))
        return out
    return src

# ---- walk the tflite ops as a DAG ----
# tmap: tflite tensor index -> the ONNX float tensor name carrying its (dequantized)
# value. This handles branches/residuals (an op reads its inputs by index, e.g. an
# Add whose second input is the block input from several ops back), not just a
# linear chain. Every op registers tmap[out_ti] so later consumers can find it.
input_ti = G.Inputs(0)
_in_sh = shape(input_ti)
if len(_in_sh) == 4:       # conv: tflite NHWC -> ONNX NCHW
    in_vi_shape = [1, _in_sh[3], _in_sh[1], _in_sh[2]]
else:                      # FC / dense: shape as-is (e.g. [1, 640])
    in_vi_shape = list(_in_sh)
X = helper.make_tensor_value_info("input", TensorProto.FLOAT, in_vi_shape)
tmap = {input_ti: act_qdq("input", input_ti, "input")}  # input float -> Q/DQ

for oi in range(G.OperatorsLength()):
    op = G.Operators(oi)
    code = max(M.OperatorCodes(op.OpcodeIndex()).BuiltinCode(),
               M.OperatorCodes(op.OpcodeIndex()).DeprecatedBuiltinCode())
    name = BOP[code]
    ins = [op.Inputs(j) for j in range(op.InputsLength())]
    out_ti = op.Outputs(0)
    tag = f"l{oi}"

    if name in ("CONV_2D", "DEPTHWISE_CONV_2D"):
        x = tmap[ins[0]]
        w_ti, b_ti = ins[1], ins[2]
        w = data(w_ti); ws, wz = qparams(w_ti); bs, bz = qparams(b_ti); b = data(b_ti)
        if name == "CONV_2D":
            w_onnx = np.transpose(w, (0, 3, 1, 2))           # OHWI -> OIHW
            group = 1
        else:
            w_onnx = np.transpose(w, (3, 0, 1, 2))           # [1,kh,kw,C] -> [C,1,kh,kw]
            group = w_onnx.shape[0]
        kh, kw = w_onnx.shape[2], w_onnx.shape[3]
        sh, sw, fused = conv_opts(op, dw=(name == "DEPTHWISE_CONV_2D"))
        # Real (a)symmetric SAME padding from the tflite in/out dims: total =
        # max((out-1)*stride + k - in, 0). Do NOT use kh-1 (that is the stride-1
        # formula; it only matches stride-2 SAME when (out-1)*stride == in-1).
        in_h, in_w = shape(ins[0])[1], shape(ins[0])[2]
        out_h, out_w = shape(out_ti)[1], shape(out_ti)[2]
        tot_h = max((out_h - 1) * sh + kh - in_h, 0)
        tot_w = max((out_w - 1) * sw + kw - in_w, 0)
        pt, pb = tot_h // 2, tot_h - tot_h // 2   # TFLite SAME_UPPER: extra at the end
        pl, pr = tot_w // 2, tot_w - tot_w // 2
        w_dq = dq(tag + "_w", w_onnx, ws, np.zeros_like(wz), axis=0)
        b_dq = dq(tag + "_b", b.astype(np.int32), bs, np.zeros_like(bz).astype(np.int32),
                  axis=0, zp_dtype=np.int32)
        conv_out = tag + "_conv"
        nodes.append(helper.make_node("Conv", [x, w_dq, b_dq], [conv_out],
                     kernel_shape=[kh, kw], strides=[sh, sw], pads=[pt, pl, pb, pr], group=group))
        tmap[out_ti] = act_qdq(fused_act(conv_out, fused, tag), out_ti, tag)

    elif name == "ADD":
        # Inverted-residual skip: two int8 branches (each with its own scale) are
        # dequantized to float, added, then requantized to the Add output's scale -
        # exactly what tflite's quantized Add does. The two inputs come from tmap
        # (one is the block input from several ops back).
        add_out = tag + "_add"
        nodes.append(helper.make_node("Add", [tmap[ins[0]], tmap[ins[1]]], [add_out]))
        tmap[out_ti] = act_qdq(fused_act(add_out, add_fused(op), tag), out_ti, tag)

    elif name == "MEAN":
        # GlobalAveragePool is a REQUANTIZING op (tflite MEAN folds in->out
        # rescale), so its output must carry quant params - put the Q/DQ on the
        # pool output, BEFORE the (quant-preserving) Flatten. Otherwise the pool
        # output is left unquantized and a requantizing kernel divides by a
        # zero output scale.
        gap = tag + "_gap"
        nodes.append(helper.make_node("GlobalAveragePool", [tmap[ins[0]]], [gap]))
        gapq = act_qdq(gap, out_ti, tag)
        flat = tag + "_flat"
        nodes.append(helper.make_node("Flatten", [gapq], [flat], axis=1))
        # Quantize the Flatten output too (it's the FC's input) - otherwise the
        # FC input scale is 0 and its effective requant collapses to zero.
        tmap[out_ti] = act_qdq(flat, out_ti, tag + "b")

    elif name == "FULLY_CONNECTED":
        x = tmap[ins[0]]
        w_ti = ins[1]; w = data(w_ti); ws, wz = qparams(w_ti)   # [OC, IC]
        w_dq = dq(tag + "_w", w, ws, np.zeros_like(wz),
                  axis=0 if ws.size > 1 else None)
        # int32 bias if present (tflite uses input index -1 for "no bias", e.g.
        # when a folded BatchNorm leaves a zero bias). When present, the folded
        # BN shift lives here, so reconstruct it; scale = in_scale * w_scale.
        b_ti = ins[2] if len(ins) > 2 else -1
        if b_ti >= 0:
            b = data(b_ti); bs, bz = qparams(b_ti)
            bias_in = dq(tag + "_b", b.astype(np.int32), bs,
                         np.zeros_like(bz).astype(np.int32),
                         axis=0 if bs.size > 1 else None, zp_dtype=np.int32)
        else:
            bias_in = tag + "_fcb"; add_init(np.zeros(w.shape[0], dtype=np.float32), bias_in)
        gemm = tag + "_gemm"
        nodes.append(helper.make_node("Gemm", [x, w_dq, bias_in], [gemm], transB=1))
        tmap[out_ti] = act_qdq(fused_act(gemm, fc_fused(op), tag), out_ti, tag)

    else:
        # Fail loud: silently skipping an op makes the reconstructed ONNX diverge
        # from the .tflite, which would void the "numerically identical model"
        # basis of the device-to-device parity comparison. Add explicit handling
        # for any new op (incl. boundary QUANTIZE/DEQUANTIZE) rather than dropping it.
        raise ValueError(
            f"tflite_to_qdq_onnx: unhandled tflite op '{name}' (op #{oi}). "
            f"The reconstructor must reproduce every op that affects the compared "
            f"output; add explicit handling for '{name}'.")

# model output = the subgraph's declared output tensor
out_ti = G.Outputs(0)
_out_sh = list(shape(out_ti))
Y = helper.make_tensor_value_info(tmap[out_ti], TensorProto.FLOAT, _out_sh)
g = helper.make_graph(nodes, "matched", [X], [Y], initializer=inits)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
m.ir_version = 8
onnx.checker.check_model(m)
onnx.save(m, onnx_path)
print(f"reconstructed {onnx_path}: {len(nodes)} nodes")
