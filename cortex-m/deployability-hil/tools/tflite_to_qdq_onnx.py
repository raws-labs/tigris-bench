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
from tflite.TransposeConvOptions import TransposeConvOptions
from tflite.ConcatenationOptions import ConcatenationOptions
from tflite.ResizeNearestNeighborOptions import ResizeNearestNeighborOptions
from tflite.ResizeBilinearOptions import ResizeBilinearOptions
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

def tconv_opts(op):
    o = TransposeConvOptions()
    o.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
    return o.StrideH(), o.StrideW(), o.FusedActivationFunction()

def concat_opts(op):
    o = ConcatenationOptions()
    o.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
    return o.Axis(), o.FusedActivationFunction()

def resize_opts(op, bilinear):
    o = (ResizeBilinearOptions if bilinear else ResizeNearestNeighborOptions)()
    o.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
    return o.AlignCorners(), o.HalfPixelCenters()

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
        if len(shape(out_ti)) == 2:
            flat = tag + "_flat"
            nodes.append(helper.make_node("Flatten", [gapq], [flat], axis=1))
            # Quantize the Flatten output too (it's the FC's input) - otherwise
            # the FC input scale is lost between pool and dense.
            tmap[out_ti] = act_qdq(flat, out_ti, tag + "b")
        else:
            # MobileNetV1 has a 1x1 classifier Conv after global pooling and
            # therefore retains NCHW rank four at this point.
            tmap[out_ti] = gapq

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

    elif name in ("SHAPE", "STRIDED_SLICE", "PACK"):
        # MobileNetV1's TFLite converter emits these three integer-only nodes
        # solely to construct the following Reshape's static [1, classes]
        # shape.  The shape is already known from the declared output tensor,
        # so retaining them would add non-inference metadata to the ONNX graph.
        tmap[out_ti] = None

    elif name == "RESHAPE":
        # The preceding static shape calculation flattens a [1, 1, 1, C]
        # classifier tensor to [1, C].  ONNX Flatten is the equivalent data
        # operation and avoids encoding TFLite's shape-calculation subgraph.
        flat = tag + "_flat"
        nodes.append(helper.make_node("Flatten", [tmap[ins[0]]], [flat], axis=1))
        tmap[out_ti] = act_qdq(flat, out_ti, tag)

    elif name == "SOFTMAX":
        softmax = tag + "_softmax"
        nodes.append(helper.make_node("Softmax", [tmap[ins[0]]], [softmax], axis=1))
        tmap[out_ti] = act_qdq(softmax, out_ti, tag)

    elif name == "TRANSPOSE_CONV":
        # U-Net decoder upconvolution. TFLite TRANSPOSE_CONV input order is
        # [output_shape(int32), weights, input, bias?] - the real activation is
        # input 2; input 0 is a (sometimes runtime-built via SHAPE/PACK) output
        # shape we ignore, taking the static output dims from the op's output
        # tensor. Weights are OHWI [C_out, kH, kW, C_in], per-C_out quantized.
        x = tmap[ins[2]]
        w_ti = ins[1]
        w = data(w_ti); ws, wz = qparams(w_ti)
        # OHWI -> ONNX ConvTranspose weight layout IOHW [C_in, C_out, kH, kW]. The
        # TiGrIS compiler does its own IOHW->OHWI transpose internally
        # (writer._transpose_weight_nhwc), so emit standard ONNX order here.
        w_onnx = np.transpose(w, (3, 0, 1, 2))
        kh, kw = w_onnx.shape[2], w_onnx.shape[3]
        sh, sw, fused = tconv_opts(op)
        in_h, in_w = shape(ins[2])[1], shape(ins[2])[2]
        out_h, out_w = shape(out_ti)[1], shape(out_ti)[2]
        # ConvTranspose output = stride*(in-1) + k - pad_begin - pad_end + out_pad.
        # TFLite transpose_conv uses one origin pad = total//2 (floor at begin) and
        # clamps total >= 0; when the requested output exceeds the natural
        # transposed size the surplus is output_padding at the end.
        def split_pad(base, out):
            tot = base - out
            if tot >= 0:
                return tot // 2, tot - tot // 2, 0
            return 0, 0, -tot
        pb_h, pe_h, opad_h = split_pad((in_h - 1) * sh + kh, out_h)
        pb_w, pe_w, opad_w = split_pad((in_w - 1) * sw + kw, out_w)
        # Per-channel weight axis is C_out = axis 1 in the IOHW layout.
        w_dq = dq(tag + "_w", w_onnx, ws, np.zeros_like(wz), axis=1)
        ct_inputs = [x, w_dq]
        if len(ins) > 3 and ins[3] >= 0:
            b_ti = ins[3]; b = data(b_ti); bs, bz = qparams(b_ti)
            ct_inputs.append(dq(tag + "_b", b.astype(np.int32), bs,
                                np.zeros_like(bz).astype(np.int32),
                                axis=0 if bs.size > 1 else None, zp_dtype=np.int32))
        ct_out = tag + "_ct"
        nodes.append(helper.make_node("ConvTranspose", ct_inputs, [ct_out],
                     kernel_shape=[kh, kw], strides=[sh, sw],
                     pads=[pb_h, pb_w, pe_h, pe_w],
                     output_padding=[opad_h, opad_w], group=1))
        tmap[out_ti] = act_qdq(fused_act(ct_out, fused, tag), out_ti, tag)

    elif name == "CONCATENATION":
        # U-Net skip merge. Each int8 operand is already dequantized (via its
        # producer's Q/DQ) to real values; concatenate in float, then requantize to
        # the concat output scale - exactly tflite's quantized concat, which
        # requantizes every input to the output scale.
        ax, fused = concat_opts(op)
        rank = len(shape(out_ti))
        if ax < 0:
            ax += rank
        if rank == 4:                    # tflite NHWC axis index -> ONNX NCHW index
            ax = (0, 2, 3, 1)[ax]
        cc = tag + "_concat"
        nodes.append(helper.make_node("Concat", [tmap[i] for i in ins], [cc], axis=ax))
        tmap[out_ti] = act_qdq(fused_act(cc, fused, tag), out_ti, tag)

    elif name == "RESIZE_NEAREST_NEIGHBOR":
        # U-Net decoder upsample. input 0 = data, input 1 = target [h, w] (const).
        # Nearest is an exact int8 copy, so it reconstructs bit-for-bit. For an
        # integer upscale with align_corners=False, tflite nearest (half_pixel or
        # not) reduces to out_coord // factor, which ONNX asymmetric +
        # nearest_mode=floor reproduces exactly.
        x = tmap[ins[0]]
        in_h, in_w = shape(ins[0])[1], shape(ins[0])[2]
        out_h, out_w = shape(out_ti)[1], shape(out_ti)[2]
        align, half = resize_opts(op, bilinear=False)
        if out_h % in_h or out_w % in_w:
            raise ValueError(
                f"tflite_to_qdq_onnx: {name} (op #{oi}) is not an integer upscale "
                f"({in_h}x{in_w} -> {out_h}x{out_w}); only integer upscale is "
                f"reconstructed bit-exactly.")
        if align:
            raise ValueError(
                f"tflite_to_qdq_onnx: {name} (op #{oi}) has align_corners=True, "
                f"which is not reproduced by the integer-upscale nearest path.")
        scales = np.array([1.0, 1.0, out_h / in_h, out_w / in_w], dtype=np.float32)
        add_init(scales, tag + "_scl")
        rz = tag + "_resize"
        nodes.append(helper.make_node("Resize", [x, "", tag + "_scl"], [rz],
                     mode="nearest", coordinate_transformation_mode="asymmetric",
                     nearest_mode="floor"))
        tmap[out_ti] = act_qdq(rz, out_ti, tag)

    elif name == "RESIZE_BILINEAR":
        # Fail loud: tflite's INT8 bilinear kernel interpolates in integer
        # fixed-point, so its output differs from a float bilinear (the only form
        # expressible with the QDQ pattern) by up to 1 LSB per element - verified
        # against an ideal float bilinear. That per-element loss breaks the
        # bit-exact "numerically identical model" basis of the parity comparison
        # and accumulates through downstream ops. Use nearest upsampling in the
        # decoder (U-Net skip decoders do), which reconstructs exactly.
        raise ValueError(
            f"tflite_to_qdq_onnx: RESIZE_BILINEAR (op #{oi}) is not reconstructed: "
            f"tflite's integer bilinear kernel is not bit-exactly reproducible with "
            f"a float QDQ graph. Use nearest-neighbor upsampling in the decoder.")

    else:
        # Fail loud: silently skipping an op makes the reconstructed ONNX diverge
        # from the .tflite, which would void the "numerically identical model"
        # basis of the device-to-device parity comparison. Add explicit handling
        # for any new op (incl. boundary QUANTIZE/DEQUANTIZE) rather than dropping it.
        raise ValueError(
            f"tflite_to_qdq_onnx: unhandled tflite op '{name}' (op #{oi}). "
            f"The reconstructor must reproduce every op that affects the compared "
            f"output; add explicit handling for '{name}'.")

# model output = the subgraph's declared output tensor. The whole graph is NCHW,
# so a rank-4 (spatial, e.g. U-Net) output must be declared NCHW too; the tflite
# output shape is NHWC.
out_ti = G.Outputs(0)
_out_sh = list(shape(out_ti))
if len(_out_sh) == 4:                    # NHWC -> NCHW
    _out_sh = [_out_sh[0], _out_sh[3], _out_sh[1], _out_sh[2]]
Y = helper.make_tensor_value_info(tmap[out_ti], TensorProto.FLOAT, _out_sh)
g = helper.make_graph(nodes, "matched", [X], [Y], initializer=inits)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
m.ir_version = 8
m = onnx.shape_inference.infer_shapes(m)
onnx.checker.check_model(m)
onnx.save(m, onnx_path)
print(f"reconstructed {onnx_path}: {len(nodes)} nodes")
