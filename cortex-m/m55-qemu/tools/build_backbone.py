"""Build a pretrained int8 QDQ ResNet-18 backbone at a given input resolution.

Produces resnet18_bb_<RES>.onnx: the full ResNet-18 feature extractor (conv1
through layer4, 20 convs + 8 residual adds + maxpool), truncated before the
global-average-pool head, with an int8 output. For the ESP32-S3 SRAM-only
tiling showcase (smaller edge input so the residual-skip tensors fit the
device's internal SRAM slow tier).
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import torch
import torchvision
from onnx import shape_inference
from onnx.utils import Extractor
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                      QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

ARCH = sys.argv[1]
RES = int(sys.argv[2]) if len(sys.argv) > 2 else 224
OUT = Path(sys.argv[3]) if len(sys.argv) > 2 else Path("/home/armin/.claude/jobs/1c449b07/tmp/resnet")
OUT.mkdir(parents=True, exist_ok=True)
print(f"Building ResNet-18 backbone at {RES}x{RES}")

# 1. Pretrained ResNet-18 -> float ONNX.
m = __import__("torchvision").models.get_model(ARCH, weights="DEFAULT").eval()
f32 = OUT / f"{ARCH}_{RES}_f32.onnx"
torch.onnx.export(m, torch.randn(1, 3, RES, RES), str(f32),
                  opset_version=17, input_names=["input"], output_names=["output"],
                  do_constant_folding=True)

# 2. Preprocess + static int8 QDQ quantization (per-channel weights).
prep = OUT / f"{ARCH}_{RES}_prep.onnx"
quant_pre_process(str(f32), str(prep), skip_symbolic_shape=False)


class RandCalib(CalibrationDataReader):
    def __init__(self, n=16):
        self.data = iter([{"input": np.random.randn(1, 3, RES, RES).astype(np.float32)}
                          for _ in range(n)])

    def get_next(self):
        return next(self.data, None)


i8 = OUT / f"{ARCH}_{RES}_i8.onnx"
quantize_static(str(prep), str(i8), RandCalib(), quant_format=QuantFormat.QDQ,
                per_channel=True, activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8)

# 3. Add kernel_shape to Conv nodes (from weight dims) so the compiler reads the
#    real receptive field (needed for correct halo / line-buffer tiling).
model = onnx.load(str(i8))
g = model.graph
inits = {t.name: t for t in g.initializer}
for node in g.node:
    if node.op_type == "Conv":
        has_ks = any(a.name == "kernel_shape" for a in node.attribute)
        if not has_ks:
            w = inits.get(node.input[1])
            if w is not None and len(w.dims) == 4:
                node.attribute.append(onnx.helper.make_attribute("kernel_shape", [int(w.dims[2]), int(w.dims[3])]))
model = shape_inference.infer_shapes(model)

# 4. Find the global-pool head and truncate to the int8 tensor feeding it.
pool_in = None
for node in g.node:   # take the LAST head-pool (SE blocks have internal pools)
    if node.op_type in ("GlobalAveragePool","ReduceMean","AveragePool"):
        pool_in = node.input[0]
if pool_in is None:
    raise SystemExit("no global-pool head found")
# int8 tensor = input of the DequantizeLinear that produces pool_in
bb_out = pool_in
for node in g.node:
    if node.op_type == "DequantizeLinear" and node.output and node.output[0] == pool_in:
        bb_out = node.input[0]
        break
print(f"backbone output tensor: {bb_out}")

ex = Extractor(model)
bb = ex.extract_model(["input"], [bb_out])
bb_path = OUT / f"{ARCH}_bb_{RES}.onnx"
onnx.save(bb, str(bb_path))

c = Counter(n.op_type for n in bb.graph.node)
print(f"saved {bb_path.name}: {bb_path.stat().st_size // 1024} KiB, ops: {dict(c)}")
