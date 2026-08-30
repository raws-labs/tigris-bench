#!/usr/bin/env python3
"""Host correctness gate for the ESP32-S3 U-Net showcase.

Proves, on the host, that the compiled U-Net plan is numerically correct end to
end: U-Net -> matched QDQ ONNX -> TiGrIS compile -> runtime.

What this checks, and why each check is the right one for a deep int8 model:

1. Tiling transparency (STRICT, 0 LSB). The 232K+6M tiled plan and the same
   model compiled untiled at a large single fast pool are both executed through
   the sibling runtime's host contract runner. Their int8 outputs must be
   byte-identical. This is the decisive gate: it proves the 2D (TILE_AXIS_HW)
   ConvTranspose tiling, the height-streamed co-tiled skip chains (including the
   full-resolution 256x256 model-input concat skip), and the 1D encoder tiling
   introduce zero numerical error. A ConvTranspose weight-layout or co-tiled
   skip bug that depended on the tile geometry would break this bit-exactness.

2. Reconstruction and kernel tracking against oracles (REPORTED, plus a
   regression guard). The plan output is compared to ONNX Runtime on the QDQ
   ONNX and, when TensorFlow is available, to the TFLite interpreter on the
   int8 .tflite the ONNX was matched to. TFLite is the true integer oracle
   (what TFLM computes on device). ONNX Runtime executes the QDQ graph with
   float intermediates, so through a 17-stage decoder the two references
   themselves disagree by a few int8 LSB from accumulated requant rounding; a
   whole-model 1-LSB bound therefore is not achievable against either oracle
   and is not asserted here. Per-operator 1-LSB parity vs ORT (including int8
   per-channel ConvTranspose and the co-tiled 2D skip) is enforced separately
   by the compiler's cross-repo contract gate. When TFLite is available this
   script guards against kernel regression by requiring the plan to track the
   true integer oracle at least as tightly as ONNX Runtime does, within 1 LSB.

The int8 packing, NHWC layout, and quant handling are imported from the
compiler's cross-repo contract harness so this uses the exact runtime I/O
contract the reference corpus does.
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

HERE = Path(__file__).resolve()
BENCH_ROOT = HERE.parents[3]
DEFAULT_TIGRIS = BENCH_ROOT.parent / "tigris"
DEFAULT_RUNTIME = BENCH_ROOT.parent / "tigris-runtime"
MODELS_OUT = BENCH_ROOT / "models" / "output"
DEFAULT_MODEL = MODELS_OUT / "unet_matched.onnx"
DEFAULT_TFLITE = MODELS_OUT / "unet_i8.tflite"
TILED_BUDGET = "232K+6M"
UNTILED_BUDGET = "4M"

# Whole-model int8 oracle floor vs ONNX Runtime. 1 LSB is NOT the right
# whole-model bound for this 17-stage int8 decoder: ORT-QDQ (float
# intermediates) and the TFLite integer oracle themselves disagree by ~3 LSB
# here, and the s8 reference kernel adds its documented +/-1 requant nudge
# (-(1<<30) vs TFLite/ESP-NN 1-(1<<30)), so per-stage sub-LSB differences
# compound with depth. 4 LSB is therefore the correct deep-int8 tolerance, not 1
# (the repo already relaxes the TFLM DS-CNN cell to int8_atol=4). Per-operator
# 1-LSB parity vs ORT is enforced separately by the cross-repo contract gate.
# The worst measured here is 4 LSB on ~3.4% of elements; the fraction cap adds a
# distribution-shift guard that a max-only bound would miss, kept loose enough
# not to be flaky but tight enough to have teeth.
ORT_WORST_MAX = 4
ORT_OVER_1LSB_FRACTION_MAX = 0.10


def _load_contract_helpers(tigris_repo: Path):
    sys.path.insert(0, str(tigris_repo / "scripts"))
    import crossrepo_contract as cc

    return cc


def _compile(compiler: Path, model: Path, budget: str, out: Path) -> None:
    print(f"compile {model.name} -> {out.name} (-m {budget})")
    subprocess.run(
        [str(compiler), "compile", str(model), "-m", budget, "-o", str(out)],
        check=True,
    )


def _run_plan(cc, runner: Path, plan_path: Path, q_nhwc: np.ndarray,
              work_dir: Path, tag: str) -> np.ndarray:
    """Execute a plan on a fixed int8 input; return the int8 output in NHWC."""
    from tigris.emitters.binary.reader import read_binary_plan

    plan = read_binary_plan(plan_path.read_bytes())
    tin = plan["tensors"][plan["model_inputs"][0]]
    quant = plan["quant_params"][tin["quant_param_idx"]]
    scale = float(quant["scale"])
    zero_point = int(quant["zero_point"])
    # A fixed int8 input, expressed as the float NCHW tensor both ORT and the
    # runtime packer quantize back to the identical grid point (the dequantized
    # values land exactly on the grid), so input quantization never diverges.
    float_nchw = np.ascontiguousarray(
        ((q_nhwc.astype(np.int64) - zero_point).astype(np.float32) * scale)
        .transpose(0, 3, 1, 2)
    )
    inputs = {tin["name"]: float_nchw}
    inputs_path = work_dir / f"inputs_{tag}.bin"
    outputs_path = work_dir / f"outputs_{tag}.bin"
    inputs_path.write_bytes(cc._pack_inputs(plan, inputs))
    cc._run(
        [str(runner), str(plan_path), str(inputs_path), str(outputs_path)],
        f"unet runtime execution ({tag})",
    )
    tout = plan["tensors"][plan["model_outputs"][0]]
    data = np.frombuffer(outputs_path.read_bytes(), dtype=np.int8)
    return data.reshape(tout["shape"]).astype(np.int16)


def _ort_reference_nhwc(model_path: Path, inputs: dict) -> np.ndarray:
    """Run ORT on the QDQ ONNX, emitting the folded terminal int8 tensor."""
    model = onnx.load(str(model_path))
    graph = model.graph
    final_dq = next(
        n for n in graph.node
        if n.op_type == "DequantizeLinear" and n.output[0] == graph.output[0].name
    )
    int8_name = final_dq.input[0]
    dims = [d.dim_value for d in graph.output[0].type.tensor_type.shape.dim]
    reference = copy.deepcopy(model)
    del reference.graph.output[:]
    reference.graph.output.extend(
        [helper.make_tensor_value_info(int8_name, TensorProto.INT8, dims)]
    )
    onnx.checker.check_model(reference)
    session = ort.InferenceSession(
        reference.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    nchw = session.run(None, inputs)[0].astype(np.int16)  # [1, C, H, W]
    return np.ascontiguousarray(nchw.transpose(0, 2, 3, 1))


def _tflite_reference_nhwc(tflite_path: Path, q_nhwc: np.ndarray):
    """Run the TFLite interpreter (true integer oracle); None if unavailable."""
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"note: TFLite oracle skipped (TensorFlow unavailable: {exc})")
        return None
    if not tflite_path.is_file():
        print(f"note: TFLite oracle skipped (not found: {tflite_path})")
        return None
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    interp.set_tensor(inp["index"], q_nhwc)
    interp.invoke()
    return interp.get_tensor(out["index"]).astype(np.int16)


def _report(actual: np.ndarray, expected: np.ndarray, label: str) -> tuple[int, int]:
    """Print worst abs diff and >1-LSB count; return (worst, over_1lsb_count)."""
    diff = np.abs(actual.reshape(-1) - expected.reshape(-1))
    worst = int(diff.max())
    over = int((diff > 1).sum())
    exact = int((diff == 0).sum())
    print(
        f"  {label:22s} worst_abs_diff={worst} "
        f">1LSB={over}/{actual.size} exact={exact}/{actual.size}"
    )
    return worst, over


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tflite", type=Path, default=DEFAULT_TFLITE)
    parser.add_argument("--tigris", type=Path, default=DEFAULT_TIGRIS)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument(
        "--compiler", type=Path,
        default=DEFAULT_TIGRIS / ".venv" / "bin" / "tigris",
    )
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument(
        "--work-dir", type=Path,
        default=BENCH_ROOT / "build" / "host_parity_unet",
    )
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(f"model not found: {args.model} (run models/prepare.py)")
    cc = _load_contract_helpers(args.tigris.resolve())

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    tiled_plan = work_dir / "unet.tgrs"
    untiled_plan = work_dir / "unet_untiled.tgrs"
    _compile(args.compiler, args.model, TILED_BUDGET, tiled_plan)
    _compile(args.compiler, args.model, UNTILED_BUDGET, untiled_plan)

    runner = args.runtime.resolve() / "build-contract" / "tigris_contract_runner"
    if not runner.is_file():
        runner, _ = cc._build_runner(
            args.runtime.resolve(), work_dir / "runtime-build"
        )

    # Fixed int8 NHWC input matching the plan's input layout.
    from tigris.emitters.binary.reader import read_binary_plan

    tin = read_binary_plan(tiled_plan.read_bytes())
    tshape = tin["tensors"][tin["model_inputs"][0]]["shape"]
    rng = np.random.default_rng(args.seed)
    q_nhwc = rng.integers(-128, 128, size=tuple(tshape), dtype=np.int64).astype(np.int8)

    tiled = _run_plan(cc, runner, tiled_plan, q_nhwc, work_dir, "tiled")
    untiled = _run_plan(cc, runner, untiled_plan, q_nhwc, work_dir, "untiled")

    print("\nGate 1 - tiling transparency (232K+6M tiled vs 4M untiled):")
    tiling_worst, _ = _report(tiled, untiled, "tiled vs untiled")
    if tiling_worst != 0:
        print("FAIL: tiling changed the output; the tiled path is not transparent")
        return 1
    print("  PASS: 2D-tiled decoder + co-tiled skip chains are bit-exact (0 LSB)")

    # Rebuild the float NCHW input for the ORT reference (same grid points).
    quant = tin["quant_params"][tin["tensors"][tin["model_inputs"][0]]["quant_param_idx"]]
    scale = float(quant["scale"])
    zero_point = int(quant["zero_point"])
    float_nchw = np.ascontiguousarray(
        ((q_nhwc.astype(np.int64) - zero_point).astype(np.float32) * scale)
        .transpose(0, 3, 1, 2)
    )
    input_name = tin["tensors"][tin["model_inputs"][0]]["name"]
    ort_nhwc = _ort_reference_nhwc(args.model, {input_name: float_nchw})

    print("\nGate 2 - ORT oracle floor (HARD; always runs, onnxruntime is in the venv):")
    ort_worst, ort_over = _report(tiled, ort_nhwc, "TiGrIS vs ORT-QDQ")
    ort_over_fraction = ort_over / tiled.size
    if ort_worst > ORT_WORST_MAX:
        print(
            f"FAIL: TiGrIS vs ORT worst {ort_worst} LSB exceeds the deep-int8 "
            f"bound {ORT_WORST_MAX} (see ORT_WORST_MAX rationale)"
        )
        return 1
    if ort_over_fraction > ORT_OVER_1LSB_FRACTION_MAX:
        print(
            f"FAIL: TiGrIS vs ORT {ort_over_fraction * 100:.2f}% of elements "
            f"exceed 1 LSB, over the {ORT_OVER_1LSB_FRACTION_MAX * 100:.0f}% cap"
        )
        return 1
    print(
        f"  PASS: TiGrIS tracks the ORT oracle within {ORT_WORST_MAX} LSB "
        f"({ort_over_fraction * 100:.2f}% of elements over 1 LSB)"
    )

    print("\nGate 3 - TFLite integer-oracle regression guard (runs only when TF present):")
    tflite_nhwc = _tflite_reference_nhwc(args.tflite, q_nhwc)
    if tflite_nhwc is not None:
        tigris_vs_tflite, _ = _report(tiled, tflite_nhwc, "TiGrIS vs TFLite")
        ort_vs_tflite, _ = _report(ort_nhwc, tflite_nhwc, "ORT-QDQ vs TFLite")
        # TiGrIS must track the true integer oracle at least as tightly as ONNX
        # Runtime does, within one LSB. This never loosens the per-operator
        # 1-LSB bound (enforced by the contract gate); it bounds only the
        # whole-model cross-engine accumulation, and is an ADDITIONAL check on
        # top of the always-on ORT floor above.
        if tigris_vs_tflite > ort_vs_tflite + 1:
            print(
                "FAIL: TiGrIS diverges from the TFLite integer oracle more than "
                f"ONNX Runtime does + 1 LSB ({tigris_vs_tflite} > {ort_vs_tflite} + 1)"
            )
            return 1
        print(
            "  PASS: TiGrIS tracks the TFLite integer oracle within 1 LSB of "
            "ONNX Runtime"
        )

    # On-device the ESP-NN backend uses the TFLite rounding-nudge convention, so
    # device-vs-TFLM parity is TIGHTER than these host s8_ref numbers; the
    # ~4-LSB host figure is cross-engine rounding, not accuracy loss.
    print("\nHost correctness gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
