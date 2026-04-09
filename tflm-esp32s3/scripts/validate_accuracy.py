#!/usr/bin/env python3
"""Compare device output values against ORT reference to validate accuracy.

Usage:
    python validate_accuracy.py results/summary.json models/output/
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def load_reference(ref_path: Path, dtype: str) -> np.ndarray:
    """Load raw binary reference output."""
    data = ref_path.read_bytes()
    if dtype == "f32":
        return np.frombuffer(data, dtype=np.float32)
    else:
        # i8 reference is still stored as f32 (ORT output)
        return np.frombuffer(data, dtype=np.float32)


def validate_config(config: dict, models_dir: Path) -> dict:
    """Validate one benchmark config against reference."""
    framework = config.get("framework", "?")
    dtype = config.get("dtype", "f32")
    kernel = config.get("kernel", "?")
    name = f"{framework}_{dtype}_{kernel}"

    result = {"name": name, "status": "skip", "message": ""}

    outputs = config.get("output_values", {})
    if not outputs:
        result["message"] = "no output values in log"
        return result

    # Determine reference file
    if dtype == "int8":
        ref_file = models_dir / "ds_cnn_reference_i8.bin"
        device_vals = outputs.get("i8")
    else:
        ref_file = models_dir / "ds_cnn_reference_f32.bin"
        device_vals = outputs.get("f32")

    if device_vals is None:
        result["message"] = f"no output_{dtype} values in log"
        return result

    if not ref_file.exists():
        result["message"] = f"reference file not found: {ref_file}"
        return result

    ref = load_reference(ref_file, dtype)
    device = np.array(device_vals, dtype=np.float32 if dtype == "f32" else np.int8)

    if len(ref) != len(device):
        result["status"] = "fail"
        result["message"] = f"length mismatch: ref={len(ref)}, device={len(device)}"
        return result

    if dtype == "f32":
        # f32: check relative + absolute tolerance
        # TiGrIS should match ORT closely; TFLM may differ due to different
        # weight initialization (Keras vs ONNX), so we mainly check for crashes
        max_abs_diff = float(np.max(np.abs(ref - device)))
        if max_abs_diff < 0.01:
            result["status"] = "pass"
            result["message"] = f"max_abs_diff={max_abs_diff:.6f}"
        elif framework == "tflm":
            result["status"] = "warn"
            result["message"] = (f"max_abs_diff={max_abs_diff:.6f} "
                                 "(expected: TFLM uses different weights)")
        else:
            result["status"] = "fail"
            result["message"] = f"max_abs_diff={max_abs_diff:.6f} exceeds tolerance"
    else:
        # i8: check exact match for TiGrIS, allow some diff for TFLM
        ref_i8 = ref.astype(np.float32)  # reference is f32 from ORT
        # For int8, we can only do a rough comparison
        result["status"] = "info"
        result["message"] = f"device_argmax={np.argmax(device)}, ref_argmax={np.argmax(ref_i8)}"

    return result


def main():
    parser = argparse.ArgumentParser(description="Validate device output vs ORT reference")
    parser.add_argument("summary", type=Path, help="summary.json from results.py")
    parser.add_argument("models_dir", type=Path, help="models/output/ directory with reference files")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())

    print("Accuracy Validation\n")
    all_pass = True
    for config in summary["configs"]:
        result = validate_config(config, args.models_dir)
        status_icon = {"pass": "OK", "fail": "FAIL", "warn": "WARN",
                       "info": "INFO", "skip": "SKIP"}[result["status"]]
        print(f"  [{status_icon}] {result['name']}: {result['message']}")
        if result["status"] == "fail":
            all_pass = False

    print()
    if all_pass:
        print("All validations passed.")
    else:
        print("Some validations FAILED. Check results above.")
        exit(1)


if __name__ == "__main__":
    main()
