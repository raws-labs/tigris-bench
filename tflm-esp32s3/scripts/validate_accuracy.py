#!/usr/bin/env python3
"""Validate every successful ESP32-S3 result against its model reference.

INT8 references are raw int8 vectors, not float32 ORT output. Reference names
are selected by canonical matrix cell so DS-CNN and MobileNet can never be
silently compared to one another, and TFLM's independently generated DS-CNN is
checked against a reference produced from its own TFLite model.

Usage:
    python validate_accuracy.py results/summary.json models/output/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_matrix import (
    BenchmarkDataError,
    EXPECTED_CELLS,
    validate_matrix,
)


def load_reference(ref_path: Path, dtype: str) -> np.ndarray:
    """Load a raw reference vector using the dtype written by model preparation."""
    if not ref_path.is_file():
        raise BenchmarkDataError(f"reference file not found: {ref_path}")
    np_dtype = np.float32 if dtype == "f32" else np.int8
    payload = ref_path.read_bytes()
    if dtype == "f32" and len(payload) % np.dtype(np.float32).itemsize:
        raise BenchmarkDataError(
            f"float32 reference byte length is not divisible by 4: {ref_path}")
    data = np.frombuffer(payload, dtype=np_dtype)
    if data.size == 0:
        raise BenchmarkDataError(f"reference file is empty: {ref_path}")
    return data


def validate_config(
        config: dict,
        models_dir: Path,
        *,
        f32_atol: float = 0.01,
        int8_atol: int = 1,
) -> dict:
    """Validate one structurally valid canonical matrix cell."""
    log_file = config["log_file"]
    spec = EXPECTED_CELLS[log_file]
    name = f"{log_file.removesuffix('.log')} ({spec.model})"

    if spec.expected_status != "ok":
        return {
            "name": name,
            "status": "pass",
            "message": f"expected status={spec.expected_status}",
        }

    if spec.reference_file is None or spec.output_key is None:
        return {
            "name": name,
            "status": "fail",
            "message": "successful cell has no reference mapping",
        }

    ref_path = models_dir / spec.reference_file
    try:
        reference = load_reference(ref_path, spec.dtype)
    except BenchmarkDataError as exc:
        return {"name": name, "status": "fail", "message": str(exc)}

    values = config["output_values"][spec.output_key]
    device_dtype = np.float32 if spec.dtype == "f32" else np.int8
    device = np.asarray(values, dtype=device_dtype)

    if reference.size != device.size:
        return {
            "name": name,
            "status": "fail",
            "message": f"length mismatch: reference={reference.size}, device={device.size}",
        }

    if spec.dtype == "f32":
        diff = np.abs(reference.astype(np.float64) - device.astype(np.float64))
        max_abs = float(diff.max(initial=0.0))
        passed = bool(np.allclose(reference, device, rtol=1e-4, atol=f32_atol))
        return {
            "name": name,
            "status": "pass" if passed else "fail",
            "message": f"max_abs_diff={max_abs:.6f}, atol={f32_atol:g}",
        }

    diff = np.abs(reference.astype(np.int16) - device.astype(np.int16))
    max_abs = int(diff.max(initial=0))
    n_diff = int(np.count_nonzero(diff))
    passed = max_abs <= int8_atol
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "message": (
            f"max_abs_diff={max_abs}, differing={n_diff}/{device.size}, "
            f"atol={int8_atol}"
        ),
    }


def validate_summary(
        summary: dict,
        models_dir: Path,
        *,
        f32_atol: float = 0.01,
        int8_atol: int = 1,
) -> list[dict]:
    configs = summary.get("configs")
    if not isinstance(configs, list):
        raise BenchmarkDataError("summary must contain a configs list")
    if summary.get("count") != len(configs):
        raise BenchmarkDataError(
            f"summary count={summary.get('count')!r}, actual configs={len(configs)}")
    validate_matrix(configs, require_complete=True)
    return [
        validate_config(
            config, models_dir, f32_atol=f32_atol, int8_atol=int8_atol)
        for config in configs
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the complete ESP32-S3 matrix against model-specific references")
    parser.add_argument("summary", type=Path, help="summary.json from results.py")
    parser.add_argument("models_dir", type=Path, help="models/output directory")
    parser.add_argument(
        "--f32-atol", type=float, default=0.01,
        help="absolute tolerance for float32 output (default: 0.01)")
    parser.add_argument(
        "--int8-atol", type=int, default=1,
        help="maximum elementwise INT8 difference (default: 1 LSB)")
    args = parser.parse_args()

    if args.f32_atol < 0 or args.int8_atol < 0:
        parser.error("tolerances must be non-negative")

    try:
        summary = json.loads(args.summary.read_text())
        results = validate_summary(
            summary,
            args.models_dir,
            f32_atol=args.f32_atol,
            int8_atol=args.int8_atol,
        )
    except (OSError, json.JSONDecodeError, BenchmarkDataError) as exc:
        print(f"Accuracy gate failed: {exc}")
        raise SystemExit(1) from exc

    print("Accuracy Validation\n")
    for result in results:
        marker = "OK" if result["status"] == "pass" else "FAIL"
        print(f"  [{marker}] {result['name']}: {result['message']}")

    failures = [result for result in results if result["status"] != "pass"]
    print()
    if failures:
        print(f"Accuracy gate FAILED: {len(failures)} cell(s) did not match.")
        raise SystemExit(1)
    print("All validations passed.")


if __name__ == "__main__":
    main()
