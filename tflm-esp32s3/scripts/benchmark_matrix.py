#!/usr/bin/env python3
"""Canonical result matrix and structural validation for the ESP32-S3 suite."""

from __future__ import annotations

from dataclasses import dataclass
import math


class BenchmarkDataError(ValueError):
    """Raised when captured benchmark data does not satisfy the suite contract."""


@dataclass(frozen=True)
class CellSpec:
    framework: str
    kernel: str
    dtype: str
    model: str
    budget_kb: int
    output_key: str | None
    reference_file: str | None
    expected_status: str = "ok"
    int8_atol: int = 1


EXPECTED_RUNS = 10

# Filenames are part of the protocol: the four MobileNet TiGrIS runs otherwise
# have identical BENCH_RESULT identity fields and are distinguished by budget.
EXPECTED_CELLS: dict[str, CellSpec] = {
    "tigris_f32_ref.log": CellSpec(
        "tigris", "f32_ref", "f32", "ds_cnn", 256,
        "f32", "ds_cnn_reference_f32.bin"),
    "tigris_i8_ref.log": CellSpec(
        "tigris", "s8_ref", "int8", "ds_cnn_matched", 256,
        "i8", "ds_cnn_matched_ref.bin"),
    "tigris_i8_espnn.log": CellSpec(
        "tigris", "esp_nn", "int8", "ds_cnn_matched", 256,
        "i8", "ds_cnn_matched_ref.bin"),
    "tflm_f32.log": CellSpec(
        "tflm", "default", "f32", "ds_cnn", 256,
        "f32", "ds_cnn_tflite_reference_f32.bin"),
    "tflm_i8.log": CellSpec(
        "tflm", "default", "int8", "ds_cnn", 256,
        "i8", "ds_cnn_tflite_reference_i8.bin", int8_atol=4),
    "tigris_mbv1_i8_espnn.log": CellSpec(
        "tigris", "esp_nn", "int8", "mobilenet_v1_matched", 256,
        "i8", "mobilenet_v1_matched_ref.bin"),
    "tflm_mbv1_i8.log": CellSpec(
        "tflm", "default", "int8", "mobilenet_v1", 256,
        None, None, expected_status="ARENA_TOO_SMALL"),
    "tigris_mbv1_i8_espnn_128k.log": CellSpec(
        "tigris", "esp_nn", "int8", "mobilenet_v1_matched", 128,
        "i8", "mobilenet_v1_matched_ref.bin"),
    "tigris_mbv1_i8_espnn_64k.log": CellSpec(
        "tigris", "esp_nn", "int8", "mobilenet_v1_matched", 64,
        "i8", "mobilenet_v1_matched_ref.bin"),
    "tigris_mbv1_i8_espnn_32k.log": CellSpec(
        "tigris", "esp_nn", "int8", "mobilenet_v1_matched", 32,
        "i8", "mobilenet_v1_matched_ref.bin"),
}


def _actual_budget_kb(config: dict, spec: CellSpec) -> object:
    if spec.framework == "tflm":
        return config.get("arena_kb")
    # Current harnesses report both the compiler budget and the actually
    # provisioned arena. Older captures only carried sram_kb. The matrix is
    # keyed by compiler budget, not by a hardware-dependent SRAM cap.
    return config.get("sram_budget_kb", config.get("sram_kb"))


def validate_cell(log_file: str, config: dict) -> list[str]:
    """Return every contract violation for one canonical result cell."""
    spec = EXPECTED_CELLS[log_file]
    errors: list[str] = []

    for field in ("framework", "kernel", "dtype", "model"):
        expected = getattr(spec, field)
        actual = config.get(field)
        if actual != expected:
            errors.append(f"{field}={actual!r}, expected {expected!r}")

    actual_budget = _actual_budget_kb(config, spec)
    if actual_budget != spec.budget_kb:
        errors.append(
            f"budget={actual_budget!r} KB, expected {spec.budget_kb} KB")

    actual_status = str(config.get("status", "ok"))
    if actual_status != spec.expected_status:
        errors.append(
            f"status={actual_status!r}, expected {spec.expected_status!r}")

    outputs = config.get("output_values") or {}
    if not isinstance(outputs, dict):
        errors.append("output_values must be an object")
        outputs = {}

    if spec.expected_status == "ok":
        if config.get("runs") != EXPECTED_RUNS:
            errors.append(
                f"runs={config.get('runs')!r}, expected {EXPECTED_RUNS}")
        for field in (
                "latency_mean_ms", "latency_min_ms", "latency_max_ms",
                "latency_stdev_ms"):
            value = config.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                errors.append(f"missing or non-numeric {field}")

        mean = config.get("latency_mean_ms")
        minimum = config.get("latency_min_ms")
        maximum = config.get("latency_max_ms")
        if all(isinstance(v, (int, float)) for v in (minimum, mean, maximum)):
            if not minimum <= mean <= maximum:
                errors.append(
                    f"latency ordering invalid: min={minimum}, mean={mean}, max={maximum}")

        expected_keys = {spec.output_key}
        if set(outputs) != expected_keys:
            errors.append(
                f"output types={sorted(outputs)}, expected {sorted(expected_keys)}")
        elif not isinstance(outputs.get(spec.output_key), list) or not outputs[spec.output_key]:
            errors.append(f"missing or empty OUTPUT_{spec.output_key.upper()}")
        else:
            values = outputs[spec.output_key]
            if spec.output_key == "i8" and any(
                    not isinstance(v, int) or isinstance(v, bool) or v < -128 or v > 127
                    for v in values):
                errors.append("OUTPUT_I8 must contain integers in [-128, 127]")
            if spec.output_key == "f32" and any(
                    not isinstance(v, (int, float)) or not math.isfinite(v)
                    for v in values):
                errors.append("OUTPUT_F32 must contain finite numbers")
    elif outputs:
        errors.append(
            f"failure cell unexpectedly contains output types {sorted(outputs)}")

    return errors


def validate_matrix(configs: list[dict], require_complete: bool = True) -> None:
    """Validate identities, completeness, statuses, metrics, and outputs."""
    by_log: dict[str, dict] = {}
    errors: list[str] = []

    if not configs:
        errors.append("no result cells found")

    for config in configs:
        log_file = config.get("log_file")
        if not isinstance(log_file, str):
            errors.append("result is missing string log_file")
            continue
        if log_file not in EXPECTED_CELLS:
            errors.append(f"unexpected result cell: {log_file}")
            continue
        if log_file in by_log:
            errors.append(f"duplicate result cell: {log_file}")
            continue
        by_log[log_file] = config

    if require_complete:
        missing = sorted(set(EXPECTED_CELLS) - set(by_log))
        if missing:
            errors.append("missing result cells: " + ", ".join(missing))

    for log_file, config in sorted(by_log.items()):
        for problem in validate_cell(log_file, config):
            errors.append(f"{log_file}: {problem}")

    if errors:
        raise BenchmarkDataError("\n".join(errors))
