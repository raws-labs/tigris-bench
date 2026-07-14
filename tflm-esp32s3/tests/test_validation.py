from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SUITE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SUITE_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from benchmark_matrix import (  # noqa: E402
    BenchmarkDataError,
    EXPECTED_CELLS,
)
import results as results_script  # noqa: E402
import validate_accuracy  # noqa: E402


REFERENCE_VALUES: dict[str, list[int] | list[float]] = {
    "ds_cnn_reference_f32.bin": [0.25, -0.5],
    "ds_cnn_tflite_reference_f32.bin": [0.125, -0.25],
    "ds_cnn_reference_i8.bin": [1, -2, 3],
    "ds_cnn_matched_ref.bin": [1, -2, 3],
    "ds_cnn_tflite_reference_i8.bin": [7, 0, -7],
    "mobilenet_v1_reference_i8.bin": [4, -5],
    "mobilenet_v1_matched_ref.bin": [8, -9],
}


class ValidationFixture(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.raw = self.root / "raw"
        self.models = self.root / "models"
        self.raw.mkdir()
        self.models.mkdir()
        self._write_references()
        self._write_complete_matrix()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_references(self) -> None:
        for name, values in REFERENCE_VALUES.items():
            if "f32" in name:
                payload = struct.pack(f"<{len(values)}f", *values)
            else:
                payload = struct.pack(f"<{len(values)}b", *values)
            (self.models / name).write_bytes(payload)

    def _output_values(self, reference_file: str) -> list[int] | list[float]:
        return REFERENCE_VALUES[reference_file]

    def _log_text(self, filename: str) -> str:
        spec = EXPECTED_CELLS[filename]
        identity = (
            f"framework={spec.framework},kernel={spec.kernel},dtype={spec.dtype},"
            f"model={spec.model}")
        if spec.expected_status != "ok":
            return (
                "boot\n"
                f"BENCH_RESULT:{identity},status={spec.expected_status},"
                f"arena_kb={spec.budget_kb},model_flash_kb=3401\n"
                "BENCH_DONE\n"
            )

        assert spec.reference_file is not None
        values = self._output_values(spec.reference_file)
        if spec.output_key == "f32":
            output = "OUTPUT_F32:" + "".join(f" {float(v):.6f}" for v in values)
        else:
            output = "OUTPUT_I8:" + "".join(f" {int(v)}" for v in values)
        budget = (
            f"arena_kb={spec.budget_kb}"
            if spec.framework == "tflm"
            else f"sram_budget_kb={spec.budget_kb},sram_actual_kb={spec.budget_kb}"
        )
        return (
            "boot\n"
            f"{output}\n"
            f"BENCH_RESULT:{identity},latency_mean_ms=2.0,latency_min_ms=1.0,"
            f"latency_max_ms=3.0,latency_stdev_ms=0.1,{budget},"
            "plan_flash_kb=10,runs=10\n"
            "BENCH_DONE\n"
        )

    def _write_complete_matrix(self) -> None:
        for filename in EXPECTED_CELLS:
            (self.raw / filename).write_text(self._log_text(filename))

    def _summary(self) -> dict:
        configs = results_script.collect(self.raw)
        return {"configs": configs, "count": len(configs)}

    def test_complete_matrix_uses_model_and_framework_specific_references(self) -> None:
        summary = self._summary()
        validations = validate_accuracy.validate_summary(summary, self.models)
        self.assertEqual(len(validations), len(EXPECTED_CELLS))
        self.assertTrue(all(v["status"] == "pass" for v in validations))

        # These raw INT8 files have lengths not divisible by four. Reading them
        # as float32 (the original bug) either truncated them or raised.
        ds_ref = validate_accuracy.load_reference(
            self.models / "ds_cnn_reference_i8.bin", "int8")
        mb_ref = validate_accuracy.load_reference(
            self.models / "mobilenet_v1_matched_ref.bin", "int8")
        self.assertEqual(ds_ref.tolist(), [1, -2, 3])
        self.assertEqual(mb_ref.tolist(), [8, -9])

    def test_missing_cell_is_fatal(self) -> None:
        (self.raw / "tigris_i8_ref.log").unlink()
        with self.assertRaisesRegex(BenchmarkDataError, "missing result cells"):
            results_script.collect(self.raw)

    def test_unexpected_cell_is_fatal(self) -> None:
        (self.raw / "surprise.log").write_text(
            self._log_text("tigris_f32_ref.log"))
        with self.assertRaisesRegex(BenchmarkDataError, "unexpected log files"):
            results_script.collect(self.raw)

    def test_truncated_capture_is_fatal(self) -> None:
        path = self.raw / "tigris_i8_espnn.log"
        path.write_text(path.read_text().replace("BENCH_DONE\n", ""))
        with self.assertRaisesRegex(BenchmarkDataError, "expected exactly one BENCH_DONE"):
            results_script.collect(self.raw)

    def test_missing_device_output_is_fatal(self) -> None:
        path = self.raw / "tigris_i8_ref.log"
        lines = [line for line in path.read_text().splitlines()
                 if not line.startswith("OUTPUT_I8:")]
        path.write_text("\n".join(lines) + "\n")
        with self.assertRaisesRegex(BenchmarkDataError, r"output types=\[\]"):
            results_script.collect(self.raw)

    def test_unexpected_status_is_fatal(self) -> None:
        path = self.raw / "tigris_mbv1_i8_espnn_64k.log"
        path.write_text(path.read_text().replace(
            ",runs=10", ",status=TIMEOUT,runs=10"))
        with self.assertRaisesRegex(BenchmarkDataError, "status='TIMEOUT'"):
            results_script.collect(self.raw)

    def test_summary_matrix_is_revalidated(self) -> None:
        summary = self._summary()
        summary["configs"].pop()
        summary["count"] -= 1
        with self.assertRaisesRegex(BenchmarkDataError, "missing result cells"):
            validate_accuracy.validate_summary(summary, self.models)

    def test_cli_passes_complete_fixture_and_fails_corrupt_output(self) -> None:
        summary_path = self.root / "summary.json"
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        collected = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "results.py"), str(self.raw),
             "-o", str(summary_path)],
            text=True, capture_output=True, env=env, check=False)
        self.assertEqual(collected.returncode, 0, collected.stdout + collected.stderr)

        validated = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "validate_accuracy.py"),
             str(summary_path), str(self.models)],
            text=True, capture_output=True, env=env, check=False)
        self.assertEqual(validated.returncode, 0, validated.stdout + validated.stderr)

        run_all_summary = self.root / "run-all-summary.json"
        gate_env = env | {
            "BENCH_VALIDATE_ONLY": "1",
            "BENCH_RAW_DIR": str(self.raw),
            "BENCH_MODELS_DIR": str(self.models),
            "BENCH_SUMMARY": str(run_all_summary),
            "PYTHON": sys.executable,
        }
        gated = subprocess.run(
            ["bash", str(SCRIPTS_DIR / "run_all.sh")],
            text=True, capture_output=True, env=gate_env, check=False)
        self.assertEqual(gated.returncode, 0, gated.stdout + gated.stderr)
        accepted_summary = run_all_summary.read_bytes()

        (self.models / "mobilenet_v1_matched_ref.bin").write_bytes(
            struct.pack("<2b", 12, -5))
        rejected = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "validate_accuracy.py"),
             str(summary_path), str(self.models)],
            text=True, capture_output=True, env=env, check=False)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("Accuracy gate FAILED", rejected.stdout)

        rejected_gate = subprocess.run(
            ["bash", str(SCRIPTS_DIR / "run_all.sh")],
            text=True, capture_output=True, env=gate_env, check=False)
        self.assertNotEqual(rejected_gate.returncode, 0)
        self.assertEqual(run_all_summary.read_bytes(), accepted_summary)


if __name__ == "__main__":
    unittest.main()
