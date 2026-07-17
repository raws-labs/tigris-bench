#!/usr/bin/env python3
"""Mutation tests for the tracked benchmark provenance contract."""

from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from provenance_validation import (
    all_source_captures_present,
    git_tracked_paths,
    validate_provenance,
)


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_PATH = ROOT / "cortex-m-deployability/results/provenance.json"
RESULTS_PATH = ROOT / "cortex-m-deployability/scripts/results.py"
RESULTS_SPEC = importlib.util.spec_from_file_location(
    "cortex_results", RESULTS_PATH)
assert RESULTS_SPEC and RESULTS_SPEC.loader
cortex_results = importlib.util.module_from_spec(RESULTS_SPEC)
RESULTS_SPEC.loader.exec_module(cortex_results)


class ProvenanceContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.document = json.loads(PROVENANCE_PATH.read_text())
        self.tracked_paths = git_tracked_paths(ROOT)

    def validate(self, document: object) -> list[str]:
        return validate_provenance(document, ROOT, self.tracked_paths)

    def assert_rejected_with(self, document: object, message: str) -> None:
        errors = self.validate(document)
        self.assertTrue(
            any(message in error for error in errors),
            f"expected {message!r} in validation errors: {errors}",
        )

    def test_current_provenance_is_valid(self) -> None:
        self.assertEqual(self.validate(self.document), [])

    def test_clean_clone_without_gitignored_captures_is_valid(self) -> None:
        original_exists = Path.exists
        original_is_file = Path.is_file

        def is_capture(path: Path) -> bool:
            return "/results/raw/" in path.as_posix()

        with (
            mock.patch.object(
                Path, "exists",
                lambda path: False if is_capture(path) else original_exists(path)),
            mock.patch.object(
                Path, "is_file",
                lambda path: False if is_capture(path) else original_is_file(path)),
        ):
            self.assertEqual(self.validate(self.document), [])
            self.assertFalse(
                all_source_captures_present(self.document, ROOT))

    def test_mutated_result_hash_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.document)
        mutated["result_artifacts"][0]["sha256"] = "0" * 64
        self.assert_rejected_with(mutated, "SHA-256 mismatch")

    def test_mutated_collector_hash_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.document)
        mutated["result_artifacts"][0]["collector"]["sha256"] = "0" * 64
        self.assert_rejected_with(mutated, "collector SHA-256 mismatch")

    def test_mutated_validator_hash_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.document)
        mutated["result_artifacts"][0]["validator"]["sha256"] = "0" * 64
        self.assert_rejected_with(mutated, "validator SHA-256 mismatch")

    def test_removed_source_capture_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.document)
        removed = mutated["result_artifacts"][0]["source_captures"].pop()
        self.assert_rejected_with(mutated, f"missing source capture {removed['path']}")

    def test_missing_unknown_revision_field_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.document)
        del mutated["execution_provenance"]["tigris_runtime_revision"]
        self.assert_rejected_with(mutated, "execution_provenance fields=")

    def test_unavailable_artifact_repository_revision_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.document)
        mutated["artifact_repository"]["revision"] = "0" * 40
        self.assert_rejected_with(mutated, "revision " + "0" * 40 + " is unavailable")

    def test_reachable_but_incorrect_repository_revision_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.document)
        mutated["artifact_repository"]["revision"] = (
            "a5fab6ff574c882c756771137f9a716f3e4f5e7b"
        )
        self.assert_rejected_with(
            mutated,
            "revision SHA-256 mismatch for "
            "cortex-m-deployability/results/summary.json",
        )


def valid_capture_provenance() -> dict:
    revision = "1" * 40
    digest = "2" * 64
    return {
        "captured_at_utc": "2026-07-17T12:00:00Z",
        "repositories": {
            name: {"revision": revision, "dirty": False}
            for name in (
                "benchmark", "tigris_compiler", "tigris_runtime",
                "tflite_micro",
            )
        },
        "dependencies": {
            name: revision
            for name in (
                "CMSIS-NN", "CMSIS-Core", "cmsis-device-f4",
                "cmsis-device-h7",
            )
        },
        "tools": {
            "arm_none_eabi_gcc": "arm-none-eabi-gcc 13.2.1",
            "cmake": "cmake version 3.28.3",
            "pico_sdk_revision": None,
        },
        "host_model_environment": {
            "requirements_sha256": digest,
            "packages": {"numpy": "2.3.5"},
        },
        "siliconrig_sdk_version": "0.2.1",
        "build": {"configure_command": "cmake -S . -B build/test -O2"},
        "artifacts": {
            kind: {"name": f"{kind}.bin", "sha256": digest, "size_bytes": 42}
            for kind in ("model", "firmware")
        },
        "board": {
            "siliconrig_board_id": "board_test",
            "board_type": "stm32-h753",
            "specs": {"mcu": "STM32H753ZI"},
        },
    }


def write_capture(path: Path, provenance: object | None) -> None:
    lines = [
        "BENCH_RESULT:framework=tigris,kernel=cmsis_nn,dtype=int8,"
        "model=ds_cnn_matched,board=nucleo_h753zi,cpu_mhz=480,status=ok,"
        "latency_median_ms=1.0,latency_median_cycles=480000,"
        "sram_peak_bytes=1024,runs=30",
    ]
    if provenance is not None:
        lines.append(
            "BENCH_PROVENANCE:"
            + json.dumps(provenance, sort_keys=True, separators=(",", ":")))
    path.write_text("\n".join(lines) + "\n")


class CaptureProvenanceTest(unittest.TestCase):
    def test_complete_capture_is_collected_and_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "h753_ds_cnn_cmsis_nn.log"
            write_capture(path, valid_capture_provenance())
            configs = cortex_results.collect(path, require_provenance=True)
            summary_provenance = cortex_results.extract_summary_provenance(
                configs, required=True)

        self.assertNotIn("_capture_provenance", configs[0])
        self.assertEqual(
            summary_provenance["source"], "BENCH_PROVENANCE")
        self.assertIn(path.name, summary_provenance["cells"])
        self.assertIn("repositories", summary_provenance["common"])

    def test_missing_capture_provenance_is_rejected_when_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.log"
            write_capture(path, None)
            with self.assertRaisesRegex(ValueError, "missing BENCH_PROVENANCE"):
                cortex_results.collect(path, require_provenance=True)

    def test_dirty_repository_is_rejected(self) -> None:
        provenance = valid_capture_provenance()
        provenance["repositories"]["tigris_runtime"]["dirty"] = True
        errors = cortex_results.validate_capture_provenance(
            provenance, "nucleo_h753zi")
        self.assertIn(
            "repositories.tigris_runtime.dirty must be false", errors)

    def test_missing_artifact_hash_is_rejected(self) -> None:
        provenance = valid_capture_provenance()
        del provenance["artifacts"]["firmware"]["sha256"]
        errors = cortex_results.validate_capture_provenance(
            provenance, "nucleo_h753zi")
        self.assertIn("artifacts.firmware.sha256 must be a SHA-256", errors)

    def test_rp2350_requires_pico_sdk_revision(self) -> None:
        provenance = valid_capture_provenance()
        errors = cortex_results.validate_capture_provenance(
            provenance, "pico2_rp2350")
        self.assertIn(
            "tools.pico_sdk_revision must be a full Git SHA for RP2350",
            errors,
        )


if __name__ == "__main__":
    unittest.main()
