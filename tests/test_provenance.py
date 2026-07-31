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
from validate_tracked_json import (
    compare_performance_summaries,
    validate_readme_results,
)


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_PATH = ROOT / "cortex-m-deployability/results/provenance.json"
RESULTS_PATH = ROOT / "cortex-m-deployability/scripts/results.py"
README_PATH = ROOT / "cortex-m-deployability/README.md"
SUMMARY_PATH = ROOT / "cortex-m-deployability/results/summary.json"
ESP_SUMMARY_PATH = ROOT / "tflm-esp32s3/results/summary.json"
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


class ReadmeResultContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.summary = json.loads(SUMMARY_PATH.read_text())
        self.readme = README_PATH.read_text()

    def test_current_readme_matches_summary(self) -> None:
        self.assertEqual(
            validate_readme_results(self.summary, self.readme), [])

    def test_mutated_readme_result_is_rejected(self) -> None:
        mutated = self.readme.replace("11.14 ms", "11.15 ms", 1)
        errors = validate_readme_results(self.summary, mutated)
        self.assertTrue(
            any("11.14 ms" in error for error in errors), errors)

    def test_mutated_summary_result_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.summary)
        cell = next(
            config for config in mutated["configs"]
            if config["board"] == "nucleo_f446re"
            and config["model"] == "ts_matched"
            and config["kernel"] == "cmsis_nn")
        cell["latency_median_ms"] += 1
        errors = validate_readme_results(mutated, self.readme)
        self.assertTrue(
            any("| TS | 2.56 ms" in error for error in errors), errors)


class PerformanceRegressionContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cortex = json.loads(SUMMARY_PATH.read_text())
        self.esp = json.loads(ESP_SUMMARY_PATH.read_text())

    @staticmethod
    def cell(document: dict, log_file: str) -> dict:
        return next(
            config for config in document["configs"]
            if config["log_file"] == log_file)

    def test_unchanged_snapshots_have_no_deltas(self) -> None:
        for suite, summary in (("cortex-m", self.cortex), ("esp32-s3", self.esp)):
            observations, errors = compare_performance_summaries(
                summary, copy.deepcopy(summary), suite)
            self.assertEqual(observations, [])
            self.assertEqual(errors, [])

    def test_material_cycle_regression_is_rejected(self) -> None:
        current = copy.deepcopy(self.cortex)
        cell = self.cell(current, "h753_ds_cnn_cmsis_nn.log")
        cell["latency_median_cycles"] = int(
            cell["latency_median_cycles"] * 1.051)
        observations, errors = compare_performance_summaries(
            current, self.cortex, "cortex-m")
        self.assertTrue(any("median latency" in item for item in observations))
        self.assertTrue(any("exceeds +5%" in error for error in errors), errors)

    def test_timing_noise_below_limit_is_reported_but_allowed(self) -> None:
        current = copy.deepcopy(self.esp)
        cell = self.cell(current, "tigris_i8_espnn.log")
        cell["latency_mean_ms"] *= 1.049
        observations, errors = compare_performance_summaries(
            current, self.esp, "esp32-s3")
        self.assertTrue(any("mean latency" in item for item in observations))
        self.assertEqual(errors, [])

    def test_material_working_memory_growth_is_rejected(self) -> None:
        current = copy.deepcopy(self.cortex)
        cell = self.cell(current, "h753_ad_cmsis_nn.log")
        cell["sram_peak_bytes"] += 129
        _, errors = compare_performance_summaries(
            current, self.cortex, "cortex-m")
        self.assertTrue(
            any("all-in working memory" in error for error in errors), errors)

    def test_small_absolute_memory_growth_is_reported_but_allowed(self) -> None:
        current = copy.deepcopy(self.cortex)
        cell = self.cell(current, "h753_ad_cmsis_nn.log")
        cell["sram_peak_bytes"] += 64
        observations, errors = compare_performance_summaries(
            current, self.cortex, "cortex-m")
        self.assertTrue(
            any("all-in working memory" in item for item in observations))
        self.assertEqual(errors, [])

    def test_exact_firmware_growth_is_rejected(self) -> None:
        current = copy.deepcopy(self.cortex)
        provenance = current["provenance"]["cells"]
        artifact = provenance["h753_ds_cnn_cmsis_nn.log"]["artifacts"][
            "firmware"]
        artifact["size_bytes"] += 4096
        _, errors = compare_performance_summaries(
            current, self.cortex, "cortex-m")
        self.assertTrue(
            any("firmware artifact" in error for error in errors), errors)

    def test_success_to_failure_is_rejected(self) -> None:
        current = copy.deepcopy(self.esp)
        self.cell(current, "tigris_i8_espnn.log")["status"] = "FAILED"
        _, errors = compare_performance_summaries(
            current, self.esp, "esp32-s3")
        self.assertTrue(any("status regressed" in error for error in errors), errors)

    def test_removed_baseline_cell_cannot_hide_a_regression(self) -> None:
        current = copy.deepcopy(self.cortex)
        current["configs"] = [
            cell for cell in current["configs"]
            if cell["log_file"] != "h753_ds_cnn_cmsis_nn.log"]
        _, errors = compare_performance_summaries(
            current, self.cortex, "cortex-m")
        self.assertTrue(
            any("removed baseline cell" in error for error in errors), errors)

    def test_removed_baseline_metric_cannot_hide_a_regression(self) -> None:
        current = copy.deepcopy(self.esp)
        del self.cell(current, "tigris_i8_espnn.log")["latency_mean_ms"]
        _, errors = compare_performance_summaries(
            current, self.esp, "esp32-s3")
        self.assertTrue(
            any("removed baseline metric mean latency" in error
                for error in errors),
            errors,
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
    def test_build_uses_the_checkout_that_core_validation_inspects(self) -> None:
        run_all = (
            ROOT / "cortex-m-deployability/scripts/run_all.sh").read_text()
        stm32_cmake = (
            ROOT / "cortex-m-deployability/CMakeLists.txt").read_text()
        pico_cmake = (
            ROOT / "cortex-m-deployability/boards/pico2_rp2350/CMakeLists.txt"
        ).read_text()
        self.assertIn(
            '-DTIGRIS_RUNTIME_ROOT="$TIGRIS_RUNTIME_ROOT"', run_all)
        self.assertIn("add_subdirectory(${TIGRIS_RUNTIME_ROOT}", stm32_cmake)
        self.assertIn("set(RT_DIR     ${TIGRIS_RUNTIME_ROOT})", pico_cmake)

    def test_cmsis_scratch_is_provisioned_outside_plan_budget(self) -> None:
        harness = (
            ROOT / "cortex-m-deployability/harness/main.c").read_text()
        run_all = (
            ROOT / "cortex-m-deployability/scripts/run_all.sh").read_text()
        self.assertIn(
            "fast_size = tigris_cmsis_nn_fast_arena_required(&plan);",
            harness,
        )
        self.assertIn("local plan fast=65536 slow=8192", run_all)

    def test_non_tiled_plan_budgets_do_not_inflate_ram_high_water(self) -> None:
        plan_builder = (
            ROOT / "cortex-m-deployability/scripts/prepare_tigris_plans.py"
        ).read_text()
        self.assertIn(
            '"ds_cnn": ("ds_cnn_matched.onnx", "20K", '
            '"ds_cnn_matched.tgrs")',
            plan_builder,
        )
        self.assertNotIn("_matched_32k.tgrs", plan_builder)

    def test_subset_runs_cannot_replace_the_canonical_summary(self) -> None:
        run_all = (
            ROOT / "cortex-m-deployability/scripts/run_all.sh").read_text()
        self.assertIn('RUN_RAW="$(mktemp -d)"', run_all)
        self.assertIn('"$HERE/scripts/results.py" "$RUN_RAW"', run_all)
        self.assertIn('if [ "$CANONICAL_RUN" -eq 1 ]', run_all)
        self.assertIn(
            "Subset run complete; canonical summary left unchanged.",
            run_all,
        )

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


class HarnessWorkspaceContractTest(unittest.TestCase):
    def test_tigris_metadata_uses_generated_plan_sized_workspace(self) -> None:
        harness = (
            ROOT / "cortex-m-deployability/harness/main.c").read_text()

        self.assertIn(
            "s_executor_workspace[\n"
            "    TIGRIS_CODEGEN_EXECUTOR_WORKSPACE_BYTES]",
            harness,
        )
        self.assertIn("tigris_codegen_run_with_workspace_buffer(", harness)
        self.assertIn("tigris_run_with_workspace_buffer(", harness)
        self.assertIn(
            "sizeof(s_executor_workspace)", harness)
        self.assertNotIn(
            "tigris_executor_workspace_t s_executor_workspace", harness)


if __name__ == "__main__":
    unittest.main()
