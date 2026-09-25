#!/usr/bin/env python3
"""Mutation tests for exact compiler/runtime benchmark pins."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from unittest import mock

import check_core_versions
from check_core_versions import (
    SUITES,
    validate_checkout,
    validate_manifest,
    validate_pins_match_producing,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "core-versions.json"
ESP = "esp32s3/latency-hil"
CORTEX = "cortex-m/deployability-hil"


class CoreVersionContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.document = json.loads(MANIFEST.read_text())

    def test_manifest_is_valid(self) -> None:
        self.assertEqual(validate_manifest(self.document), [])

    def test_every_suite_is_pinned(self) -> None:
        mutated = copy.deepcopy(self.document)
        del mutated["suites"][ESP]
        self.assertTrue(
            any("suites must pin exactly" in error
                for error in validate_manifest(mutated)))

    def test_incompatible_schema_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.document)
        mutated["suites"][ESP]["runtime"]["accepts_schemas"] = [2, 3]
        self.assertTrue(
            any(
                error.startswith(ESP) and "does not accept" in error
                for error in validate_manifest(mutated)
            )
        )

    def test_abbreviated_commit_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.document)
        mutated["suites"][CORTEX]["compiler"]["commit"] = "f7f6f42"
        self.assertTrue(
            any(
                error.startswith(CORTEX) and "full Git SHA" in error
                for error in validate_manifest(mutated)
            )
        )

    def test_cortex_suite_requires_cortex_m_pin(self) -> None:
        mutated = copy.deepcopy(self.document)
        del mutated["suites"][CORTEX]["tigris_cortex_m"]
        self.assertTrue(
            any(error.startswith(CORTEX) and "must be objects" in error
                for error in validate_manifest(mutated)))

    def test_pins_match_each_suites_producing_core(self) -> None:
        self.assertEqual(validate_pins_match_producing(self.document), [])

    def test_moving_one_suite_pin_fails_only_that_suite(self) -> None:
        mutated = copy.deepcopy(self.document)
        mutated["suites"][ESP]["runtime"]["commit"] = "0" * 40
        errors = validate_pins_match_producing(mutated)
        self.assertTrue(errors)
        self.assertTrue(all(error.startswith(ESP) for error in errors))

    def test_summary_without_producing_revisions_fails(self) -> None:
        def no_provenance(suite: str) -> dict[str, object]:
            raise KeyError("provenance")

        with mock.patch.object(check_core_versions, "producing_revisions",
                               side_effect=no_provenance):
            errors = validate_pins_match_producing(self.document)
        self.assertEqual(len(errors), len(SUITES))

    def test_sibling_checkouts_match_when_present(self) -> None:
        compiler = ROOT.parent / "tigris"
        runtime = ROOT.parent / "tigris-runtime"
        cortex_m = ROOT.parent / "tigris-cortex-m"
        if not all((p / ".git").exists() for p in (compiler, runtime, cortex_m)):
            self.skipTest("sibling core checkouts are not present")
        self.assertEqual(
            validate_checkout(self.document["suites"][CORTEX],
                              compiler, runtime, cortex_m), [])


if __name__ == "__main__":
    unittest.main()
