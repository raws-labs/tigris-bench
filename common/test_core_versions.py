#!/usr/bin/env python3
"""Mutation tests for exact compiler/runtime benchmark pins."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from check_core_versions import validate_checkout, validate_manifest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "core-versions.json"


class CoreVersionContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.document = json.loads(MANIFEST.read_text())

    def test_manifest_is_valid(self) -> None:
        self.assertEqual(validate_manifest(self.document), [])

    def test_incompatible_schema_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.document)
        mutated["runtime"]["accepts_schemas"] = [2, 3]
        self.assertTrue(
            any(
                "does not accept" in error
                for error in validate_manifest(mutated)
            )
        )

    def test_abbreviated_commit_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.document)
        mutated["compiler"]["commit"] = "f7f6f42"
        self.assertTrue(
            any(
                "full Git SHA" in error
                for error in validate_manifest(mutated)
            )
        )

    def test_sibling_checkouts_match_when_present(self) -> None:
        compiler = ROOT.parent / "tigris"
        runtime = ROOT.parent / "tigris-runtime"
        if not (compiler / ".git").exists() or not (runtime / ".git").exists():
            self.skipTest("sibling core checkouts are not present")
        self.assertEqual(validate_checkout(self.document, compiler, runtime), [])


if __name__ == "__main__":
    unittest.main()
