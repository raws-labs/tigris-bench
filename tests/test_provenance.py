#!/usr/bin/env python3
"""Mutation tests for the tracked benchmark provenance contract."""

from __future__ import annotations

import copy
import json
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


if __name__ == "__main__":
    unittest.main()
