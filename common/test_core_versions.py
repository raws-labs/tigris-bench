#!/usr/bin/env python3
"""Mutation tests for the per-suite TiGrIS release pins."""

from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import check_core_versions
from check_core_versions import (
    SUITES,
    suite_repositories,
    validate_checkout,
    validate_manifest,
    validate_pins_match_producing,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "core-versions.json"
ESP = "esp32s3/latency-hil"
CORTEX = "cortex-m/deployability-hil"


def git(path: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)


class CoreVersionContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.document = json.loads(MANIFEST.read_text())

    def test_manifest_is_valid(self) -> None:
        self.assertEqual(validate_manifest(self.document), [])

    def test_every_suite_is_pinned(self) -> None:
        mutated = copy.deepcopy(self.document)
        del mutated["suites"][ESP]
        self.assertTrue(any("suites must pin exactly" in error
                            for error in validate_manifest(mutated)))

    def test_cortex_suite_requires_cortex_m_release(self) -> None:
        mutated = copy.deepcopy(self.document)
        del mutated["suites"][CORTEX]["tigris_cortex_m"]
        self.assertTrue(any(error.startswith(CORTEX) for error in validate_manifest(mutated)))

    def test_pin_must_be_a_release_version(self) -> None:
        for value in ("0.11", "v0.11.0", "f64f57d", 11):
            mutated = copy.deepcopy(self.document)
            mutated["suites"][ESP]["tigris"] = value
            self.assertTrue(any("release version" in error
                                for error in validate_manifest(mutated)), value)

    def test_one_release_covers_compiler_and_runtime(self) -> None:
        repos = suite_repositories({"tigris": "1.2.3"})
        self.assertEqual({name: tag for name, (_, tag) in repos.items()},
                         {"tigris_compiler": "v1.2.3", "tigris_runtime": "v1.2.3"})

    def test_moving_one_suite_pin_fails_only_that_suite(self) -> None:
        tags = {suite: {repo: tag for repo, (_, tag) in
                        suite_repositories(self.document["suites"][suite]).items()}
                for suite in SUITES}
        mutated = copy.deepcopy(self.document)
        mutated["suites"][ESP]["tigris"] = "99.0.0"
        with mock.patch.object(check_core_versions, "producing_tags",
                               side_effect=lambda suite: tags[suite]):
            self.assertEqual(validate_pins_match_producing(self.document), [])
            errors = validate_pins_match_producing(mutated)
        self.assertTrue(errors)
        self.assertTrue(all(error.startswith(ESP) for error in errors))

    def test_summary_without_release_tags_fails(self) -> None:
        with mock.patch.object(check_core_versions, "producing_tags",
                               side_effect=KeyError("provenance")):
            errors = validate_pins_match_producing(self.document)
        self.assertEqual(len(errors), len(SUITES))

    def test_checkout_must_sit_clean_on_the_release_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            compiler, runtime = Path(tmp, "compiler"), Path(tmp, "runtime")
            for path in (compiler, runtime):
                path.mkdir()
                git(path, "init", "-q")
                git(path, "config", "user.email", "t@example.com")
                git(path, "config", "user.name", "t")
            (compiler / "compatibility.json").write_text(
                json.dumps({"integration": {"compiler_emits_schema": 9}}))
            (runtime / "include").mkdir()
            (runtime / "include/tigris.h").write_text("#define TIGRIS_SCHEMA_VERSION 9\n")
            for path in (compiler, runtime):
                git(path, "add", "-A")
                git(path, "commit", "-q", "-m", "release")
                git(path, "tag", "v1.2.3")
            roots = {"tigris_compiler": compiler, "tigris_runtime": runtime}
            pins = {"tigris": "1.2.3"}
            self.assertEqual(validate_checkout(pins, roots), [])

            (runtime / "include/tigris.h").write_text("#define TIGRIS_SCHEMA_VERSION 8\n")
            errors = validate_checkout(pins, roots)
            self.assertTrue(any("tracked modifications" in e for e in errors))
            self.assertTrue(any("compiler emits 9" in e for e in errors))

            git(runtime, "commit", "-q", "-am", "after the release")
            self.assertTrue(any("is not release tag" in e
                                for e in validate_checkout(pins, roots)))


    def test_esp_registry_runtime_must_match_the_pin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "main").mkdir()
            (project / "main/idf_component.yml").write_text(
                'dependencies:\n  raws-labs/tigris-runtime: "==1.2.3"\n')
            (project / "dependencies.lock").write_text(
                "dependencies:\n  raws-labs/tigris-runtime:\n"
                f"    component_hash: {'a' * 64}\n    version: 1.2.3\n"
                "direct_dependencies:\n")
            self.assertEqual(
                check_core_versions.validate_esp_component("1.2.3", project), [])
            errors = check_core_versions.validate_esp_component("1.2.4", project)
            self.assertEqual(len(errors), 2)


if __name__ == "__main__":
    unittest.main()
