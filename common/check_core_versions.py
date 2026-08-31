#!/usr/bin/env python3
"""Fail closed when a canonical benchmark uses unpinned TiGrIS core sources."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
RUNTIME_SCHEMA_RE = re.compile(
    r"^#define\s+TIGRIS_SCHEMA_VERSION(?:_V\d+)?\s+(\d+)\s*$",
    re.MULTILINE,
)


def _schema_list(value: object, label: str, errors: list[str]) -> list[int]:
    if not isinstance(value, list) or not value:
        errors.append(f"{label} must be a non-empty list")
        return []
    if any(not isinstance(item, int) or item < 1 for item in value):
        errors.append(f"{label} must contain positive integers")
        return []
    result = list(value)
    if result != sorted(set(result)):
        errors.append(f"{label} must be sorted and duplicate-free")
    return result


def validate_manifest(document: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(document, dict):
        return ["top level must be an object"]
    if document.get("format_version") != 1:
        errors.append("format_version must be 1")
    if not isinstance(document.get("profile"), str) or not document["profile"]:
        errors.append("profile must be a non-empty string")
    plan_schema = document.get("plan_schema")
    if not isinstance(plan_schema, int) or plan_schema < 1:
        errors.append("plan_schema must be a positive integer")

    compatibility = document.get("compatibility_manifest")
    compiler = document.get("compiler")
    runtime = document.get("runtime")
    if not all(isinstance(item, dict) for item in (compatibility, compiler, runtime)):
        return errors + [
            "compatibility_manifest, compiler, and runtime must be objects"
        ]
    assert isinstance(compatibility, dict)
    assert isinstance(compiler, dict)
    assert isinstance(runtime, dict)

    for label, component in (
        ("compatibility_manifest", compatibility),
        ("compiler", compiler),
        ("runtime", runtime),
    ):
        commit = component.get("commit")
        if not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit):
            errors.append(f"{label}.commit must be a full Git SHA")
        repository = component.get("repository")
        if not isinstance(repository, str) or not repository.startswith(
            "https://github.com/raws-labs/"
        ):
            errors.append(f"{label}.repository must be a RAWS Labs HTTPS URL")

    if compatibility.get("commit") != compiler.get("commit"):
        errors.append("compatibility manifest must come from the pinned compiler")
    if compatibility.get("path") != "compatibility.json":
        errors.append("compatibility_manifest.path must be compatibility.json")
    for label, component in (("compiler", compiler), ("runtime", runtime)):
        if component.get("branch") != "develop":
            errors.append(f"{label}.branch must be develop")

    emitted = compiler.get("emits_schema")
    accepted = _schema_list(
        runtime.get("accepts_schemas"), "runtime.accepts_schemas", errors
    )
    if emitted != plan_schema:
        errors.append("compiler schema must match plan_schema")
    if isinstance(emitted, int) and emitted not in accepted:
        errors.append("pinned runtime does not accept the compiler schema")
    return errors


def _git(path: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


def validate_checkout(
    document: dict[str, object], compiler_root: Path, runtime_root: Path
) -> list[str]:
    errors: list[str] = []
    for label, path, expected in (
        ("compiler", compiler_root, document["compiler"]),
        ("runtime", runtime_root, document["runtime"]),
    ):
        assert isinstance(expected, dict)
        try:
            actual = _git(path, "rev-parse", "HEAD")
            dirty = _git(path, "status", "--porcelain", "--untracked-files=no")
        except RuntimeError as exc:
            errors.append(f"cannot inspect {label} checkout {path}: {exc}")
            continue
        if actual != expected["commit"]:
            errors.append(
                f"{label} HEAD {actual} does not match pin {expected['commit']}"
            )
        if dirty:
            errors.append(f"{label} checkout has tracked modifications")

    compatibility_path = compiler_root / "compatibility.json"
    try:
        compatibility = json.loads(compatibility_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read compiler compatibility manifest: {exc}")
    else:
        integration = compatibility.get("integration", {})
        compiler = document["compiler"]
        runtime = document["runtime"]
        assert isinstance(compiler, dict)
        assert isinstance(runtime, dict)
        if integration.get("compiler_emits_schema") != compiler.get("emits_schema"):
            errors.append("compiler checkout compatibility schema disagrees with pin")
        if integration.get("runtime_accepts_schemas") != runtime.get(
            "accepts_schemas"
        ):
            errors.append("compiler compatibility runtime set disagrees with pin")

    header = runtime_root / "include/tigris.h"
    try:
        accepted = sorted(
            {int(value) for value in RUNTIME_SCHEMA_RE.findall(header.read_text())}
        )
    except OSError as exc:
        errors.append(f"cannot read runtime schema header: {exc}")
    else:
        runtime = document["runtime"]
        assert isinstance(runtime, dict)
        if accepted != runtime.get("accepts_schemas"):
            errors.append(
                f"runtime header accepts {accepted}, pin declares "
                f"{runtime.get('accepts_schemas')}"
            )
    return errors


CORTEX_SUMMARY = ROOT / "cortex-m/deployability-hil/results/summary.json"


def validate_pins_match_producing(document: object) -> list[str]:
    """The pinned core must be the core that produced the tracked device numbers.

    Advancing the pins without a rerun (or vice versa) would advertise a core the
    committed results did not come from. This ties the pins to the summary's
    embedded producing revisions."""
    if not isinstance(document, dict):
        return []
    try:
        summary = json.loads(CORTEX_SUMMARY.read_text())
        repos = summary["provenance"]["common"]["repositories"]
        producing = {
            "compiler": repos["tigris_compiler"]["revision"],
            "runtime": repos["tigris_runtime"]["revision"],
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        return [f"cannot read producing revisions from {CORTEX_SUMMARY.name}: {exc}"]
    errors: list[str] = []
    for label in ("compiler", "runtime"):
        pinned = document.get(label)
        if isinstance(pinned, dict) and pinned.get("commit") != producing[label]:
            errors.append(
                f"{label} pin {pinned.get('commit')} does not match the core that "
                f"produced the tracked results ({producing[label]}); re-pin with a "
                f"rerun so the pins and the committed numbers agree")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "core-versions.json")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--compiler-root", type=Path, default=ROOT.parent / "tigris")
    parser.add_argument(
        "--runtime-root", type=Path, default=ROOT.parent / "tigris-runtime"
    )
    parser.add_argument(
        "--allow-unpinned",
        action="store_true",
        help="warn instead of failing checkout mismatches for development runs",
    )
    args = parser.parse_args()
    try:
        document = json.loads(args.manifest.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read {args.manifest}: {exc}")
        return 1
    errors = validate_manifest(document)
    errors.extend(validate_pins_match_producing(document))
    if not errors and not args.manifest_only:
        errors.extend(
            validate_checkout(
                document, args.compiler_root.resolve(), args.runtime_root.resolve()
            )
        )
    if errors and args.allow_unpinned:
        for error in errors:
            print(f"WARNING: {error}")
        print("Development override accepted unpinned core sources.")
        return 0
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    compiler = document["compiler"]
    runtime = document["runtime"]
    assert isinstance(compiler, dict)
    assert isinstance(runtime, dict)
    print(
        "Pinned TiGrIS core verified: "
        f"compiler={compiler['commit']} runtime={runtime['commit']} "
        f"schema={document['plan_schema']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
