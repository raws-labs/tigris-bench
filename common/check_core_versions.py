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


# Each suite pins the core that produced its tracked numbers, so a change that
# only affects one target reruns only that target's suite.
SUITES = {
    "cortex-m/deployability-hil": ("compiler", "runtime", "tigris_cortex_m"),
    "esp32s3/latency-hil": ("compiler", "runtime"),
}


def validate_suite_pins(pins: object, components: tuple[str, ...]) -> list[str]:
    errors: list[str] = []
    if not isinstance(pins, dict):
        return ["must be an object"]
    plan_schema = pins.get("plan_schema")
    if not isinstance(plan_schema, int) or plan_schema < 1:
        errors.append("plan_schema must be a positive integer")

    labels = ("compatibility_manifest", *components)
    if not all(isinstance(pins.get(label), dict) for label in labels):
        return errors + [f"{', '.join(labels)} must be objects"]
    compatibility = pins["compatibility_manifest"]
    compiler = pins["compiler"]
    runtime = pins["runtime"]

    for label in labels:
        component = pins[label]
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
    if "tigris_cortex_m" in components and pins["tigris_cortex_m"].get("branch") != "main":
        errors.append("tigris_cortex_m.branch must be main")

    emitted = compiler.get("emits_schema")
    accepted = _schema_list(
        runtime.get("accepts_schemas"), "runtime.accepts_schemas", errors
    )
    if emitted != plan_schema:
        errors.append("compiler schema must match plan_schema")
    if isinstance(emitted, int) and emitted not in accepted:
        errors.append("pinned runtime does not accept the compiler schema")
    return errors


def validate_manifest(document: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(document, dict):
        return ["top level must be an object"]
    if document.get("format_version") != 2:
        errors.append("format_version must be 2")
    if not isinstance(document.get("profile"), str) or not document["profile"]:
        errors.append("profile must be a non-empty string")
    suites = document.get("suites")
    if not isinstance(suites, dict) or set(suites) != set(SUITES):
        return errors + [f"suites must pin exactly {', '.join(sorted(SUITES))}"]
    for suite, components in SUITES.items():
        errors.extend(
            f"{suite}: {problem}"
            for problem in validate_suite_pins(suites[suite], components))
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
    pins: dict[str, object], compiler_root: Path, runtime_root: Path,
    cortex_m_root: Path | None = None
) -> list[str]:
    errors: list[str] = []
    checkouts = [
        ("compiler", compiler_root, pins["compiler"]),
        ("runtime", runtime_root, pins["runtime"]),
    ]
    if "tigris_cortex_m" in pins:
        if cortex_m_root is None:
            return ["this suite pins tigris_cortex_m; pass --cortex-m-root"]
        checkouts.append(("tigris_cortex_m", cortex_m_root, pins["tigris_cortex_m"]))
    for label, path, expected in checkouts:
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
        compiler = pins["compiler"]
        runtime = pins["runtime"]
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
        runtime = pins["runtime"]
        assert isinstance(runtime, dict)
        if accepted != runtime.get("accepts_schemas"):
            errors.append(
                f"runtime header accepts {accepted}, pin declares "
                f"{runtime.get('accepts_schemas')}"
            )
    return errors


def producing_revisions(suite: str) -> dict[str, object]:
    """The core revisions a suite's tracked summary records it was produced with."""
    summary = json.loads((ROOT / suite / "results/summary.json").read_text())
    if suite == "cortex-m/deployability-hil":
        repos = summary["provenance"]["common"]["repositories"]
    else:
        repos = summary["provenance"]["repositories"]
    return {
        label: repos[f"tigris_{label}" if label != "tigris_cortex_m" else label]["revision"]
        for label in SUITES[suite]
    }


def validate_pins_match_producing(document: object) -> list[str]:
    """Each suite's pins must be the core that produced its tracked numbers.

    Advancing a pin without a rerun (or vice versa) would advertise a core the
    committed results did not come from."""
    if not isinstance(document, dict) or not isinstance(document.get("suites"), dict):
        return []
    errors: list[str] = []
    for suite in SUITES:
        pins = document["suites"].get(suite)
        if not isinstance(pins, dict):
            continue
        try:
            producing = producing_revisions(suite)
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            errors.append(f"{suite}: cannot read producing revisions from its summary: {exc}")
            continue
        for label, revision in producing.items():
            pinned = pins.get(label)
            if isinstance(pinned, dict) and pinned.get("commit") != revision:
                errors.append(
                    f"{suite}: {label} pin {pinned.get('commit')} does not match the "
                    f"core that produced its tracked results ({revision}); re-pin "
                    f"with a rerun so the pins and the committed numbers agree")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "core-versions.json")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--suite", choices=sorted(SUITES),
                        help="suite whose pins the checkouts must match")
    parser.add_argument("--compiler-root", type=Path, default=ROOT.parent / "tigris")
    parser.add_argument(
        "--runtime-root", type=Path, default=ROOT.parent / "tigris-runtime"
    )
    parser.add_argument(
        "--cortex-m-root", type=Path, default=ROOT.parent / "tigris-cortex-m"
    )
    parser.add_argument(
        "--allow-unpinned",
        action="store_true",
        help="warn instead of failing checkout mismatches for development runs",
    )
    args = parser.parse_args()
    if not args.manifest_only and not args.suite:
        parser.error("--suite is required unless --manifest-only is given")
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
                document["suites"][args.suite], args.compiler_root.resolve(),
                args.runtime_root.resolve(), args.cortex_m_root.resolve()
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
    for suite in ([args.suite] if args.suite else sorted(SUITES)):
        pins = document["suites"][suite]
        commits = " ".join(
            f"{label}={pins[label]['commit']}" for label in SUITES[suite])
        print(f"Pinned TiGrIS core verified for {suite}: {commits} "
              f"schema={pins['plan_schema']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
