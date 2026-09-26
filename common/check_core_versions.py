#!/usr/bin/env python3
"""Fail closed when a canonical benchmark uses unpinned TiGrIS core sources.

Each suite pins the TiGrIS release (and, for Cortex-M, the tigris-cortex-m
release) its tracked numbers come from. A release pins both the compiler and
the runtime, which share a version. Checkouts must sit exactly on the release
tags, and a suite's summary must record those tags."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION_RE = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
RUNTIME_SCHEMA_RE = re.compile(
    r"^#define\s+TIGRIS_SCHEMA_VERSION(?:_V\d+)?\s+(\d+)\s*$",
    re.MULTILINE,
)

# Pin name -> the repositories one release of it covers.
REPOSITORIES = {
    "tigris": {
        "tigris_compiler": "https://github.com/raws-labs/tigris.git",
        "tigris_runtime": "https://github.com/raws-labs/tigris-runtime.git",
    },
    "tigris_cortex_m": {
        "tigris_cortex_m": "https://github.com/raws-labs/tigris-cortex-m.git",
    },
}

# Each suite pins the releases its tracked numbers were produced with, so a
# change that only affects one target reruns only that target's suite.
SUITES = {
    "cortex-m/deployability-hil": ("tigris", "tigris_cortex_m"),
    "esp32s3/latency-hil": ("tigris",),
}


def suite_repositories(pins: dict[str, str]) -> dict[str, tuple[str, str]]:
    """Repository name -> (URL, release tag) for one suite's pins."""
    return {
        repo: (url, f"v{pins[pin]}")
        for pin in pins
        for repo, url in REPOSITORIES[pin].items()
    }


ESP_PROJECT = ROOT / "esp32s3/latency-hil/tigris-esp"
ESP_COMPONENT = "raws-labs/tigris-runtime"


def esp_component(project: Path = ESP_PROJECT) -> dict[str, str | None]:
    """The registry runtime the ESP project requires and the one its lock holds."""
    manifest = (project / "main/idf_component.yml").read_text()
    lock = (project / "dependencies.lock").read_text()
    required = re.search(
        rf'^\s*{re.escape(ESP_COMPONENT)}:\s*"==([0-9.]+)"', manifest, re.MULTILINE)
    block = re.search(
        rf"^  {re.escape(ESP_COMPONENT)}:\n((?:    .*\n)+)", lock, re.MULTILINE)
    locked = re.search(r"^    version: ([0-9.]+)$", block.group(1), re.MULTILINE) if block else None
    digest = re.search(r"^    component_hash: ([0-9a-f]{64})$", block.group(1),
                       re.MULTILINE) if block else None
    return {
        "required": required.group(1) if required else None,
        "version": locked.group(1) if locked else None,
        "component_hash": digest.group(1) if digest else None,
    }


def validate_esp_component(version: str, project: Path = ESP_PROJECT) -> list[str]:
    """The ESP firmware compiles the registry runtime, so its exact requirement
    and its lock must name the suite's pinned release."""
    try:
        component = esp_component(project)
    except OSError as exc:
        return [f"cannot read the ESP project's component files: {exc}"]
    errors = []
    if component["required"] != version:
        errors.append(f"tigris-esp requires {ESP_COMPONENT} =={component['required']}, "
                      f"the suite pins {version}")
    if component["version"] != version or component["component_hash"] is None:
        errors.append(f"tigris-esp dependencies.lock holds {ESP_COMPONENT} "
                      f"{component['version']}, the suite pins {version}")
    return errors


def validate_manifest(document: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(document, dict):
        return ["top level must be an object"]
    if document.get("format_version") != 3:
        errors.append("format_version must be 3")
    if not isinstance(document.get("profile"), str) or not document["profile"]:
        errors.append("profile must be a non-empty string")
    suites = document.get("suites")
    if not isinstance(suites, dict) or set(suites) != set(SUITES):
        return errors + [f"suites must pin exactly {', '.join(sorted(SUITES))}"]
    for suite, names in SUITES.items():
        pins = suites[suite]
        if not isinstance(pins, dict) or set(pins) != set(names):
            errors.append(f"{suite}: must pin exactly {', '.join(names)}")
            continue
        for name in names:
            if not isinstance(pins[name], str) or not VERSION_RE.fullmatch(pins[name]):
                errors.append(f"{suite}: {name} must be a release version X.Y.Z")
    esp = suites.get("esp32s3/latency-hil")
    if isinstance(esp, dict) and isinstance(esp.get("tigris"), str):
        errors.extend(f"esp32s3/latency-hil: {problem}"
                      for problem in validate_esp_component(esp["tigris"]))
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


def validate_checkout(pins: dict[str, str], roots: dict[str, Path]) -> list[str]:
    """Every checkout sits clean on its release tag, and the runtime accepts
    the plan schema the compiler of the same release emits."""
    errors: list[str] = []
    for repo, (_, tag) in suite_repositories(pins).items():
        path = roots.get(repo)
        if path is None:
            errors.append(f"no checkout given for {repo}")
            continue
        try:
            head = _git(path, "rev-parse", "HEAD")
            tagged = _git(path, "rev-parse", f"{tag}^{{commit}}")
            dirty = _git(path, "status", "--porcelain", "--untracked-files=no")
        except RuntimeError as exc:
            errors.append(f"cannot inspect {repo} checkout {path} at {tag}: {exc}")
            continue
        if head != tagged:
            errors.append(f"{repo} HEAD {head} is not release tag {tag} ({tagged})")
        if dirty:
            errors.append(f"{repo} checkout has tracked modifications")

    compiler_root = roots.get("tigris_compiler")
    runtime_root = roots.get("tigris_runtime")
    if compiler_root is None or runtime_root is None:
        return errors
    try:
        compatibility = json.loads((compiler_root / "compatibility.json").read_text())
        emitted = compatibility["integration"]["compiler_emits_schema"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        return errors + [f"cannot read the compiler's emitted plan schema: {exc}"]
    try:
        accepted = sorted({
            int(value)
            for value in RUNTIME_SCHEMA_RE.findall(
                (runtime_root / "include/tigris.h").read_text())
        })
    except OSError as exc:
        return errors + [f"cannot read runtime schema header: {exc}"]
    if emitted not in accepted:
        errors.append(
            f"runtime accepts plan schemas {accepted}, compiler emits {emitted}")
    return errors


def producing_tags(suite: str) -> dict[str, object]:
    """Repository name -> release tag the suite's tracked summary records."""
    summary = json.loads((ROOT / suite / "results/summary.json").read_text())
    provenance = summary["provenance"]
    repos = provenance["common"]["repositories"] if "common" in provenance \
        else provenance["repositories"]
    return {repo: repos[repo].get("tag") for repo in repos}


def validate_pins_match_producing(document: object) -> list[str]:
    """Each suite's pins must be the releases that produced its tracked numbers.

    Advancing a pin without a rerun (or vice versa) would advertise a release the
    committed results did not come from."""
    if not isinstance(document, dict) or not isinstance(document.get("suites"), dict):
        return []
    errors: list[str] = []
    for suite in SUITES:
        pins = document["suites"].get(suite)
        if not isinstance(pins, dict):
            continue
        try:
            producing = producing_tags(suite)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
            errors.append(f"{suite}: cannot read producing releases from its summary: {exc}")
            continue
        for repo, (_, tag) in suite_repositories(pins).items():
            recorded = producing.get(repo)
            if recorded != tag:
                errors.append(
                    f"{suite}: pins {repo} {tag} but its tracked results record "
                    f"{recorded}; rerun the suite so the pins and the committed "
                    f"numbers agree")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "core-versions.json")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--suite", choices=sorted(SUITES),
                        help="suite whose pins the checkouts must match")
    parser.add_argument("--compiler-root", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--cortex-m-root", type=Path)
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
        roots = {
            name: path.resolve()
            for name, path in (("tigris_compiler", args.compiler_root),
                               ("tigris_runtime", args.runtime_root),
                               ("tigris_cortex_m", args.cortex_m_root))
            if path is not None
        }
        errors.extend(validate_checkout(document["suites"][args.suite], roots))
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
        releases = " ".join(f"{name}={version}" for name, version in pins.items())
        print(f"Pinned releases verified for {suite}: {releases}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
