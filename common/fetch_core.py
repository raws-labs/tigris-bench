#!/usr/bin/env python3
"""Materialize one suite's pinned TiGrIS releases and print their locations.

Each pinned repository is cloned at its release tag into build/core/, and the
compiler is installed from the published tigris-ml wheel of the same release.
The output is shell assignments for run_all.sh:

    eval "$(python3 common/fetch_core.py --suite esp32s3/latency-hil)"
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from check_core_versions import ROOT, SUITES, suite_repositories, validate_checkout

ROOT_VARIABLES = {
    "tigris_compiler": "TIGRIS_COMPILER_ROOT",
    "tigris_runtime": "TIGRIS_RUNTIME_ROOT",
    "tigris_cortex_m": "TIGRIS_CORTEX_M_ROOT",
}


def run(*command: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed:\n{completed.stderr.strip()}")
    return completed.stdout.strip()


def checkout(url: str, tag: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--quiet", "--depth", "1", "--branch", tag, url, str(path))
    return path


def compiler(version: str, venv: Path) -> Path:
    tigris = venv / "bin" / "tigris"
    if not tigris.exists():
        run(sys.executable, "-m", "venv", str(venv))
        run(str(venv / "bin" / "pip"), "install", "--quiet", f"tigris-ml=={version}")
    reported = run(str(tigris), "--version")
    if not reported.startswith(f"tigris, version {version} "):
        raise RuntimeError(f"{tigris} reports {reported!r}, expected release {version}")
    return tigris


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, choices=sorted(SUITES))
    parser.add_argument("--manifest", type=Path, default=ROOT / "core-versions.json")
    parser.add_argument("--dest", type=Path, default=ROOT / "build" / "core")
    args = parser.parse_args()
    pins = json.loads(args.manifest.read_text())["suites"][args.suite]
    try:
        roots = {
            repo: checkout(url, tag, args.dest / f"{repo}-{tag}").resolve()
            for repo, (url, tag) in suite_repositories(pins).items()
        }
        errors = validate_checkout(pins, roots)
        if errors:
            raise RuntimeError("\n".join(errors))
        tigris = compiler(pins["tigris"], args.dest / f"tigris-ml-{pins['tigris']}")
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for repo, path in roots.items():
        print(f"{ROOT_VARIABLES[repo]}={shlex.quote(str(path))}")
    print(f"TIGRIS_COMPILER={shlex.quote(str(tigris))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
