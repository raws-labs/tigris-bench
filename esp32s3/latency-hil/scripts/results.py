#!/usr/bin/env python3
"""Parse benchmark serial logs and print a results table.

Usage:
    python results.py results/raw/                    # print table
    python results.py results/raw/ -o summary.json    # also write JSON
    python results.py results/raw/tigris_f32_ref.log  # single file
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from rich.console import Console
from rich.table import Table

from benchmark_matrix import BenchmarkDataError, EXPECTED_CELLS, validate_matrix


def parse_bench_result(line: str) -> dict | None:
    m = re.search(r"BENCH_RESULT:(.*)", line)
    if not m:
        return None
    result: dict = {}
    for pair in m.group(1).strip().split(","):
        if "=" not in pair:
            raise BenchmarkDataError(f"malformed BENCH_RESULT field: {pair!r}")
        key, val = (s.strip() for s in pair.split("=", 1))
        if not key or not val:
            raise BenchmarkDataError(f"empty BENCH_RESULT key/value: {pair!r}")
        if key in result:
            raise BenchmarkDataError(f"duplicate BENCH_RESULT field: {key}")
        try:
            result[key] = float(val) if any(c in val for c in ".eE") else int(val)
        except ValueError:
            result[key] = val
    return result or None


def parse_output_values(lines: list[str]) -> dict:
    outputs: dict = {}
    for line in lines:
        if "OUTPUT_F32:" in line:
            if "f32" in outputs:
                raise BenchmarkDataError("duplicate OUTPUT_F32 line")
            try:
                outputs["f32"] = [
                    float(v) for v in line.split("OUTPUT_F32:", 1)[1].split()]
            except ValueError as exc:
                raise BenchmarkDataError(f"malformed OUTPUT_F32 line: {line!r}") from exc
        elif "OUTPUT_I8:" in line:
            if "i8" in outputs:
                raise BenchmarkDataError("duplicate OUTPUT_I8 line")
            try:
                values = [int(v) for v in line.split("OUTPUT_I8:", 1)[1].split()]
            except ValueError as exc:
                raise BenchmarkDataError(f"malformed OUTPUT_I8 line: {line!r}") from exc
            if any(v < -128 or v > 127 for v in values):
                raise BenchmarkDataError("OUTPUT_I8 contains a value outside [-128, 127]")
            outputs["i8"] = values
    return outputs


def parse_log(path: Path) -> dict:
    lines = path.read_text(errors="replace").splitlines()
    result_lines = [i for i, line in enumerate(lines) if "BENCH_RESULT:" in line]
    done_lines = [i for i, line in enumerate(lines) if line.strip() == "BENCH_DONE"]
    if len(result_lines) != 1:
        raise BenchmarkDataError(
            f"{path.name}: expected exactly one BENCH_RESULT, found {len(result_lines)}")
    if len(done_lines) != 1:
        raise BenchmarkDataError(
            f"{path.name}: expected exactly one BENCH_DONE, found {len(done_lines)}")
    if result_lines[0] > done_lines[0]:
        raise BenchmarkDataError(f"{path.name}: BENCH_RESULT appears after BENCH_DONE")

    try:
        result = parse_bench_result(lines[result_lines[0]])
    except BenchmarkDataError as exc:
        raise BenchmarkDataError(f"{path.name}: {exc}") from exc
    if result is None:  # Kept for the type checker; result_lines proves otherwise.
        raise BenchmarkDataError(f"{path.name}: empty BENCH_RESULT")
    try:
        outputs = parse_output_values(lines)
    except BenchmarkDataError as exc:
        raise BenchmarkDataError(f"{path.name}: {exc}") from exc
    if outputs:
        result["output_values"] = outputs
    result["log_file"] = path.name
    return result


def collect(path: Path, require_complete: bool | None = None) -> list[dict]:
    if not path.exists():
        raise BenchmarkDataError(f"input path does not exist: {path}")
    if path.is_dir():
        log_paths = sorted(path.glob("*.log"))
        names = {p.name for p in log_paths}
        unexpected = sorted(names - set(EXPECTED_CELLS))
        if unexpected:
            raise BenchmarkDataError("unexpected log files: " + ", ".join(unexpected))
        configs = [parse_log(p) for p in log_paths]
        validate_matrix(
            configs, require_complete=True if require_complete is None else require_complete)
        return configs

    result = parse_log(path)
    validate_matrix(
        [result], require_complete=False if require_complete is None else require_complete)
    return [result]


PROVENANCE_PREFIX = "BENCH_PROVENANCE:"


def provenance_line(compiler_root: Path, runtime_root: Path) -> str:
    """The line run_all.sh appends to every TiGrIS capture it saves."""
    record = {"repositories": {
        "tigris_compiler": core_state(compiler_root),
        "tigris_runtime": core_state(runtime_root),
    }}
    return PROVENANCE_PREFIX + json.dumps(record, sort_keys=True, separators=(",", ":"))


def collect_provenance(path: Path) -> dict | None:
    """The core every TiGrIS capture names; None when no capture carries one."""
    log_paths = sorted(path.glob("tigris_*.log")) if path.is_dir() else (
        [path] if path.name.startswith("tigris_") else [])
    records = {}
    for log_path in log_paths:
        lines = [line for line in log_path.read_text(errors="replace").splitlines()
                 if line.startswith(PROVENANCE_PREFIX)]
        if len(lines) > 1:
            raise BenchmarkDataError(f"{log_path.name}: more than one provenance line")
        records[log_path.name] = lines[0] if lines else None
    present = {name for name, line in records.items() if line is not None}
    if not present:
        return None
    if present != set(records):
        missing = sorted(set(records) - present)
        raise BenchmarkDataError("captures without provenance: " + ", ".join(missing))
    if len(set(records.values())) != 1:
        raise BenchmarkDataError("TiGrIS captures name different cores")
    try:
        return json.loads(next(iter(records.values()))[len(PROVENANCE_PREFIX):])
    except json.JSONDecodeError as exc:
        raise BenchmarkDataError(f"malformed provenance line: {exc}") from exc


def render_table(configs: list[dict]) -> Table:
    table = Table(title="Benchmark results", title_style="bold")
    table.add_column("Config", style="cyan", no_wrap=True)
    table.add_column("Latency (ms)", justify="right", no_wrap=True)
    table.add_column("SRAM (KB)", justify="right", no_wrap=True)
    table.add_column("Flash (KB)", justify="right", no_wrap=True)
    has_tiling = any(c.get("stages_tiled") or c.get("stages_chain") for c in configs)
    if has_tiling:
        table.add_column("Tiling", justify="right", no_wrap=True)

    for c in configs:
        framework = c.get("framework", "?")
        dtype = c.get("dtype", "?")
        kernel = c.get("kernel", "?")
        model = c.get("model", "")
        base = model.removesuffix("_i8") or "?"
        name = f"{framework} {base} {dtype} ({kernel})"
        sram = c.get(
            "sram_budget_kb", c.get("sram_kb", c.get("arena_kb", "?")))
        flash = c.get("plan_flash_kb", c.get("model_flash_kb", "?"))

        if c.get("status") == "ARENA_TOO_SMALL":
            latency = "[red]FAIL[/red]"
        else:
            mean = c.get("latency_mean_ms", 0)
            stdev = c.get("latency_stdev_ms", 0)
            latency = f"{mean:.1f} ± {stdev:.1f}"

        row = [name, latency, str(sram), str(flash)]
        if has_tiling:
            st = c.get("stages_tiled", 0)
            sc = c.get("stages_chain", 0)
            tt = c.get("total_tiles", 0)
            if st or sc:
                label = f"{st}t+{sc}c"
                if tt:
                    label += f" ({tt} tiles)"
                row.append(label)
            else:
                row.append("-")
        table.add_row(*row)

    return table


def core_state(root: Path) -> dict:
    """Revision and release tag of a core checkout, refusing tracked modifications."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(root), *args], text=True,
                              capture_output=True, check=True).stdout.strip()
    try:
        revision = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BenchmarkDataError(f"cannot read revision of {root}: {exc}") from exc
    if dirty:
        raise BenchmarkDataError(f"{root} has tracked modifications")
    state = {"revision": revision}
    try:
        state["tag"] = git("describe", "--tags", "--exact-match")
    except subprocess.CalledProcessError:
        pass
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse benchmark logs and print a results table")
    parser.add_argument("path", type=Path, nargs="?",
                        help="Directory of .log files or a single log file")
    parser.add_argument("-o", "--output", type=Path, help="Also write summary JSON to this path")
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Allow missing canonical cells when collecting a directory (unexpected or malformed "
             "cells still fail).")
    parser.add_argument(
        "--provenance-line", nargs=2, type=Path, metavar=("COMPILER_ROOT", "RUNTIME_ROOT"),
        help="Print the provenance line for captures built from these checkouts and exit")
    args = parser.parse_args()
    if args.path is None and not args.provenance_line:
        parser.error("path is required")

    console = Console()
    if args.provenance_line:
        try:
            print(provenance_line(*args.provenance_line))
        except BenchmarkDataError as exc:
            console.print(f"[bold red]RESULT GATE FAILED[/bold red] {exc}")
            raise SystemExit(1) from exc
        return
    try:
        require_complete = args.path.is_dir() and not args.allow_partial
        configs = collect(args.path, require_complete=require_complete)
        provenance = collect_provenance(args.path)
    except BenchmarkDataError as exc:
        console.print("[bold red]RESULT GATE FAILED[/bold red]")
        for line in str(exc).splitlines():
            console.print(f"  [red]x[/red] {line}")
        raise SystemExit(1) from exc

    console.print(render_table(configs))

    if args.output:
        document = {"configs": configs, "count": len(configs)}
        if provenance is not None:
            document["provenance"] = provenance
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2) + "\n")
        console.print(f"\nWrote {len(configs)} results to {args.output}")


if __name__ == "__main__":
    main()
