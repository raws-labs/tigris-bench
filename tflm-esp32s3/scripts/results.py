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
from pathlib import Path

from rich.console import Console
from rich.table import Table


def parse_bench_result(line: str) -> dict | None:
    m = re.search(r"BENCH_RESULT:(.*)", line)
    if not m:
        return None
    result: dict = {}
    for pair in m.group(1).strip().split(","):
        if "=" not in pair:
            continue
        key, val = (s.strip() for s in pair.split("=", 1))
        try:
            result[key] = float(val) if "." in val else int(val)
        except ValueError:
            result[key] = val
    return result or None


def parse_output_values(lines: list[str]) -> dict:
    outputs: dict = {}
    for line in lines:
        if "OUTPUT_F32:" in line:
            outputs["f32"] = [float(v) for v in line.split("OUTPUT_F32:")[1].split()]
        elif "OUTPUT_I8:" in line:
            outputs["i8"] = [int(v) for v in line.split("OUTPUT_I8:")[1].split()]
    return outputs


def parse_log(path: Path) -> dict | None:
    lines = path.read_text().splitlines()
    result = next((r for line in lines if (r := parse_bench_result(line))), None)
    if result is None:
        return None
    outputs = parse_output_values(lines)
    if outputs:
        result["output_values"] = outputs
    result["log_file"] = path.name
    return result


def collect(path: Path) -> list[dict]:
    if path.is_dir():
        return [r for p in sorted(path.glob("*.log")) if (r := parse_log(p))]
    r = parse_log(path)
    return [r] if r else []


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
        sram = c.get("sram_kb", c.get("arena_kb", "?"))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse benchmark logs and print a results table")
    parser.add_argument("path", type=Path, help="Directory of .log files or a single log file")
    parser.add_argument("-o", "--output", type=Path, help="Also write summary JSON to this path")
    args = parser.parse_args()

    configs = collect(args.path)
    console = Console()

    if not configs:
        console.print("[yellow]No BENCH_RESULT lines found.[/yellow]")
        return

    console.print(render_table(configs))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"configs": configs, "count": len(configs)}, indent=2) + "\n")
        console.print(f"\nWrote {len(configs)} results to {args.output}")


if __name__ == "__main__":
    main()
