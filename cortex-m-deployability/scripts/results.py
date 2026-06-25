#!/usr/bin/env python3
"""Parse Cortex-M benchmark serial logs and print a results table.

Reads the BENCH_RESULT line each harness prints and renders latency (ms +
cycles), RAM working set, and flash size. Adapted from the ESP suite's
results.py; understands the extra Cortex-M fields (board, cpu_mhz, cycles).

Usage:
    python results.py results/raw/                  # print table
    python results.py results/raw/ -o summary.json  # also write JSON
    python results.py results/raw/h753_cmsis_nn.log # single file
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


def parse_clock_stage(lines: list[str]) -> int | None:
    """The TiGrIS harness prints `CLOCK_DIAG: stage=N ...`; stage 5 = the rated
    PLL clock locked. Returns None if no diagnostic line (e.g. the TFLM harness)."""
    for line in lines:
        m = re.search(r"CLOCK_DIAG:\s*stage=(\d+)", line)
        if m:
            return int(m.group(1))
    return None


def parse_log(path: Path) -> dict | None:
    lines = path.read_text().splitlines()
    result = next((r for line in lines if (r := parse_bench_result(line))), None)
    if result is None:
        return None
    outputs = parse_output_values(lines)
    if outputs:
        result["output_values"] = outputs
    stage = parse_clock_stage(lines)
    if stage is not None:
        result["clock_stage"] = stage
    result["log_file"] = path.name
    return result


# Expected core clock per board. A run that silently fell back to a lower clock
# (e.g. the H753 PLL/VOS bring-up degrading to the 64 MHz HSI reset clock) must
# never be published as if it hit the rated speed - that exact bug shipped
# 64 MHz numbers mislabelled 480 once already.
EXPECTED_MHZ: dict[str, int] = {
    "nucleo_h753zi": 480,
    "nucleo_f446re": 180,
    "pico2_rp2350": 150,
}
CLOCK_OK_STAGE = 5  # bsp.c clock_init: stage 5 = target PLL locked and selected


def validate_clock(configs: list[dict], expect_mhz: int | None) -> list[str]:
    """Return a list of clock violations: cpu_mhz off the expected value, or a
    CLOCK_DIAG stage that is present but not the locked-PLL stage."""
    violations: list[str] = []
    for c in configs:
        board = c.get("board", "?")
        # A known board is ALWAYS validated against its own rated clock; the
        # --expect-mhz override only supplies a value for boards not in the table
        # (so it cannot accidentally hold a multi-board directory to one clock).
        exp = EXPECTED_MHZ.get(board, expect_mhz)
        log = c.get("log_file", "?")
        tag = f"{c.get('framework', '?')}/{c.get('kernel', '?')}"
        if exp is not None and c.get("cpu_mhz") != exp:
            violations.append(
                f"{log} [{tag}]: cpu_mhz={c.get('cpu_mhz')} != expected {exp} "
                f"- clock fell back below the rated speed")
        stage = c.get("clock_stage")
        if stage is not None and stage != CLOCK_OK_STAGE:
            violations.append(
                f"{log} [{tag}]: CLOCK_DIAG stage={stage} != {CLOCK_OK_STAGE} "
                f"- target clock was not reached")
    return violations


def collect(path: Path) -> list[dict]:
    if path.is_dir():
        return [r for p in sorted(path.glob("*.log")) if (r := parse_log(p))]
    r = parse_log(path)
    return [r] if r else []


def render_table(configs: list[dict]) -> Table:
    table = Table(title="Cortex-M deployability results", title_style="bold")
    table.add_column("Config", style="cyan", no_wrap=True)
    table.add_column("Board", no_wrap=True)
    table.add_column("MHz", justify="right", no_wrap=True)
    table.add_column("Latency (ms)", justify="right", no_wrap=True)
    table.add_column("Median cyc", justify="right", no_wrap=True)
    table.add_column("RAM peak (KB)", justify="right", no_wrap=True)
    table.add_column("Flash (KB)", justify="right", no_wrap=True)

    for c in configs:
        framework = c.get("framework", "?")
        dtype = c.get("dtype", "?")
        kernel = c.get("kernel", "?")
        model = c.get("model", "")
        base = model.removesuffix("_i8") or "?"
        name = f"{framework} {base} {dtype} ({kernel})"

        # sram_peak_bytes is the MEASURED runtime working set (TiGrIS: fast+slow
        # high-water + scratch + metadata; TFLM: arena_used_bytes) - the honest,
        # apples-to-apples RAM figure, not a provisioned-budget field.
        peak_b = c.get("sram_peak_bytes")
        if peak_b is not None:
            ram = f"{peak_b / 1024:.1f}"
        else:
            ram = c.get("sram_actual_kb", c.get("sram_kb", c.get("arena_kb", "?")))
        flash = c.get("plan_flash_kb", c.get("model_flash_kb", "?"))

        if c.get("status") not in (None, "ok"):
            latency = f"[red]{c.get('status')}[/red]"
            cyc = "-"
        else:
            # Report MEDIAN (the documented statistic) for both the ms and the
            # cycle column; stdev shows the spread.
            median = c.get("latency_median_ms", 0)
            stdev = c.get("latency_stdev_ms", 0)
            latency = f"{median:.2f} ± {stdev:.2f}"
            cyc = str(c.get("latency_median_cycles", "-"))

        table.add_row(name, str(c.get("board", "?")), str(c.get("cpu_mhz", "?")),
                      latency, cyc, str(ram), str(flash))
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse benchmark logs and print a results table")
    parser.add_argument("path", type=Path, help="Directory of .log files or a single log file")
    parser.add_argument("-o", "--output", type=Path, help="Also write summary JSON to this path")
    parser.add_argument("--expect-mhz", type=int, default=None,
                        help="Required core clock in MHz (default: per-board). A run off "
                             "this clock is rejected so degraded-clock numbers can't be published.")
    parser.add_argument("--no-clock-guard", action="store_true",
                        help="Disable the clock guard (not recommended).")
    args = parser.parse_args()

    configs = collect(args.path)
    console = Console()

    if not configs:
        console.print("[yellow]No BENCH_RESULT lines found.[/yellow]")
        return

    console.print(render_table(configs))

    # Clock guard: refuse to emit a summary from a run whose core clock silently
    # fell back below the rated speed. Validate BEFORE writing, so summary.json
    # never contains off-clock numbers.
    if not args.no_clock_guard:
        violations = validate_clock(configs, args.expect_mhz)
        if violations:
            console.print("\n[bold red]CLOCK GUARD: rejecting degraded-clock run(s):[/bold red]")
            for v in violations:
                console.print(f"  [red]✗[/red] {v}")
            console.print("[red]Off-clock numbers are not publishable. Re-flash/re-run, or pass "
                          "--expect-mhz / --no-clock-guard to override.[/red]")
            raise SystemExit(1)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"configs": configs, "count": len(configs)}, indent=2) + "\n")
        console.print(f"\nWrote {len(configs)} results to {args.output}")


if __name__ == "__main__":
    main()
