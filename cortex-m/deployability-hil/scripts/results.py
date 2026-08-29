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
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_REVISION_RE = re.compile(r"[0-9a-f]{40}")
PROVENANCE_PREFIX = "BENCH_PROVENANCE:"
COMMON_PROVENANCE_FIELDS = {
    "repositories",
    "dependencies",
    "tools",
    "host_model_environment",
    "siliconrig_sdk_version",
}


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


def validate_capture_provenance(value: object, board: object) -> list[str]:
    """Validate the evidence appended by run_all.sh before it can be published."""
    if not isinstance(value, dict):
        return ["record must be a JSON object"]
    errors: list[str] = []
    required = COMMON_PROVENANCE_FIELDS | {
        "captured_at_utc", "build", "artifacts", "board",
    }
    missing = sorted(required - value.keys())
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")

    timestamp = value.get("captured_at_utc")
    if not isinstance(timestamp, str):
        errors.append("captured_at_utc must be an RFC 3339 string")
    else:
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("timezone missing")
        except ValueError:
            errors.append("captured_at_utc must include a valid timezone")

    repositories = value.get("repositories")
    expected_repositories = {
        "benchmark", "tigris_compiler", "tigris_runtime", "tflite_micro",
    }
    if not isinstance(repositories, dict):
        errors.append("repositories must be an object")
    else:
        for name in sorted(expected_repositories):
            state = repositories.get(name)
            if not isinstance(state, dict):
                errors.append(f"repositories.{name} must be an object")
                continue
            if not GIT_REVISION_RE.fullmatch(str(state.get("revision", ""))):
                errors.append(f"repositories.{name}.revision must be a full Git SHA")
            if state.get("dirty") is not False:
                errors.append(f"repositories.{name}.dirty must be false")

    dependencies = value.get("dependencies")
    expected_dependencies = {
        "CMSIS-NN", "CMSIS-Core", "cmsis-device-f4", "cmsis-device-h7",
    }
    if not isinstance(dependencies, dict):
        errors.append("dependencies must be an object")
    else:
        for name in sorted(expected_dependencies):
            if not GIT_REVISION_RE.fullmatch(str(dependencies.get(name, ""))):
                errors.append(f"dependencies.{name} must be a full Git SHA")

    tools = value.get("tools")
    if not isinstance(tools, dict):
        errors.append("tools must be an object")
    else:
        for name in ("arm_none_eabi_gcc", "cmake"):
            if not isinstance(tools.get(name), str) or not tools[name]:
                errors.append(f"tools.{name} must be a non-empty string")
        pico_revision = tools.get("pico_sdk_revision")
        if board == "pico2_rp2350":
            if not GIT_REVISION_RE.fullmatch(str(pico_revision or "")):
                errors.append("tools.pico_sdk_revision must be a full Git SHA for RP2350")
        elif pico_revision is not None and not GIT_REVISION_RE.fullmatch(
                str(pico_revision)):
            errors.append("tools.pico_sdk_revision must be null or a full Git SHA")

    environment = value.get("host_model_environment")
    if not isinstance(environment, dict):
        errors.append("host_model_environment must be an object")
    else:
        if not SHA256_RE.fullmatch(str(environment.get("requirements_sha256", ""))):
            errors.append(
                "host_model_environment.requirements_sha256 must be a SHA-256")
        packages = environment.get("packages")
        if not isinstance(packages, dict) or not packages:
            errors.append("host_model_environment.packages must be a non-empty object")
        elif any(not isinstance(name, str) or not isinstance(version, str)
                 or not name or not version for name, version in packages.items()):
            errors.append("host_model_environment.packages entries must be strings")

    if (not isinstance(value.get("siliconrig_sdk_version"), str)
            or not value["siliconrig_sdk_version"]):
        errors.append("siliconrig_sdk_version must be a non-empty string")

    build = value.get("build")
    if (not isinstance(build, dict)
            or not isinstance(build.get("configure_command"), str)
            or not build["configure_command"]):
        errors.append("build.configure_command must be a non-empty string")

    artifacts = value.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("artifacts must be an object")
    else:
        for kind in ("model", "firmware"):
            artifact = artifacts.get(kind)
            if not isinstance(artifact, dict):
                errors.append(f"artifacts.{kind} must be an object")
                continue
            if not isinstance(artifact.get("name"), str) or not artifact["name"]:
                errors.append(f"artifacts.{kind}.name must be a non-empty string")
            if not SHA256_RE.fullmatch(str(artifact.get("sha256", ""))):
                errors.append(f"artifacts.{kind}.sha256 must be a SHA-256")
            size = artifact.get("size_bytes")
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                errors.append(f"artifacts.{kind}.size_bytes must be positive")

    board_record = value.get("board")
    if not isinstance(board_record, dict):
        errors.append("board must be an object")
    else:
        if (not isinstance(board_record.get("siliconrig_board_id"), str)
                or not board_record["siliconrig_board_id"]):
            errors.append("board.siliconrig_board_id must be a non-empty string")
        if not isinstance(board_record.get("board_type"), str):
            errors.append("board.board_type must be a string")
        if "specs" not in board_record:
            errors.append("board.specs must be recorded")
    return errors


def parse_capture_provenance(lines: list[str]) -> object | None:
    records = [
        line[len(PROVENANCE_PREFIX):]
        for line in lines if line.startswith(PROVENANCE_PREFIX)
    ]
    if not records:
        return None
    if len(records) != 1:
        raise ValueError(f"expected one {PROVENANCE_PREFIX} record, found {len(records)}")
    try:
        return json.loads(records[0])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {PROVENANCE_PREFIX} JSON: {exc}") from exc


def parse_log(path: Path, require_provenance: bool = False) -> dict | None:
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
    provenance = parse_capture_provenance(lines)
    if provenance is None:
        if require_provenance:
            raise ValueError(f"{path.name}: missing {PROVENANCE_PREFIX} record")
    else:
        errors = validate_capture_provenance(provenance, result.get("board"))
        if errors:
            raise ValueError(
                f"{path.name}: invalid capture provenance: {'; '.join(errors)}")
        result["_capture_provenance"] = provenance
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


def collect(path: Path, require_provenance: bool = False) -> list[dict]:
    if path.is_dir():
        return [
            r for p in sorted(path.glob("*.log"))
            if (r := parse_log(p, require_provenance))
        ]
    r = parse_log(path, require_provenance)
    return [r] if r else []


def extract_summary_provenance(configs: list[dict], required: bool) -> dict | None:
    """Deduplicate run-wide evidence while retaining per-firmware evidence."""
    records: dict[str, dict] = {}
    for config in configs:
        provenance = config.pop("_capture_provenance", None)
        if provenance is not None:
            records[config["log_file"]] = provenance

    if not records:
        if required:
            raise ValueError("no capture provenance records were collected")
        return None
    if len(records) != len(configs):
        raise ValueError("cannot mix legacy and provenance-bearing captures")

    first = next(iter(records.values()))
    common = {field: first[field] for field in sorted(COMMON_PROVENANCE_FIELDS)}
    cells: dict[str, dict] = {}
    for log_file, provenance in records.items():
        candidate = {
            field: provenance[field] for field in sorted(COMMON_PROVENANCE_FIELDS)
        }
        if candidate != common:
            raise ValueError(
                f"{log_file}: run-wide provenance differs from other captures")
        cells[log_file] = {
            key: value for key, value in provenance.items()
            if key not in COMMON_PROVENANCE_FIELDS
        }
    return {
        "source": PROVENANCE_PREFIX.removesuffix(":"),
        "common": common,
        "cells": cells,
    }


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
    parser.add_argument(
        "--require-provenance", action="store_true",
        help="Reject logs without complete build, artifact, board, and tool provenance.")
    args = parser.parse_args()

    console = Console()
    try:
        configs = collect(args.path, args.require_provenance)
        provenance = extract_summary_provenance(
            configs, args.require_provenance)
    except ValueError as exc:
        console.print(f"[bold red]PROVENANCE: {exc}[/bold red]")
        raise SystemExit(1) from exc

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
        document = {"configs": configs, "count": len(configs)}
        if provenance is not None:
            document["provenance"] = provenance
        args.output.write_text(json.dumps(document, indent=2) + "\n")
        console.print(f"\nWrote {len(configs)} results to {args.output}")


if __name__ == "__main__":
    main()
