#!/usr/bin/env python3
"""Validate clone-local JSON artifacts without regenerating hardware results."""

from __future__ import annotations

import json
from pathlib import Path

from provenance_validation import (
    all_source_captures_present,
    git_tracked_paths,
    validate_provenance,
)


ROOT = Path(__file__).resolve().parents[1]
CORTEX_README = Path("cortex-m-deployability/README.md")
CORTEX_SUMMARY = Path("cortex-m-deployability/results/summary.json")
CORTEX_EXPECTED_MATRIX = Path(
    "cortex-m-deployability/results/expected-matrix.json")
CORTEX_PROVENANCE = Path("cortex-m-deployability/results/provenance.json")
IDENTITY_FIELDS = ("board", "model", "framework", "kernel")
EXPECTED_MHZ = {
    "nucleo_h753zi": 480,
    "nucleo_f446re": 180,
    "pico2_rp2350": 150,
}
MODEL_LABELS = {
    "DS-CNN": ("ds_cnn_matched", "ds_cnn"),
    "AD": ("ad_matched", "ad"),
    "TS": ("ts_matched", "ts"),
}


def reject_nonstandard_number(value: str) -> None:
    raise ValueError(f"non-standard JSON number {value}")


def format_identity(identity: tuple[str, ...]) -> str:
    return "/".join(identity)


def validate_expected_matrix(document: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(document, dict):
        return ["top level must be an object"]
    if document.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if document.get("suite") != "cortex-m-deployability":
        errors.append("suite must be 'cortex-m-deployability'")

    cells = document.get("cells")
    if not isinstance(cells, list):
        return errors + ["cells must be a list"]
    if not cells:
        errors.append("cells must not be empty")

    expected_fields = set(IDENTITY_FIELDS) | {"expected_status"}
    seen: set[tuple[str, ...]] = set()
    for index, cell in enumerate(cells):
        tag = f"cells[{index}]"
        if not isinstance(cell, dict):
            errors.append(f"{tag} must be an object")
            continue
        if set(cell) != expected_fields:
            errors.append(
                f"{tag} fields={sorted(cell)}, expected {sorted(expected_fields)}")
            continue
        if any(not isinstance(cell[field], str) for field in expected_fields):
            errors.append(f"{tag} fields must all be strings")
            continue

        identity = tuple(cell[field] for field in IDENTITY_FIELDS)
        if identity in seen:
            errors.append(f"{tag} duplicates matrix cell {format_identity(identity)}")
        seen.add(identity)
        if cell["expected_status"] not in {"ok", "ARENA_TOO_SMALL"}:
            errors.append(
                f"{tag} has unsupported expected_status "
                f"{cell['expected_status']!r}")
    return errors


def tracked_json_paths(tracked_paths: set[Path]) -> list[Path]:
    paths = {path for path in tracked_paths if path.suffix == ".json"}
    # Include required artifacts while they are still untracked in a working
    # tree; once committed, the set operation naturally de-duplicates them.
    paths.update((CORTEX_SUMMARY, CORTEX_EXPECTED_MATRIX, CORTEX_PROVENANCE))
    return sorted(paths)


def collect_matrix_cells(
        document: object,
        collection_field: str,
        status_field: str,
        source: str,
) -> tuple[dict[tuple[str, ...], str], list[str]]:
    cells: dict[tuple[str, ...], str] = {}
    errors: list[str] = []
    if not isinstance(document, dict):
        return cells, errors
    records = document.get(collection_field)
    if not isinstance(records, list):
        return cells, errors

    required = (*IDENTITY_FIELDS, status_field)
    for index, record in enumerate(records):
        if (not isinstance(record, dict)
                or any(not isinstance(record.get(field), str)
                       for field in required)):
            continue
        identity = tuple(record[field] for field in IDENTITY_FIELDS)
        if identity in cells:
            errors.append(
                f"duplicate {source} cell {format_identity(identity)} "
                f"at {collection_field}[{index}]")
            continue
        cells[identity] = record[status_field]
    return cells, errors


def validate_matrix_coverage(
        summary: object, expected_matrix: object) -> list[str]:
    actual, actual_errors = collect_matrix_cells(
        summary, "configs", "status", "summary")
    expected, expected_errors = collect_matrix_cells(
        expected_matrix, "cells", "expected_status", "expected matrix")
    errors = actual_errors + expected_errors

    for identity in sorted(expected.keys() - actual.keys()):
        errors.append(f"missing matrix cell {format_identity(identity)}")
    for identity in sorted(actual.keys() - expected.keys()):
        errors.append(f"unexpected matrix cell {format_identity(identity)}")
    for identity in sorted(actual.keys() & expected.keys()):
        if actual[identity] != expected[identity]:
            errors.append(
                f"matrix cell {format_identity(identity)} has status "
                f"{actual[identity]!r}, expected {expected[identity]!r}")
    return errors


def validate_removal_mutation(
        summary: object, expected_matrix: object) -> list[str]:
    if not isinstance(summary, dict):
        return ["removal mutation self-check requires an object summary"]
    configs = summary.get("configs")
    if not isinstance(configs, list) or not configs:
        return ["removal mutation self-check requires at least one config"]

    removed = configs[0]
    if (not isinstance(removed, dict)
            or any(not isinstance(removed.get(field), str)
                   for field in IDENTITY_FIELDS)):
        return ["removal mutation self-check could not identify removed config"]
    removed_identity = tuple(removed[field] for field in IDENTITY_FIELDS)
    mutated_summary = {**summary, "configs": configs[1:]}
    expected_error = f"missing matrix cell {format_identity(removed_identity)}"
    mutation_errors = validate_matrix_coverage(mutated_summary, expected_matrix)
    if expected_error not in mutation_errors:
        return [
            "removal mutation self-check failed to reject removed cell "
            f"{format_identity(removed_identity)}"
        ]
    return []


def validate_cortex_summary(document: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(document, dict):
        return ["top level must be an object"]

    configs = document.get("configs")
    has_embedded_provenance = isinstance(document.get("provenance"), dict)
    if not isinstance(configs, list):
        return ["configs must be a list"]
    if document.get("count") != len(configs):
        errors.append(
            f"count={document.get('count')!r}, actual configs={len(configs)}")
    if not configs:
        errors.append("configs must not be empty")

    seen: set[tuple[object, ...]] = set()
    ok_cells = 0
    for index, config in enumerate(configs):
        tag = f"configs[{index}]"
        if not isinstance(config, dict):
            errors.append(f"{tag} must be an object")
            continue

        required = ("board", "model", "framework", "kernel", "dtype", "status")
        missing = [field for field in required if field not in config]
        if missing:
            errors.append(f"{tag} missing fields: {', '.join(missing)}")
            continue
        non_strings = [field for field in required
                       if not isinstance(config[field], str)]
        if non_strings:
            errors.append(
                f"{tag} fields must be strings: {', '.join(non_strings)}")
            continue

        identity = tuple(config[field] for field in IDENTITY_FIELDS)
        if identity in seen:
            errors.append(
                f"{tag} duplicates cell identity {format_identity(identity)}")
        seen.add(identity)

        if config["dtype"] != "int8":
            errors.append(f"{tag} dtype={config['dtype']!r}, expected 'int8'")

        board = config["board"]
        expected_mhz = EXPECTED_MHZ.get(board)
        if expected_mhz is None:
            errors.append(f"{tag} has unknown board {board!r}")
        elif config.get("cpu_mhz") != expected_mhz:
            errors.append(
                f"{tag} cpu_mhz={config.get('cpu_mhz')!r}, expected {expected_mhz}")
        clock_stage = config.get("clock_stage")
        if clock_stage is not None and clock_stage != 5:
            errors.append(f"{tag} clock_stage={clock_stage!r}, expected 5")

        status = config["status"]
        if status == "ARENA_TOO_SMALL":
            if config.get("output_values"):
                errors.append(f"{tag} OOM cell must not contain output values")
            continue
        if status != "ok":
            errors.append(f"{tag} has unexpected status {status!r}")
            continue

        ok_cells += 1
        runs = config.get("runs")
        if not isinstance(runs, int) or runs < 30:
            errors.append(f"{tag} runs={runs!r}, expected at least 30")
        for field in ("latency_median_ms", "latency_median_cycles", "sram_peak_bytes"):
            value = config.get(field)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or value <= 0):
                errors.append(f"{tag} {field} must be a positive number")
        if (has_embedded_provenance
                and config["framework"] == "tigris"):
            workspace = config.get("sram_executor_workspace_bytes")
            if not isinstance(workspace, int) or workspace <= 0:
                errors.append(
                    f"{tag} must count a positive executor workspace")

        output = (config.get("output_values") or {}).get("i8")
        if not isinstance(output, list) or not output:
            errors.append(f"{tag} must contain a non-empty OUTPUT_I8 vector")
        elif any(
                not isinstance(value, int) or isinstance(value, bool)
                or value < -128 or value > 127
                for value in output):
            errors.append(f"{tag} OUTPUT_I8 must contain integers in [-128, 127]")

    if ok_cells == 0:
        errors.append("summary contains no successful cells")
    return errors


def _result_cell(
        summary: dict[str, object],
        board: str,
        model: str,
        framework: str,
        kernel: str,
) -> dict[str, object]:
    configs = summary.get("configs")
    if not isinstance(configs, list):
        raise ValueError("summary configs must be a list")
    matches = [
        config for config in configs
        if isinstance(config, dict)
        and config.get("board") == board
        and config.get("model") == model
        and config.get("framework") == framework
        and config.get("kernel") == kernel
    ]
    if len(matches) != 1:
        raise ValueError(
            "expected one result for "
            f"{board}/{model}/{framework}/{kernel}, found {len(matches)}")
    return matches[0]


def _latency(cell: dict[str, object], decimals: int = 2) -> str:
    value = cell.get("latency_median_ms")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("result has no numeric median latency")
    return f"{value:.{decimals}f}"


def _ram(cell: dict[str, object]) -> str:
    value = cell.get("sram_peak_bytes")
    if not isinstance(value, int):
        raise ValueError("result has no integer RAM peak")
    return f"{value / 1024:.1f}"


def _cycles(cell: dict[str, object]) -> str:
    value = cell.get("latency_median_cycles")
    if not isinstance(value, int):
        raise ValueError("result has no integer median cycle count")
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} M"
    return f"{value / 1_000:.0f} K"


def _firmware_kb(
        summary: dict[str, object], cell: dict[str, object]) -> str:
    provenance = summary.get("provenance")
    log_file = cell.get("log_file")
    try:
        value = provenance["cells"][log_file]["artifacts"]["firmware"]["size_bytes"]
    except (KeyError, TypeError):
        raise ValueError(
            f"result {log_file!r} has no firmware size provenance") from None
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"result {log_file!r} has an invalid firmware size")
    return f"{value / 1024:.0f}"


def expected_readme_result_rows(
        document: object,
) -> tuple[list[str], list[str]]:
    if not isinstance(document, dict):
        return [], ["summary must be an object"]
    rows: list[str] = []
    try:
        board = "nucleo_h753zi"
        for label, (tigris_model, tflm_model) in MODEL_LABELS.items():
            for framework_label, framework, kernel, model in (
                ("TiGrIS", "tigris", "cmsis_nn", tigris_model),
                ("TFLM", "tflm", "cmsis_nn", tflm_model),
                ("TiGrIS", "tigris", "s8_ref", tigris_model),
            ):
                cell = _result_cell(
                    document, board, model, framework, kernel)
                latency_decimals = (
                    3 if cell["latency_median_ms"] < 1 else 2)
                rows.append(
                    f"| {framework_label} | {kernel} | "
                    f"{_latency(cell, latency_decimals)} ms | "
                    f"{_cycles(cell)} | {_ram(cell)} KB | "
                    f"{_firmware_kb(document, cell)} KB |")

        for label, (tigris_model, tflm_model) in reversed(MODEL_LABELS.items()):
            cmsis = _result_cell(
                document, "nucleo_f446re", tigris_model,
                "tigris", "cmsis_nn")
            tflm = _result_cell(
                document, "nucleo_f446re", tflm_model,
                "tflm", "cmsis_nn")
            s8 = _result_cell(
                document, "nucleo_f446re", tigris_model,
                "tigris", "s8_ref")
            rows.append(
                f"| {label} | {_latency(cmsis)} ms | {_latency(tflm)} ms | "
                f"{_latency(s8)} ms | {_ram(cmsis)} / {_ram(tflm)} KB |")

        for label, (tigris_model, _) in reversed(MODEL_LABELS.items()):
            cmsis = _result_cell(
                document, "pico2_rp2350", tigris_model,
                "tigris", "cmsis_nn")
            s8 = _result_cell(
                document, "pico2_rp2350", tigris_model,
                "tigris", "s8_ref")
            rows.append(
                f"| {label} | {_latency(cmsis)} ms | {_latency(s8)} ms | "
                f"{_ram(cmsis)} KB |")

        for board, label, tflm in (
            ("nucleo_h753zi", "H753ZI (512 KB)", "OOM at AllocateTensors"),
            ("pico2_rp2350", "RP2350 (520 KB)", "n/a (no M33 lib)"),
        ):
            cell = _result_cell(
                document, board, "mbv2_a35_r224_matched",
                "tigris", "cmsis_nn")
            seconds = cell["latency_median_ms"] / 1000
            rows.append(
                f"| {label} | runs, {seconds:.2f} s, {_ram(cell)} KB | "
                f"{tflm} |")
    except (KeyError, TypeError, ValueError) as exc:
        return [], [str(exc)]
    return rows, []


def validate_readme_results(document: object, readme: str) -> list[str]:
    rows, errors = expected_readme_result_rows(document)
    for row in rows:
        count = readme.count(row)
        if count != 1:
            errors.append(
                f"derived result row occurs {count} times; expected exactly once: "
                f"{row}")
    return errors


def main() -> None:
    tracked_paths = git_tracked_paths(ROOT)
    paths = tracked_json_paths(tracked_paths)
    errors: list[str] = []
    documents: dict[Path, object] = {}
    for relative in paths:
        path = ROOT / relative
        try:
            document = json.loads(
                path.read_text(), parse_constant=reject_nonstandard_number)
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append(f"{relative}: invalid JSON: {exc}")
            continue
        documents[relative] = document

    summary = documents.get(CORTEX_SUMMARY)
    expected_matrix = documents.get(CORTEX_EXPECTED_MATRIX)
    provenance = documents.get(CORTEX_PROVENANCE)
    summary_errors = (
        validate_cortex_summary(summary)
        if CORTEX_SUMMARY in documents else [])
    expected_errors = (
        validate_expected_matrix(expected_matrix)
        if CORTEX_EXPECTED_MATRIX in documents else [])
    errors.extend(
        f"{CORTEX_SUMMARY}: {problem}" for problem in summary_errors)
    errors.extend(
        f"{CORTEX_EXPECTED_MATRIX}: {problem}"
        for problem in expected_errors)
    if CORTEX_SUMMARY in documents:
        try:
            readme = (ROOT / CORTEX_README).read_text()
        except (OSError, UnicodeError) as exc:
            errors.append(f"{CORTEX_README}: cannot read: {exc}")
        else:
            errors.extend(
                f"{CORTEX_README}: {problem}"
                for problem in validate_readme_results(summary, readme))
    provenance_errors: list[str] = []
    reconstruction_checked = False
    if CORTEX_PROVENANCE in documents:
        provenance_errors = validate_provenance(
            provenance, ROOT, tracked_paths)
        errors.extend(
            f"{CORTEX_PROVENANCE}: {problem}"
            for problem in provenance_errors)
        reconstruction_checked = (
            not provenance_errors
            and all_source_captures_present(provenance, ROOT)
        )

    mutation_checked = False
    if (CORTEX_SUMMARY in documents
            and CORTEX_EXPECTED_MATRIX in documents
            and not summary_errors
            and not expected_errors):
        coverage_errors = validate_matrix_coverage(summary, expected_matrix)
        errors.extend(
            f"Cortex matrix contract: {problem}"
            for problem in coverage_errors)
        if not coverage_errors:
            mutation_errors = validate_removal_mutation(summary, expected_matrix)
            errors.extend(
                f"Cortex matrix contract: {problem}"
                for problem in mutation_errors)
            mutation_checked = not mutation_errors

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    if mutation_checked:
        print("Verified that removing an expected Cortex matrix cell fails.")
    if reconstruction_checked:
        print("Rebuilt the Cortex summary byte-for-byte from 27 source captures.")
    print(f"Validated {len(paths)} JSON artifact(s).")


if __name__ == "__main__":
    main()
