#!/usr/bin/env python3
"""Compile the benchmark's TFLite models into fresh TiGrIS plans for one HIL run.

Each plan is compiled from the exact .tflite file the TFLM cell embeds, so both
frameworks run the same weights. Models listed in EXPORT_MODELS are also
rewritten with `tigris export` for the tflm_export cells, which run the result
on stock TFLM; it is written as <model>_export.tflite plus the C header the TFLM
harness includes.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess


PLAN_SPECS = {
    # Non-tiled budgets are kept close to measured activation demand. A
    # needlessly large budget suppresses compaction and inflates high-water RAM.
    "ds_cnn": ("ds_cnn_i8.tflite", "20K"),
    "ad": ("ad_i8.tflite", "2K"),
    "ts": ("ts_i8.tflite", "2K"),
    "mbv2": ("mbv2_a35_r224_i8.tflite", "128K"),
}
# `tigris export` writes no model for ds_cnn, ad and ts: no rewrite lowers
# their TFLM arena.
EXPORT_MODELS = ("mbv2",)


def write_c_header(model: pathlib.Path, header: pathlib.Path, symbol: str) -> None:
    """The TFLM harness includes <symbol>.h defining <symbol>[] and <symbol>_len."""
    data = model.read_bytes()
    lines = [f"/* Generated from {model.name} - do not edit. */",
             f"alignas(16) const unsigned char {symbol}[] = {{"]
    for i in range(0, len(data), 12):
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in data[i:i + 12]) + ",")
    lines += ["};", f"const unsigned int {symbol}_len = {len(data)};", ""]
    header.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", required=True, type=pathlib.Path)
    parser.add_argument("--models-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("models", nargs="+", choices=sorted(PLAN_SPECS))
    args = parser.parse_args()

    if not args.compiler.is_file():
        parser.error(f"TiGrIS compiler not found: {args.compiler}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        source_name, memory = PLAN_SPECS[model]
        source = args.models_dir / source_name
        if not source.is_file():
            parser.error(f"TFLite source not found: {source}")
        plan = args.output_dir / f"{model}.tgrs"
        print(f"  compile {model}: {source.name} -> {plan.name} ({memory})")
        subprocess.run(
            [str(args.compiler), "compile", str(source), "-m", memory,
             "-o", str(plan)],
            check=True,
        )
        if model in EXPORT_MODELS:
            exported = args.output_dir / f"{model}_export.tflite"
            print(f"  export {model}: {source.name} -> {exported.name}")
            subprocess.run(
                [str(args.compiler), "export", str(source), "-o", str(exported)],
                check=True,
            )
            write_c_header(exported, args.output_dir / f"{model}_export_tflite_i8.h",
                           f"{model}_export_tflite_i8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
