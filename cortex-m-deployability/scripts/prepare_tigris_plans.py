#!/usr/bin/env python3
"""Compile benchmark ONNX sources into fresh TiGrIS plans for one HIL run."""

from __future__ import annotations

import argparse
import pathlib
import subprocess


PLAN_SPECS = {
    "ds_cnn": ("ds_cnn_matched.onnx", "32K", "ds_cnn_matched_32k.tgrs"),
    "ad": ("ad_matched.onnx", "32K", "ad_matched_32k.tgrs"),
    "ts": ("ts_matched.onnx", "32K", "ts_matched_32k.tgrs"),
    "mbv2": ("mbv2_a35_r224_matched.onnx", "128K", "mbv2_a35_128k.tgrs"),
}


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
        source_name, memory, output_name = PLAN_SPECS[model]
        source = args.models_dir / source_name
        output = args.output_dir / output_name
        if not source.is_file():
            parser.error(f"matched ONNX source not found: {source}")
        print(f"  compile {model}: {source.name} -> {output.name} ({memory})")
        subprocess.run(
            [str(args.compiler), "compile", str(source), "-m", memory,
             "-o", str(output)],
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
