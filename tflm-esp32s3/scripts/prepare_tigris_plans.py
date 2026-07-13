#!/usr/bin/env python3
"""Compile the ESP32-S3 suite's TiGrIS plans for the current compiler/runtime."""

from __future__ import annotations

import argparse
import pathlib
import subprocess


# These are inputs to the benchmark, not versioned plan artifacts.  Rebuilding
# them for every device run prevents a benchmark from silently exercising an
# obsolete compiler/runtime contract.
PLAN_SPECS = (
    ("ds_cnn.onnx", "256K", "ds_cnn.tgrs"),
    ("ds_cnn_i8.onnx", "256K", "ds_cnn_i8.tgrs"),
    ("mobilenet_v1_i8.onnx", "256K", "mobilenet_v1_i8.tgrs"),
    ("mobilenet_v1_i8.onnx", "128K", "mobilenet_v1_i8_128k.tgrs"),
    ("mobilenet_v1_i8.onnx", "64K", "mobilenet_v1_i8_64k.tgrs"),
    ("mobilenet_v1_i8.onnx", "32K", "mobilenet_v1_i8_32k.tgrs"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", required=True, type=pathlib.Path)
    parser.add_argument("--models-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    if not args.compiler.is_file():
        parser.error(f"TiGrIS compiler not found: {args.compiler}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for source_name, memory, output_name in PLAN_SPECS:
        source = args.models_dir / source_name
        output = args.output_dir / output_name
        if not source.is_file():
            parser.error(f"ONNX source not found: {source}")
        print(f"  compile {source_name} -> {output_name} ({memory})")
        subprocess.run(
            [str(args.compiler), "compile", str(source), "-m", memory,
             "-o", str(output)],
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
