"""Slim large per-cell output vectors out of an ESP32-S3 summary before commit.

The device firmware dumps a model's entire output tensor. For classifiers that
is O(10) values, but a dense spatial output (e.g. the U-Net segmentation map,
256*256*8 = 524288 int8) would bloat the tracked summary.json by megabytes.

This step runs AFTER validate_accuracy.py has already checked the full device
vector against the model reference (see run_all.sh collect_and_validate), so the
numerical parity gate is unaffected. It replaces any output vector longer than
MAX_INLINE_OUTPUT with a short canary slice plus a digest (length + sha256) --
a few hundred bytes instead of megabytes. The canary keeps the cell valid under
benchmark_matrix.validate_cell (a non-empty int list) and doubles as a
tamper/regression tripwire; full-resolution parity stays reproducible
off-summary via host_parity_unet.py.

Cells whose outputs are all small (every existing classifier) are returned
byte-for-byte unchanged, so regenerating a U-Net-free summary is a no-op.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MAX_INLINE_OUTPUT = 4096  # keep classifier vectors inline; trim dense maps
CANARY_LEN = 12           # kept inline as a tripwire and to satisfy validate_cell


def _digest(values: list[int]) -> dict:
    """Length + sha256 of the output as raw two's-complement int8 bytes.

    The byte encoding matches how the model reference (.bin) is stored, so the
    digest is recomputable from a reference or a fresh capture.
    """
    raw = bytes((int(v) & 0xFF) for v in values)
    return {
        "len": len(values),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "canary_len": min(CANARY_LEN, len(values)),
    }


def slim_config(config: dict, max_inline: int = MAX_INLINE_OUTPUT) -> dict:
    """Replace any output_values vector longer than max_inline with a canary
    slice, recording its length and sha256 under config["output_digest"].

    Mutates and returns config. A cell with only small vectors (or no
    output_values, e.g. an ARENA_TOO_SMALL failure cell) is left unchanged.
    """
    outputs = config.get("output_values")
    if not isinstance(outputs, dict):
        return config
    for key, values in list(outputs.items()):
        if isinstance(values, list) and len(values) > max_inline:
            config.setdefault("output_digest", {})[key] = _digest(values)
            outputs[key] = values[:CANARY_LEN]
    return config


def slim_summary(summary: dict, max_inline: int = MAX_INLINE_OUTPUT) -> dict:
    """Slim every cell in a {"configs": [...]} summary in place."""
    for config in summary.get("configs", []):
        slim_config(config, max_inline)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="summary JSON to slim in place")
    parser.add_argument(
        "--max-inline", type=int, default=MAX_INLINE_OUTPUT,
        help=f"trim output vectors longer than this (default {MAX_INLINE_OUTPUT})")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    slim_summary(summary, args.max_inline)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
