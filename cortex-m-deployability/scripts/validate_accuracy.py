#!/usr/bin/env python3
"""Device-to-device parity for the Cortex-M deployability benchmark.

Parity gates the whole benchmark: a model that runs fast but wrong is a
non-result. The ONLY valid comparison (project policy) is TiGrIS and TFLM both on
CMSIS-NN, on the SAME board, comparing the int8 OUTPUT_I8 vectors the devices
actually produced. The host ORT / tflite interpreter is NOT a valid baseline (it
uses different kernels and differs from on-device CMSIS-NN by ~1 LSB).

This reads the captured serial logs, groups them by (board, model), and checks:
  - tigris/cmsis_nn  vs  tflm/cmsis_nn   the headline device-to-device parity
                                         (must be bit-exact: same CMSIS-NN kernels,
                                         same int8 weights via the matched plan)
  - tigris/s8_ref    vs  tigris/cmsis_nn reference-vs-optimized self-consistency
                                         (allowed +-1 LSB: the s8 requant nudge)

A pair is only checked when both members are present; a missing TFLM baseline for
a (board, model) is reported as INCOMPLETE, not silently passed.

Usage:
    python validate_accuracy.py results/raw/
    python validate_accuracy.py results/raw/ --tol 0 --self-tol 1
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

_KV = re.compile(r"(\w+)=([^,]+)")


def parse_log(path: Path) -> dict | None:
    """Pull framework/kernel/board/model/status + the OUTPUT_I8 vector from a log."""
    text = path.read_text(errors="replace").replace("\r\n", "\n")
    m = re.search(r"^BENCH_RESULT:(.*)$", text, re.MULTILINE)
    if not m:
        return None
    fields = dict(_KV.findall(m.group(1)))
    out: list[int] = []
    for line in re.findall(r"OUTPUT_I8:([^\n]*)", text):
        out.extend(int(v) for v in line.split())
    model = fields.get("model", "?")
    return {
        "log": path.name,
        "framework": fields.get("framework", "?"),
        "kernel": fields.get("kernel", "?"),
        "board": fields.get("board", "?"),
        "model": model,
        # TiGrIS runs the reconstructed plan (model "ad_matched"); TFLM runs the
        # source tflite (model "ad"). They are the same network - group by base so
        # the device-to-device pair lines up.
        "model_base": model.removesuffix("_matched"),
        "status": fields.get("status", "?"),
        "output": out,
    }


def compare(a: dict, b: dict, tol: int) -> tuple[str, str]:
    """Return (status, message) for an int8 output-vector comparison."""
    va, vb = a["output"], b["output"]
    if not va or not vb:
        return "skip", "missing OUTPUT_I8"
    if len(va) != len(vb):
        return "fail", f"length mismatch: {len(va)} vs {len(vb)}"
    diff = np.abs(np.array(va, np.int32) - np.array(vb, np.int32))
    max_abs = int(diff.max())
    n_diff = int((diff > 0).sum())
    msg = f"max_abs_diff={max_abs}, differing={n_diff}/{len(va)}"
    return ("pass" if max_abs <= tol else "fail"), msg


def main() -> None:
    ap = argparse.ArgumentParser(description="device-to-device parity")
    ap.add_argument("raw", type=Path, nargs="?", default=Path("results/raw"),
                    help="dir of captured *.log files (default results/raw/)")
    ap.add_argument("--tol", type=int, default=0,
                    help="max allowed abs diff for tigris-cmsis vs tflm (default 0, bit-exact)")
    ap.add_argument("--self-tol", type=int, default=1,
                    help="max allowed abs diff for s8 vs cmsis (default 1, requant nudge)")
    args = ap.parse_args()

    logs = [r for p in sorted(args.raw.glob("*.log")) if (r := parse_log(p))]
    # Group by (board, model) -> {(framework, kernel): record}. Last log wins per cell.
    groups: dict[tuple[str, str], dict[tuple[str, str], dict]] = {}
    for r in logs:
        if r["status"] != "ok":
            continue
        groups.setdefault((r["board"], r["model_base"]), {})[(r["framework"], r["kernel"])] = r

    print("Device-to-device parity (TiGrIS vs TFLM, both CMSIS-NN; + s8 self-check)\n")
    all_ok = True
    any_cell = False
    for (board, model), cell in sorted(groups.items()):
        tg_c = cell.get(("tigris", "cmsis_nn"))
        tg_s = cell.get(("tigris", "s8_ref"))
        tflm = cell.get(("tflm", "cmsis_nn"))
        print(f"  {board} / {model}")

        if tg_c and tflm:
            any_cell = True
            st, msg = compare(tg_c, tflm, args.tol)
            print(f"    [{'OK' if st == 'pass' else st.upper()}] tigris/cmsis vs tflm/cmsis : {msg}")
            all_ok &= st != "fail"
        else:
            print(f"    [INCOMPLETE] tigris/cmsis vs tflm/cmsis : "
                  f"{'no TFLM baseline' if tg_c else 'no TiGrIS cmsis'}")

        if tg_c and tg_s:
            st, msg = compare(tg_s, tg_c, args.self_tol)
            print(f"    [{'OK' if st == 'pass' else st.upper()}] tigris/s8 vs tigris/cmsis  : {msg}")
            all_ok &= st != "fail"

    print()
    if not any_cell:
        print("No (board, model) cell had BOTH a TiGrIS-cmsis and a TFLM run to compare.")
        raise SystemExit(1)
    if all_ok:
        print("All device-to-device parity checks passed.")
    else:
        print("Some parity checks FAILED.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
