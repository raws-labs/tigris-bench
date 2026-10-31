#!/usr/bin/env python3
"""Device-to-device parity for the Cortex-M deployability benchmark.

Parity gates the whole benchmark: a model that runs fast but wrong is a
non-result. The ONLY valid comparison (project policy) is TiGrIS and TFLM both on
CMSIS-NN, on the SAME board, comparing the int8 OUTPUT_I8 vectors the devices
actually produced. The host ORT / tflite interpreter is NOT a valid baseline (it
uses different kernels and differs from on-device CMSIS-NN by ~1 LSB).

This reads results/summary.json (the tracked artifact, which carries each cell's
OUTPUT_I8 vector, so parity is verifiable from a clone without the raw serial
logs), or a directory of raw *.log captures. It groups by (board, model) and checks:
  - tigris/cmsis_nn  vs  tflm/cmsis_nn   the headline device-to-device parity
                                         (must be bit-exact: same CMSIS-NN kernels,
                                         plan compiled from the same .tflite)
  - tflm_export      vs  tflm/cmsis_nn   and vs tigris/cmsis_nn: the `tigris
                                         export` rewrite on stock TFLM (model
                                         "<model>_export") must be bit-exact too
  - tigris/s8_ref    vs  tigris/cmsis_nn reference-vs-optimized self-consistency
                                         (allowed +-1 LSB: the s8 requant nudge)

A pair is only checked when both members are present; a missing TFLM baseline for
a (board, model) is reported as INCOMPLETE, not silently passed. A subset holding
only the two TiGrIS configs is still gated on the s8 self-check; the run fails
when no pair of either kind was comparable.

Usage:
    python validate_accuracy.py                 # reads results/summary.json
    python validate_accuracy.py results/raw/    # or a dir of raw logs
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
    return record(path.name, fields.get("framework", "?"), fields.get("kernel", "?"),
                  fields.get("board", "?"), fields.get("model", "?"),
                  fields.get("status", "?"), out)


def record(log: str, framework: str, kernel: str, board: str, model: str,
           status: str, output: list[int]) -> dict:
    # The TFLM cell running the `tigris export` rewrite of a model reports
    # "<model>_export"; it is grouped with its source model as framework
    # "tflm_export" so every pair of the same network lines up.
    if framework == "tflm" and model.endswith("_export"):
        framework, model = "tflm_export", model.removesuffix("_export")
    return {"log": log, "framework": framework, "kernel": kernel, "board": board,
            "model": model, "status": status, "output": output}


def compare(a: dict, b: dict, tol: int) -> tuple[str, str]:
    """Return (status, message) for an int8 output-vector comparison."""
    va, vb = a["output"], b["output"]
    # Both records reported a successful inference, so an absent vector is
    # missing evidence, not a reason to skip the comparison.
    if not va or not vb:
        missing = " and ".join(r["log"] for r in (a, b) if not r["output"])
        return "fail", f"successful record without OUTPUT_I8 ({missing})"
    if len(va) != len(vb):
        return "fail", f"length mismatch: {len(va)} vs {len(vb)}"
    diff = np.abs(np.array(va, np.int32) - np.array(vb, np.int32))
    max_abs = int(diff.max())
    n_diff = int((diff > 0).sum())
    msg = f"max_abs_diff={max_abs}, differing={n_diff}/{len(va)}"
    return ("pass" if max_abs <= tol else "fail"), msg


def records_from_summary(path: Path) -> list[dict]:
    """Build records from a results.py summary.json so parity is verifiable from a
    clone (the tracked artifact carries each cell's OUTPUT_I8), no raw logs needed."""
    import json
    return [
        record(path.name, c.get("framework", "?"), c.get("kernel", "?"),
               c.get("board", "?"), c.get("model", "?"), c.get("status", "?"),
               (c.get("output_values") or {}).get("i8", []))
        for c in json.loads(path.read_text()).get("configs", [])
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="device-to-device parity")
    ap.add_argument("src", type=Path, nargs="?", default=Path("results/summary.json"),
                    help="results.py summary.json (default) or a dir of captured *.log files")
    ap.add_argument("--tol", type=int, default=0,
                    help="max allowed abs diff for tigris-cmsis vs tflm (default 0, bit-exact)")
    ap.add_argument("--self-tol", type=int, default=1,
                    help="max allowed abs diff for s8 vs cmsis (default 1, requant nudge)")
    args = ap.parse_args()

    if args.src.is_dir():
        logs = [r for p in sorted(args.src.glob("*.log")) if (r := parse_log(p))]
    else:
        logs = records_from_summary(args.src)
    # Group by (board, model) -> {(framework, kernel): record}. Last log wins per cell.
    groups: dict[tuple[str, str], dict[tuple[str, str], dict]] = {}
    for r in logs:
        if r["status"] != "ok":
            continue
        groups.setdefault((r["board"], r["model"]), {})[(r["framework"], r["kernel"])] = r

    print("Device-to-device parity (TiGrIS vs TFLM, both CMSIS-NN; + s8 self-check)\n")
    all_ok = True
    n_cross = 0
    n_self = 0
    for (board, model), cell in sorted(groups.items()):
        tg_c = cell.get(("tigris", "cmsis_nn"))
        tg_s = cell.get(("tigris", "s8_ref"))
        tflm = cell.get(("tflm", "cmsis_nn"))
        export = cell.get(("tflm_export", "cmsis_nn"))
        print(f"  {board} / {model}")

        if tg_c and tflm:
            n_cross += 1
            st, msg = compare(tg_c, tflm, args.tol)
            print(f"    [{'OK' if st == 'pass' else st.upper()}] tigris/cmsis vs tflm/cmsis : {msg}")
            all_ok &= st == "pass"
        else:
            print(f"    [INCOMPLETE] tigris/cmsis vs tflm/cmsis : "
                  f"{'no TFLM baseline' if tg_c else 'no TiGrIS cmsis'}")

        for label, other in (("tflm/cmsis ", tflm), ("tigris/cmsis", tg_c)):
            if export and other:
                n_cross += 1
                st, msg = compare(export, other, args.tol)
                print(f"    [{'OK' if st == 'pass' else st.upper()}] "
                      f"tflm_export vs {label}: {msg}")
                all_ok &= st == "pass"

        if tg_c and tg_s:
            n_self += 1
            st, msg = compare(tg_s, tg_c, args.self_tol)
            print(f"    [{'OK' if st == 'pass' else st.upper()}] tigris/s8 vs tigris/cmsis  : {msg}")
            all_ok &= st == "pass"

    print()
    if n_cross == 0 and n_self == 0:
        print("No (board, model) cell held a comparable pair.")
        raise SystemExit(1)
    if all_ok:
        checked = []
        if n_cross:
            checked.append(f"{n_cross} cross-framework")
        if n_self:
            checked.append(f"{n_self} s8 self-check")
        print(f"All device-to-device parity checks passed ({', '.join(checked)}).")
    else:
        print("Some parity checks FAILED.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
