#!/usr/bin/env python3
"""Produce ONE matched DS-CNN model for the TiGrIS-vs-TFLM parity benchmark.

Single source of truth: ds_cnn_i8.tflite. From it we derive:
  - ds_cnn_tflite_i8.h     C array TFLM embeds  (regenerated so .h == .tflite,
                           fixing the integrity bug where they had diverged)
  - ds_cnn_matched.onnx    QDQ ONNX with the tflite's exact int8 weights+scales
  - ds_cnn_matched_64k.tgrs TiGrIS plan compiled from that ONNX
  - ds_cnn_matched_ref.bin golden int8 output = the tflite interpreter (int8=1)

Both frameworks then run the identical quantized model, validated against the
same reference -> a true weight-for-weight parity comparison.

  python prepare_matched.py [path/to/models/output]
Needs: tflite, onnx, numpy, tensorflow (interpreter + ref); tigris on PATH.
"""
import sys, subprocess, re
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent.parent / "tflm-esp32s3/models/output"
TFLITE = OUT / "ds_cnn_i8.tflite"
TOOL = HERE.parent / "tools/tflite_to_qdq_onnx.py"
assert TFLITE.exists(), f"missing {TFLITE}"


def gen_c_header(tflite: Path, header: Path, name: str):
    data = tflite.read_bytes()
    lines = [f"/* Auto-generated from {tflite.name}. Do not edit. */",
             f"const unsigned char {name}[] = {{"]
    for i in range(0, len(data), 12):
        lines.append("  " + "".join(f"0x{b:02x}, " for b in data[i:i + 12]).rstrip())
    lines += ["};", f"const unsigned int {name}_len = {len(data)};", ""]
    header.write_text("\n".join(lines))
    print(f"  header: {header.name} ({len(data)} bytes) -- now matches {tflite.name}")


def gen_reference(tflite: Path, ref: Path):
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=str(tflite)); it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    it.set_tensor(i["index"], np.ones(i["shape"], dtype=i["dtype"])); it.invoke()
    q = it.get_tensor(o["index"]).flatten().astype(np.int8)
    q.tofile(ref)
    print(f"  reference: {ref.name} (int8=1 input): {list(int(v) for v in q)}")


def main():
    print("Matched DS-CNN from", TFLITE.name)
    gen_c_header(TFLITE, OUT / "ds_cnn_tflite_i8.h", "ds_cnn_tflite_i8")
    onnx_p = OUT / "ds_cnn_matched.onnx"
    subprocess.run([sys.executable, str(TOOL), str(TFLITE), str(onnx_p)], check=True)
    for budget, name in [("256K", "ds_cnn_matched.tgrs"), ("64K", "ds_cnn_matched_64k.tgrs")]:
        subprocess.run(["tigris", "compile", str(onnx_p), "-m", budget, "-o", str(OUT / name)], check=True)
        print(f"  plan: {name} (-m {budget})")
    gen_reference(TFLITE, OUT / "ds_cnn_matched_ref.bin")
    print("Done.")


if __name__ == "__main__":
    main()
