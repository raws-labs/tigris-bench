#!/usr/bin/env python3
"""Fetch the trained MobileNetV2 for the Cortex-M deployability benchmark.

STMicroelectronics' ImageNet-trained MobileNetV2 alpha 0.35 at 224x224 from
stm32ai-modelzoo, a full 1000-class INT8 classifier: uint8 input, float32
softmax output, 1,905,936 B. Its naive activation peak is 735 KiB, like the
untrained `mbv2`: TFLite Micro fails AllocateTensors on the H753 at 480 KiB,
and TiGrIS tiles it there.

The file is pinned to a release tag of the model zoo and checked by SHA-256.

Usage: python prepare_st_mbv2.py
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
sys.path.insert(0, str(HERE))
from prepare import generate_c_header  # noqa: E402

URL = ("https://media.githubusercontent.com/media/STMicroelectronics/"
       "stm32ai-modelzoo/v4.1.0/image_classification/mobilenetv2/"
       "Public_pretrainedmodel_public_dataset/ImageNet/mobilenetv2_a035_224/"
       "mobilenetv2_a035_224_int8.tflite")
SHA256 = "194192ddbf615a2c2ffb842a4370e8ff88fa6547958137e97cb09aaf44747067"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tflite = OUT / "st_mbv2_a35_r224_i8.tflite"
    if not tflite.is_file() or hashlib.sha256(tflite.read_bytes()).hexdigest() != SHA256:
        print(f"Fetching {URL}")
        with urllib.request.urlopen(URL, timeout=120) as response:
            data = response.read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != SHA256:
            raise SystemExit(f"SHA-256 mismatch: got {digest}, expected {SHA256}")
        tflite.write_bytes(data)
    generate_c_header(tflite, OUT / "st_mbv2_tflite_i8.h", "st_mbv2_tflite_i8")
    print(f"Wrote {tflite.name} and st_mbv2_tflite_i8.h")


if __name__ == "__main__":
    main()
