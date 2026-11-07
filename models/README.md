# Model Provenance

## DS-CNN (Keyword Spotting)

Architecture from MLPerf Tiny benchmark suite. Identical to the DS-CNN used
in the TiGrIS test fixtures (`tigris/src/tigris/cli/fixtures.py:build_ds_cnn`).

- **Input**: [1, 1, 49, 10], NCHW mel spectrogram (1s audio @ 16kHz, 49 frames, 10 MFCC bins)
- **Output**: [1, 12], 12 keyword classes
- **Architecture**: Conv(10x4,s2) + BN + ReLU, then 4x [DWConv(3x3) + BN + ReLU, PW Conv(1x1) + BN + ReLU], GAP, FC(12)
- **Params**: ~26K (f32: ~104 KB weights)
- **Weights**: random and untrained. The ONNX build draws them with
  `np.random.seed(42)`; the Keras build, which produces the `.tflite` files, uses
  the default initializers under `tf.random.set_seed(42)`
- **Calibration**: int8 quantization uses 100 random-normal samples
  (`np.random.seed(42)`)

## Variants

| File | Format | dtype | Notes |
|------|--------|-------|-------|
| `ds_cnn.onnx` | ONNX | f32 | Source model |
| `ds_cnn_i8.onnx` | ONNX | int8 | `onnxruntime.quantization.quantize_static` with MinMax calibration |
| `ds_cnn.tgrs` | TiGrIS | f32 | Compiled with `-m 256K` |
| `ds_cnn_i8.tgrs` | TiGrIS | int8 | Compiled with `-m 256K` |
| `ds_cnn.tflite` | TFLite | f32 | Keras conversion |
| `ds_cnn_i8.tflite` | TFLite | int8 | Full integer quantization |
| `ds_cnn_reference_f32.bin` | raw vector | f32 | ONNX Runtime reference for TiGrIS |
| `ds_cnn_reference_i8.bin` | raw vector | int8 | ONNX Runtime reference for TiGrIS |
| `ds_cnn_tflite_reference_f32.bin` | raw vector | f32 | TFLite reference for TFLM |
| `ds_cnn_tflite_reference_i8.bin` | raw vector | int8 | TFLite reference for TFLM |
| `mobilenet_v1_reference_i8.bin` | raw vector | int8 | ONNX-quantized MobileNet reference |
| `mobilenet_v1_matched_ref.bin` | raw vector | int8 | TFLite-reconstructed MobileNet reference for TiGrIS |

## MobileNetV2 0.35 (image classification)

Built by `prepare_mobilenetv2.py` from Keras `applications.MobileNetV2`.

- **Input**: [1, 224, 224, 3], NHWC int8
- **Output**: [1, 10], int8 logits, no softmax
- **Architecture**: MobileNetV2 backbone at alpha 0.35 (35 CONV_2D,
  17 DEPTHWISE_CONV_2D, 10 inverted-residual ADD), global average pool, FC(10)
- **Params**: 401,888 (394,848 int8 weights, 7,040 int32 biases), 423,016 B of
  weights and biases
- **Weights**: the backbone carries the Keras ImageNet weights. The Dense(10)
  head is freshly initialized under `tf.random.set_seed(42)` and never trained,
  so the logits carry no class meaning
- **Calibration**: int8 conversion uses 100 random-normal samples
  (`np.random.seed(42)`), not images
- **Largest activation**: 1x112x112x48 int8, 602,112 B

| File | Format | dtype | Notes |
|------|--------|-------|-------|
| `mbv2_a35_r224_i8.tflite` | TFLite | int8 | Full integer quantization; the model both frameworks run on Cortex-M |
| `mbv2_tflite_i8.h` | C array | int8 | `mbv2_a35_r224_i8.tflite` for the TFLM harness |

## Regeneration

```bash
pip install -r ../requirements.txt
pip install tigris-ml
python prepare.py
python prepare_mobilenetv2.py   # downloads the ImageNet backbone weights once
```
