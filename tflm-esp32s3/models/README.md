# Model Provenance

## DS-CNN (Keyword Spotting)

Architecture from MLPerf Tiny benchmark suite. Identical to the DS-CNN used
in the TiGrIS test fixtures (`tigris/src/tigris/cli/fixtures.py:build_ds_cnn`).

- **Input**: [1, 1, 49, 10], NCHW mel spectrogram (1s audio @ 16kHz, 49 frames, 10 MFCC bins)
- **Output**: [1, 12], 12 keyword classes
- **Architecture**: Conv(10x4,s2) + BN + ReLU, then 4x [DWConv(3x3) + BN + ReLU, PW Conv(1x1) + BN + ReLU], GAP, FC(12)
- **Params**: ~26K (f32: ~104 KB weights)
- **Seed**: `np.random.seed(42)` for reproducible weights

## Variants

| File | Format | dtype | Notes |
|------|--------|-------|-------|
| `ds_cnn.onnx` | ONNX | f32 | Source model |
| `ds_cnn_i8.onnx` | ONNX | int8 | `onnxruntime.quantization.quantize_static` with MinMax calibration |
| `ds_cnn.tgrs` | TiGrIS | f32 | Compiled with `-m 256K` |
| `ds_cnn_i8.tgrs` | TiGrIS | int8 | Compiled with `-m 256K` |
| `ds_cnn.tflite` | TFLite | f32 | Keras conversion |
| `ds_cnn_i8.tflite` | TFLite | int8 | Full integer quantization |

## Regeneration

```bash
pip install -r ../requirements.txt
pip install tigris-ml
python prepare.py
```
