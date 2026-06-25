/*
 * TFLite Micro baseline for the Cortex-M deployability benchmark.
 *
 * Runs an INT8 .tflite through TFLM's MicroInterpreter on the same
 * board/BSP/clock as the TiGrIS harness, timed with the DWT cycle counter, and
 * emits the same BENCH_RESULT protocol so results.py parses both side by side.
 * The model is selected at build time (TFLM_MODEL, default ds_cnn).
 *
 * PARITY: when TiGrIS runs the *matched* plan (its ONNX reconstructed from this
 * exact .tflite via tools/tflite_to_qdq_onnx.py), both frameworks execute the
 * same INT8 weights through CMSIS-NN, so the OUTPUT_I8 vectors are directly
 * comparable device-to-device (DS-CNN matched is bit-identical). Both also fill
 * the input with int8 value 1, so the comparison is well-defined.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>

extern "C" {
#include "platform.h"
}

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

/* Model selection (override from CMake via -DTFLM_MODEL=<name>). Defaults to
 * DS-CNN. TFLM_MODEL_SYM is the bare C array symbol, TFLM_MODEL_NAME the label,
 * TFLM_MODEL_HEADER the include. The op resolver below is a superset, so the
 * same harness runs any of the benchmark models. */
#ifndef TFLM_MODEL_HEADER
#define TFLM_MODEL_HEADER "ds_cnn_tflite_i8.h"
#define TFLM_MODEL_SYM    ds_cnn_tflite_i8
#define TFLM_MODEL_NAME   "ds_cnn"
#endif
#define TFLM_XCAT(a, b) a##b
#define TFLM_CAT(a, b) TFLM_XCAT(a, b)
#define TFLM_MODEL_LEN TFLM_CAT(TFLM_MODEL_SYM, _len)
#include TFLM_MODEL_HEADER

#ifndef WARMUP_RUNS
#define WARMUP_RUNS 3
#endif
#ifndef BENCH_RUNS
#define BENCH_RUNS 30
#endif
#ifndef TFLM_ARENA_BYTES
#define TFLM_ARENA_BYTES (128 * 1024)
#endif

alignas(16) static uint8_t tensor_arena[TFLM_ARENA_BYTES];

static float cycles_to_ms(uint32_t c)
{
    return (float)((double)c * 1000.0 / (double)platform_cpu_hz());
}

static void sort_u32(uint32_t *a, int n)
{
    for (int i = 1; i < n; i++) {
        uint32_t v = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; j--; }
        a[j + 1] = v;
    }
}

static void emit_failed(const char *status)
{
    printf("\nBENCH_RESULT:framework=tflm,kernel=cmsis_nn,dtype=int8,model="
           TFLM_MODEL_NAME ",board=%s,status=%s,runs=0\n",
           platform_board_name(), status);
    printf("BENCH_DONE\n");
}

int main(void)
{
    platform_init();
    printf("\nTFLM Cortex-M Benchmark (board=%s, %lu MHz)\n\n",
           platform_board_name(), (unsigned long)(platform_cpu_hz() / 1000000u));

    const tflite::Model *model = tflite::GetModel(TFLM_MODEL_SYM);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("schema version mismatch: %lu vs %d\n",
               (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
        emit_failed("SCHEMA_MISMATCH");
        platform_halt();
    }

    /* DS-CNN op set (CMSIS-NN optimized kernels selected at lib build time). */
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddMean();            /* GlobalAveragePool lowers to Mean */
    resolver.AddAveragePool2D();
    resolver.AddSoftmax();
    resolver.AddQuantize();
    resolver.AddDequantize();

    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, TFLM_ARENA_BYTES);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors failed (arena too small)\n");
        emit_failed("ARENA_TOO_SMALL");
        platform_halt();
    }

    TfLiteTensor *input = interpreter.input(0);
    TfLiteTensor *output = interpreter.output(0);
    uint32_t arena_used = (uint32_t)interpreter.arena_used_bytes();

    printf("Model:      " TFLM_MODEL_NAME " (int8)\n");
    printf("Arena used: %lu bytes\n", (unsigned long)arena_used);
    printf("Model size: %u bytes\n", TFLM_MODEL_LEN);
    printf("Input:      %u bytes   Output: %u bytes\n",
           (unsigned)input->bytes, (unsigned)output->bytes);

    uint32_t cyc[BENCH_RUNS];

    printf("\nWarmup (%d runs)...\n", WARMUP_RUNS);
    for (int i = 0; i < WARMUP_RUNS; i++) {
        memset(input->data.int8, 1, input->bytes);  /* int8 = 1, as TiGrIS */
        uint32_t c0 = platform_cycles();
        TfLiteStatus st = interpreter.Invoke();
        uint32_t c1 = platform_cycles();
        if (st != kTfLiteOk) { emit_failed("INVOKE_FAILED"); platform_halt(); }
        printf("  warmup[%d]: %.3f ms\n", i, cycles_to_ms(c1 - c0));
    }

    printf("\nBenchmark (%d runs)...\n", BENCH_RUNS);
    for (int i = 0; i < BENCH_RUNS; i++) {
        memset(input->data.int8, 1, input->bytes);
        uint32_t c0 = platform_cycles();
        TfLiteStatus st = interpreter.Invoke();
        uint32_t c1 = platform_cycles();
        if (st != kTfLiteOk) { emit_failed("INVOKE_FAILED"); platform_halt(); }
        cyc[i] = c1 - c0;
        printf("  run[%d]: %.3f ms (%lu cyc)\n", i, cycles_to_ms(cyc[i]),
               (unsigned long)cyc[i]);
    }

    uint32_t sorted[BENCH_RUNS];
    memcpy(sorted, cyc, sizeof(cyc));
    sort_u32(sorted, BENCH_RUNS);
    uint64_t sum = 0;
    for (int i = 0; i < BENCH_RUNS; i++) sum += cyc[i];
    uint32_t mean_cyc = (uint32_t)(sum / BENCH_RUNS);
    uint32_t min_cyc = sorted[0], max_cyc = sorted[BENCH_RUNS - 1];
    uint32_t median_cyc = (BENCH_RUNS & 1)
        ? sorted[BENCH_RUNS / 2]
        : (uint32_t)(((uint64_t)sorted[BENCH_RUNS / 2 - 1] + sorted[BENCH_RUNS / 2]) / 2);
    double var = 0;
    for (int i = 0; i < BENCH_RUNS; i++) {
        double d = (double)cyc[i] - (double)mean_cyc;
        var += d * d;
    }
    uint32_t stdev_cyc = (uint32_t)__builtin_sqrt(var / BENCH_RUNS);

    printf("\nOutput values:\n  OUTPUT_I8:");
    for (size_t j = 0; j < output->bytes; j++)
        printf(" %d", (int)output->data.int8[j]);
    printf("\n");

    printf("\nBENCH_RESULT:"
           "framework=tflm,kernel=cmsis_nn,dtype=int8,model=" TFLM_MODEL_NAME ",board=%s,"
           "cpu_mhz=%lu,status=ok,"
           "latency_mean_ms=%.3f,latency_median_ms=%.3f,latency_min_ms=%.3f,"
           "latency_max_ms=%.3f,latency_stdev_ms=%.3f,latency_median_cycles=%lu,"
           "sram_peak_bytes=%lu,arena_kb=%lu,plan_flash_kb=%lu,runs=%d\n",
           platform_board_name(),
           (unsigned long)(platform_cpu_hz() / 1000000u),
           cycles_to_ms(mean_cyc), cycles_to_ms(median_cyc), cycles_to_ms(min_cyc),
           cycles_to_ms(max_cyc), cycles_to_ms(stdev_cyc), (unsigned long)median_cyc,
           /* arena_used_bytes() is TFLM's true activation working set. */
           (unsigned long)arena_used, (unsigned long)(arena_used / 1024),
           (unsigned long)(TFLM_MODEL_LEN / 1024), BENCH_RUNS);
    printf("BENCH_DONE\n");

    platform_halt();
    return 0;
}
