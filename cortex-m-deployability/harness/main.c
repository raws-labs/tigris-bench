/*
 * TiGrIS Cortex-M Deployability Benchmark
 *
 * Loads a DS-CNN INT8 .tgrs plan embedded in flash, runs warmup + N timed
 * inference runs timed with the DWT cycle counter, prints the INT8 output for
 * parity checking, and emits a machine-parseable result line.
 *
 * Board-independent: all hardware access goes through platform.h, implemented
 * per board under boards/<board>/.
 *
 * Kernel selection via build define:
 *   -DBENCH_KERNEL_CMSIS_NN   tigris_dispatch_kernel_cmsis_nn (int8 + CMSIS-NN)
 *   default                   tigris_dispatch_kernel_s8       (int8 reference)
 *
 * Arena sizing via build define (so the linked image's .bss reflects the real
 * working set that `arm-none-eabi-size` reports):
 *   -DTIGRIS_FAST_ARENA_BYTES=N   -DTIGRIS_SLOW_ARENA_BYTES=N
 *
 * Machine-parseable output:
 *   OUTPUT_I8: <int8 values...>
 *   BENCH_RESULT:framework=tigris,kernel=...,latency_median_ms=...,...
 *   BENCH_DONE
 */

#include <stdio.h>
#include <string.h>

#include "platform.h"

#include "tigris.h"
#include "tigris_loader.h"
#include "tigris_mem.h"
#include "tigris_executor.h"
#include "tigris_kernels_s8.h"
#if defined(BENCH_KERNEL_CMSIS_NN)
#include "tigris_kernels_cmsis_nn.h"
#endif

/* Embedded plan blob (generated from the .tgrs by tools/bin2c.py). */
extern const unsigned char g_tigris_plan[];
extern const unsigned int  g_tigris_plan_len;

#ifndef WARMUP_RUNS
#define WARMUP_RUNS 3
#endif
#ifndef BENCH_RUNS
#define BENCH_RUNS 30          /* spec: median of N >= 30 + min/max */
#endif

/* Static backing store - no heap. Sized at build time; defaults are generous
 * for DS-CNN and fit the H753 AXI SRAM. Set per build config to sweep budgets. */
#ifndef TIGRIS_FAST_ARENA_BYTES
#define TIGRIS_FAST_ARENA_BYTES (128u * 1024u)
#endif
#ifndef TIGRIS_SLOW_ARENA_BYTES
#define TIGRIS_SLOW_ARENA_BYTES (256u * 1024u)
#endif
#ifndef BENCH_MAX_TENSORS
#define BENCH_MAX_TENSORS 256
#endif

static uint8_t s_fast_arena[TIGRIS_FAST_ARENA_BYTES] __attribute__((aligned(16)));
static uint8_t s_slow_arena[TIGRIS_SLOW_ARENA_BYTES] __attribute__((aligned(16)));
static void   *s_tensor_ptrs[BENCH_MAX_TENSORS];

static const char *kernel_name(void)
{
#if defined(BENCH_KERNEL_CMSIS_NN)
    return "cmsis_nn";
#else
    return "s8_ref";
#endif
}

static tigris_kernel_fn select_kernel(void)
{
#if defined(BENCH_KERNEL_CMSIS_NN)
    return tigris_dispatch_kernel_cmsis_nn;
#else
    return tigris_dispatch_kernel_s8;
#endif
}

#ifdef BENCH_PROFILE_OPS
/* Per-op cycle profiling: wraps the real dispatch, times each op with the cycle
 * counter and accumulates per op index. No I/O inside the timed region. */
#ifndef BENCH_MAX_OPS
#define BENCH_MAX_OPS 256
#endif
static tigris_kernel_fn s_prof_inner;
static uint64_t s_op_cycles[BENCH_MAX_OPS];
static uint8_t  s_op_type[BENCH_MAX_OPS];

static int prof_dispatch(const tigris_plan_t *plan, const tigris_op_t *op,
                         uint16_t op_index, tigris_mem_t *mem, void *user_ctx)
{
    uint32_t c0 = platform_cycles();
    int r = s_prof_inner(plan, op, op_index, mem, user_ctx);
    uint32_t c1 = platform_cycles();
    if (op_index < BENCH_MAX_OPS) {
        s_op_cycles[op_index] += (uint32_t)(c1 - c0);
        s_op_type[op_index] = op->op_type;
    }
    return r;
}
#endif

/* cycles -> milliseconds at the known core clock. */
static float cycles_to_ms(uint32_t cycles)
{
    return (float)((double)cycles * 1000.0 / (double)platform_cpu_hz());
}

/* Run one inference. Re-inits the arena state, refills inputs, returns the
 * DWT cycle count for the tigris_run() call. Sets *ok to 0 on failure. */
static uint32_t run_once(tigris_plan_t *plan, tigris_mem_t *mem,
                         tigris_kernel_fn dispatch,
                         tigris_exec_stats_t *out_stats, int *ok)
{
    tigris_mem_init(mem, mem->tensor_ptrs, mem->num_tensors,
                    mem->fast_base, mem->fast_size,
                    mem->slow_base, mem->slow_size);

    for (uint8_t i = 0; i < plan->header->num_model_inputs; i++) {
        uint16_t tidx = plan->model_inputs[i];
        uint32_t sz = plan->tensors[tidx].size_bytes;
        if (tigris_mem_alloc_slow(mem, tidx, sz) != TIGRIS_MEM_OK) {
            *ok = 0;
            return 0;
        }
        /* int8 value 1 - matches prepare.py's reference input generator. */
        memset(mem->tensor_ptrs[tidx], 1, sz);
    }

    tigris_exec_stats_t stats;
    uint32_t c0 = platform_cycles();
    tigris_exec_error_t err = tigris_run(plan, mem, dispatch, NULL, &stats);
    uint32_t c1 = platform_cycles();

    if (err != TIGRIS_EXEC_OK) {
        printf("inference failed: %s\n", tigris_exec_error_str(err));
        *ok = 0;
        return 0;
    }
    if (out_stats)
        *out_stats = stats;
    *ok = 1;
    return c1 - c0;
}

/* In-place insertion sort - tiny N, no libc qsort dependency. */
static void sort_u32(uint32_t *a, int n)
{
    for (int i = 1; i < n; i++) {
        uint32_t v = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > v) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = v;
    }
}

static void emit_failed(const char *model, const char *status)
{
    printf("\nBENCH_RESULT:framework=tigris,kernel=%s,dtype=int8,model=%s,"
           "board=%s,status=%s,runs=0\n",
           kernel_name(), model, platform_board_name(), status);
    printf("BENCH_DONE\n");
}

int main(void)
{
    platform_init();

    printf("\nTiGrIS Cortex-M Benchmark (board=%s, kernel=%s, %lu MHz)\n\n",
           platform_board_name(), kernel_name(),
           (unsigned long)(platform_cpu_hz() / 1000000u));

    /* Bring-up: dump the clock diagnostic (stage 5 = target clock locked). */
    const volatile uint32_t *cd = platform_clock_diag();
    if (cd) {
        printf("CLOCK_DIAG: stage=%lu cr3=%08lx csr1=%08lx d3cr=%08lx "
               "rcc_cr=%08lx cfgr=%08lx idcode=%08lx flash_acr=%08lx\n",
               (unsigned long)cd[0], (unsigned long)cd[1], (unsigned long)cd[2],
               (unsigned long)cd[3], (unsigned long)cd[4], (unsigned long)cd[5],
               (unsigned long)cd[6], (unsigned long)cd[7]);
    }

    /* 1. Load the embedded plan. */
    tigris_plan_t plan;
    tigris_error_t perr = tigris_plan_load(g_tigris_plan, g_tigris_plan_len, &plan);
    if (perr != TIGRIS_OK) {
        printf("plan load failed: %s\n", tigris_error_str(perr));
        emit_failed("unknown", "PLAN_LOAD_FAILED");
        platform_halt();
    }

    const char *model_name = tigris_model_name(&plan);
    uint8_t input_dtype = plan.tensors[plan.model_inputs[0]].dtype;
    int is_quantized = (input_dtype == 3); /* ONNX dtype 3 = INT8 */

    printf("Model:   %s\n", model_name);
    printf("Type:    %s\n", is_quantized ? "int8" : "f32");
    printf("Ops:     %u\n", plan.header->num_ops);
    printf("Stages:  %u\n", plan.header->num_stages);
    printf("Tensors: %u\n", plan.header->num_tensors);
    printf("Budget:  %lu bytes\n", (unsigned long)plan.header->budget);

    if (!is_quantized) {
        printf("error: this harness benchmarks the INT8 plan only\n");
        emit_failed(model_name, "NOT_INT8");
        platform_halt();
    }
    if (plan.header->num_tensors > BENCH_MAX_TENSORS) {
        printf("error: num_tensors %u > BENCH_MAX_TENSORS %u\n",
               plan.header->num_tensors, BENCH_MAX_TENSORS);
        emit_failed(model_name, "TOO_MANY_TENSORS");
        platform_halt();
    }

    /* 2. Size the arenas. fast_size = plan budget (+ decompression overhead),
     *    bounded by the static backing array. */
    /* Tight provisioning policy. The executor compacts the fast pool REACTIVELY
     * (only on OOM), so a generous arena yields a lazy bump high-water well above
     * the compiler's planned peak (e.g. DS-CNN 32 KB vs a 16 KB plan) and a
     * misleadingly large RAM figure. We provision the fast arena at the planned
     * peak plus a bounded margin so compaction engages and the measured working
     * set reflects TiGrIS's true minimum SRAM, directly comparable to TFLM's
     * interpreter.arena_used_bytes(). The margin covers (a) per-tensor 16-byte
     * alignment, which the planned peak (computed on raw sizes) omits, and (b) one
     * compaction transient. */
    uint32_t align_reserve =
        (uint32_t)plan.header->num_tensors * (uint32_t)TIGRIS_TENSOR_ALIGN + 512u;
    uint32_t base;
    if (plan.header->peak > 0 && plan.header->peak <= plan.header->budget) {
        /* Non-tiled: the naive peak fits the compile budget, so provision tight at
         * the peak - the measured working set is then the true minimum SRAM. */
        base = plan.header->peak;
    } else if (plan.header->budget > 0) {
        /* Tiled: the naive peak exceeds the budget; the runtime tiles activations
         * down to the budget, so provision the BUDGET, not the (untiled) peak. */
        base = plan.header->budget;
    } else {
        base = TIGRIS_FAST_ARENA_BYTES;
    }
    uint32_t fast_size = base + align_reserve + tigris_weight_decompression_overhead(&plan);
#ifdef TIGRIS_FAST_OVERRIDE
    /* Diagnostic: force a specific fast-arena size to probe the runtime minimum. */
    if (TIGRIS_FAST_OVERRIDE)
        fast_size = (uint32_t)(TIGRIS_FAST_OVERRIDE);
#endif
    if (fast_size > sizeof(s_fast_arena)) {
        printf("error: fast arena needs %lu B, have %lu B\n",
               (unsigned long)fast_size, (unsigned long)sizeof(s_fast_arena));
        emit_failed(model_name, "ARENA_TOO_SMALL");
        platform_halt();
    }
    uint32_t slow_size = sizeof(s_slow_arena);

    tigris_mem_t mem;
    tigris_mem_error_t merr = tigris_mem_init(
        &mem, s_tensor_ptrs, plan.header->num_tensors,
        s_fast_arena, fast_size, s_slow_arena, slow_size);
    if (merr != TIGRIS_MEM_OK) {
        printf("mem init failed: %s\n", tigris_mem_error_str(merr));
        emit_failed(model_name, "MEM_INIT_FAILED");
        platform_halt();
    }

    tigris_kernel_fn dispatch = select_kernel();

#if defined(BENCH_KERNEL_CMSIS_NN)
    /* Reserve CMSIS-NN kernel scratch from the arena top (no stack VLAs). Must
     * run after mem_init and before any inference; reduces mem.fast_size. */
    if (tigris_cmsis_nn_prepare(&plan, &mem) != 0) {
        printf("cmsis_nn prepare failed (arena too small for scratch)\n");
        emit_failed(model_name, "SCRATCH_TOO_SMALL");
        platform_halt();
    }
#endif

#ifdef BENCH_PROFILE_OPS
    s_prof_inner = dispatch;
    dispatch = prof_dispatch;
#endif

    printf("\nFast arena: %lu KB   Slow arena: %lu KB\n\n",
           (unsigned long)(fast_size / 1024), (unsigned long)(slow_size / 1024));

    /* 3. Warmup (discard - flash/I-cache priming on M7). */
    printf("Warmup (%d runs)...\n", WARMUP_RUNS);
    for (int i = 0; i < WARMUP_RUNS; i++) {
        int ok = 0;
        uint32_t c = run_once(&plan, &mem, dispatch, NULL, &ok);
        if (!ok) {
            emit_failed(model_name, "INFERENCE_FAILED");
            platform_halt();
        }
        printf("  warmup[%d]: %.3f ms\n", i, cycles_to_ms(c));
    }

    /* 4. Timed runs. */
#ifdef BENCH_PROFILE_OPS
    memset(s_op_cycles, 0, sizeof(s_op_cycles));  /* discard warmup accumulation */
#endif
    printf("\nBenchmark (%d runs)...\n", BENCH_RUNS);
    uint32_t cyc[BENCH_RUNS];
    tigris_exec_stats_t last_stats;
    memset(&last_stats, 0, sizeof(last_stats));
    for (int i = 0; i < BENCH_RUNS; i++) {
        int ok = 0;
        tigris_exec_stats_t *sp = (i == BENCH_RUNS - 1) ? &last_stats : NULL;
        cyc[i] = run_once(&plan, &mem, dispatch, sp, &ok);
        if (!ok) {
            emit_failed(model_name, "INFERENCE_FAILED");
            platform_halt();
        }
        printf("  run[%d]: %.3f ms (%lu cyc)\n",
               i, cycles_to_ms(cyc[i]), (unsigned long)cyc[i]);
    }

#ifdef BENCH_PROFILE_OPS
    /* Per-op cycle breakdown (mean over the timed runs). */
    printf("\nPer-op cycles (mean of %d runs):\n", BENCH_RUNS);
    for (uint16_t i = 0; i < plan.header->num_ops && i < BENCH_MAX_OPS; i++)
        printf("  OP[%2u] type=%u  %lu cyc\n", i, (unsigned)s_op_type[i],
               (unsigned long)(s_op_cycles[i] / BENCH_RUNS));
#endif

    /* 5. Statistics over cycle counts (the clock-independent unit). */
    uint32_t sorted[BENCH_RUNS];
    memcpy(sorted, cyc, sizeof(cyc));
    sort_u32(sorted, BENCH_RUNS);

    uint64_t sum = 0;
    for (int i = 0; i < BENCH_RUNS; i++)
        sum += cyc[i];
    uint32_t mean_cyc = (uint32_t)(sum / BENCH_RUNS);
    uint32_t min_cyc = sorted[0];
    uint32_t max_cyc = sorted[BENCH_RUNS - 1];
    uint32_t median_cyc = (BENCH_RUNS & 1)
        ? sorted[BENCH_RUNS / 2]
        : (uint32_t)(((uint64_t)sorted[BENCH_RUNS / 2 - 1] +
                      sorted[BENCH_RUNS / 2]) / 2);

    double var = 0;
    for (int i = 0; i < BENCH_RUNS; i++) {
        double d = (double)cyc[i] - (double)mean_cyc;
        var += d * d;
    }
    uint32_t stdev_cyc = (uint32_t)__builtin_sqrt(var / BENCH_RUNS);

    /* 6. Output values for parity validation. */
    printf("\nOutput values:\n");
    uint16_t n_out = plan.header->num_model_outputs;
    uint16_t fallback_out;
    const uint16_t *out_list;
    if (n_out > 0) {
        out_list = plan.model_outputs;
    } else {
        /* Stale plans built before the num_model_outputs compiler fix report 0
         * outputs; fall back to the last op's first output (as the ESP harness
         * does). For DS-CNN that is the 12-class tensor. */
        const tigris_op_t *last = &plan.ops[plan.header->num_ops - 1];
        fallback_out = tigris_op_outputs(&plan, last)[0];
        out_list = &fallback_out;
        n_out = 1;
    }
    for (uint16_t i = 0; i < n_out; i++) {
        uint16_t tidx = out_list[i];
        void *out_ptr = mem.tensor_ptrs[tidx];
        if (!out_ptr)
            continue;
        int8_t *out = (int8_t *)out_ptr;
        uint32_t n = plan.tensors[tidx].size_bytes;
        printf("  OUTPUT_I8:");
        for (uint32_t j = 0; j < n; j++)
            printf(" %d", (int)out[j]);
        printf("\n");
    }

    /* 6b. Honest, MEASURED SRAM working set. This is what is directly comparable
     *     to TFLM's interpreter.arena_used_bytes(): the actual peak bytes the
     *     framework needs in SRAM at runtime, not a compile-time estimate.
     *       act_peak  = fast + slow arena high-water (the live activation set)
     *       scratch   = CMSIS-NN scratch carved from the fast arena (0 for s8_ref)
     *       meta      = the runtime tensor-pointer table (TiGrIS's RAM metadata;
     *                   TFLM keeps its equivalent TfLiteEvalTensor table inside
     *                   the arena, so arena_used already counts it - we add ours)
     *     plan.header->peak (the old reported value) is the compiler's compile-
     *     time activation estimate; kept as plan_act_peak_bytes for reference. */
    uint32_t scratch_bytes = fast_size - mem.fast_size;     /* prepare() carve; 0 if none */
    uint32_t meta_bytes    = (uint32_t)plan.header->num_tensors * (uint32_t)sizeof(void *);
    uint32_t fast_peak     = mem.fast_peak;
    uint32_t slow_peak     = last_stats.slow_peak;
    uint32_t act_peak      = fast_peak + slow_peak;
    uint32_t working_bytes = act_peak + scratch_bytes + meta_bytes;

    /* 7. Machine-parseable result line. Reports both cycles (honest, clock-
     *    independent) and ms (for humans). */
    printf("\nBENCH_RESULT:"
           "framework=tigris,"
           "kernel=%s,"
           "dtype=int8,"
           "model=%s,"
           "board=%s,"
           "cpu_mhz=%lu,"
           "status=ok,"
           "latency_mean_ms=%.3f,"
           "latency_median_ms=%.3f,"
           "latency_min_ms=%.3f,"
           "latency_max_ms=%.3f,"
           "latency_stdev_ms=%.3f,"
           "latency_median_cycles=%lu,"
           "latency_min_cycles=%lu,"
           "latency_max_cycles=%lu,"
           /* sram_peak_bytes is the MEASURED runtime working set (act+scratch+meta),
            * directly comparable to TFLM arena_used_bytes. The breakdown + the old
            * compile-time estimate (plan_act_peak_bytes) follow for transparency. */
           "sram_peak_bytes=%lu,"
           "sram_act_peak_bytes=%lu,"
           "sram_fast_peak_bytes=%lu,"
           "sram_slow_peak_bytes=%lu,"
           "sram_scratch_bytes=%lu,"
           "sram_meta_bytes=%lu,"
           "plan_act_peak_bytes=%lu,"
           "sram_budget_kb=%lu,"
           "sram_provisioned_kb=%lu,"
           "slow_peak_kb=%lu,"
           "plan_flash_kb=%lu,"
           "runs=%d,"
           "stages_normal=%u,"
           "stages_tiled=%u,"
           "stages_chain=%u,"
           "total_tiles=%u\n",
           kernel_name(),
           model_name,
           platform_board_name(),
           (unsigned long)(platform_cpu_hz() / 1000000u),
           cycles_to_ms(mean_cyc),
           cycles_to_ms(median_cyc),
           cycles_to_ms(min_cyc),
           cycles_to_ms(max_cyc),
           cycles_to_ms(stdev_cyc),
           (unsigned long)median_cyc,
           (unsigned long)min_cyc,
           (unsigned long)max_cyc,
           (unsigned long)working_bytes,
           (unsigned long)act_peak,
           (unsigned long)fast_peak,
           (unsigned long)slow_peak,
           (unsigned long)scratch_bytes,
           (unsigned long)meta_bytes,
           (unsigned long)plan.header->peak,
           (unsigned long)(plan.header->budget / 1024),
           (unsigned long)(fast_size / 1024),
           (unsigned long)(last_stats.slow_peak / 1024),
           (unsigned long)(g_tigris_plan_len / 1024),
           BENCH_RUNS,
           (unsigned)last_stats.stages_normal,
           (unsigned)last_stats.stages_tiled,
           (unsigned)last_stats.stages_chain,
           (unsigned)last_stats.total_tiles);

    printf("BENCH_DONE\n");

    platform_halt();
    return 0;
}
