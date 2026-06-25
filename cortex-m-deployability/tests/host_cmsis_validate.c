/*
 * Host-side correctness gate for the CMSIS-NN backend.
 *
 * Builds the TiGrIS runtime + ARM CMSIS-NN for x86 (CMSIS-NN's pure-C path is
 * bit-exact with its device kernels for the int8 ops, so a host match implies
 * an on-device match), runs a DS-CNN INT8 plan through
 * tigris_dispatch_kernel_cmsis_nn, and compares the output to the golden
 * reference. This exercises the adapter (op -> CMSIS-NN argument marshalling)
 * which has never been compiled or run before, with no board required.
 *
 *   host_cmsis_validate <plan.tgrs> <reference_i8.bin> [cmsis_nn|s8]
 *
 * Exit 0 on PASS. See third_party/ for the vendored sources; this file is
 * built by tests/run_host_validate.sh.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tigris.h"
#include "tigris_loader.h"
#include "tigris_mem.h"
#include "tigris_executor.h"
#include "tigris_kernels_s8.h"
#include "tigris_kernels_cmsis_nn.h"

#define INT8_TOL 2  /* allow the known +-1 quantized-multiplier nudge slack */

static uint8_t *read_file(const char *path, long *len_out)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = aligned_alloc(32, ((size_t)len + 31u) & ~31u);
    if (buf && fread(buf, 1, (size_t)len, f) != (size_t)len) { free(buf); buf = NULL; }
    fclose(f);
    if (buf) *len_out = len;
    return buf;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s <plan.tgrs> <reference_i8.bin> [cmsis_nn|s8]\n", argv[0]);
        return 2;
    }
    const char *plan_path = argv[1];
    const char *ref_path = argv[2];
    const char *which = (argc > 3) ? argv[3] : "cmsis_nn";
    const char *input_path = (argc > 4) ? argv[4] : NULL;  /* raw int8 input, else fill with 1 */

    long plan_len = 0;
    uint8_t *plan_buf = read_file(plan_path, &plan_len);
    if (!plan_buf) return 2;

    tigris_plan_t plan;
    tigris_error_t perr = tigris_plan_load(plan_buf, (uint32_t)plan_len, &plan);
    if (perr != TIGRIS_OK) {
        fprintf(stderr, "plan load failed: %s\n", tigris_error_str(perr));
        return 2;
    }

    uint32_t fast_size = plan.header->budget ? plan.header->budget : (64u * 1024u);
    fast_size += tigris_weight_decompression_overhead(&plan);
    uint32_t slow_size = 1u << 20;  /* 1 MB, generous for host */

    uint8_t *fast = aligned_alloc(32, fast_size);
    uint8_t *slow = aligned_alloc(32, slow_size);
    void **tensor_ptrs = calloc(plan.header->num_tensors, sizeof(void *));
    if (!fast || !slow || !tensor_ptrs) { fprintf(stderr, "alloc failed\n"); return 2; }

    tigris_mem_t mem;
    if (tigris_mem_init(&mem, tensor_ptrs, plan.header->num_tensors,
                        fast, fast_size, slow, slow_size) != TIGRIS_MEM_OK) {
        fprintf(stderr, "mem init failed\n");
        return 2;
    }

    /* Exercise the arena-backed scratch path (mirrors the device harness). */
    if (strcmp(which, "s8") != 0)
        tigris_cmsis_nn_prepare(&plan, &mem);

    for (uint8_t i = 0; i < plan.header->num_model_inputs; i++) {
        uint16_t tidx = plan.model_inputs[i];
        uint32_t sz = plan.tensors[tidx].size_bytes;
        if (tigris_mem_alloc_slow(&mem, tidx, sz) != TIGRIS_MEM_OK) {
            fprintf(stderr, "input alloc failed\n");
            return 2;
        }
        if (input_path && i == 0) {
            long il = 0;
            uint8_t *ibuf = read_file(input_path, &il);
            if (!ibuf || (uint32_t)il < sz) { fprintf(stderr, "bad input file\n"); return 2; }
            memcpy(mem.tensor_ptrs[tidx], ibuf, sz);
            free(ibuf);
        } else {
            memset(mem.tensor_ptrs[tidx], 1, sz);  /* matches the reference generator */
        }
    }

    tigris_kernel_fn dispatch = (strcmp(which, "s8") == 0)
        ? tigris_dispatch_kernel_s8
        : tigris_dispatch_kernel_cmsis_nn;

    tigris_exec_stats_t stats;
    tigris_exec_error_t eerr = tigris_run(&plan, &mem, dispatch, NULL, &stats);
    if (eerr != TIGRIS_EXEC_OK) {
        fprintf(stderr, "inference failed: %s\n", tigris_exec_error_str(eerr));
        return 2;
    }

    uint16_t out_idx;
    if (plan.header->num_model_outputs > 0) {
        out_idx = plan.model_outputs[0];
    } else {
        /* Stale plans built before the num_model_outputs compiler fix report 0
         * outputs; fall back to the last op's first output (as the ESP harness
         * does). For DS-CNN that is the 12-class tensor. */
        const tigris_op_t *last = &plan.ops[plan.header->num_ops - 1];
        out_idx = tigris_op_outputs(&plan, last)[0];
    }
    int8_t *out = (int8_t *)mem.tensor_ptrs[out_idx];
    uint32_t n = plan.tensors[out_idx].size_bytes;

    long ref_len = 0;
    uint8_t *ref_buf = read_file(ref_path, &ref_len);
    if (!ref_buf) return 2;
    if ((uint32_t)ref_len != n) {
        fprintf(stderr, "length mismatch: device=%u ref=%ld\n", n, ref_len);
        return 1;
    }
    const int8_t *ref = (const int8_t *)ref_buf;

    int max_abs = 0, dev_arg = 0, ref_arg = 0;
    for (uint32_t i = 0; i < n; i++) {
        int d = out[i] - ref[i];
        if (d < 0) d = -d;
        if (d > max_abs) max_abs = d;
        if (out[i] > out[dev_arg]) dev_arg = (int)i;
        if (ref[i] > ref[ref_arg]) ref_arg = (int)i;
    }

    printf("kernel=%-8s  ops=%u stages=%u  out_len=%u\n",
           which, plan.header->num_ops, plan.header->num_stages, n);
    printf("device:");
    for (uint32_t i = 0; i < n; i++) printf(" %d", out[i]);
    printf("\nref:   ");
    for (uint32_t i = 0; i < n; i++) printf(" %d", ref[i]);
    printf("\nmax_abs_diff=%d  argmax dev=%d ref=%d\n", max_abs, dev_arg, ref_arg);

    int pass = (max_abs <= INT8_TOL) && (dev_arg == ref_arg);
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
