/*
 * Hardware-free correctness gate for the CMSIS-NN backend.
 *
 * Builds the TiGrIS runtime and ARM CMSIS-NN's portable C path for the host,
 * runs one INT8 plan through both the reference and CMSIS-NN dispatchers, and
 * compares their outputs. This exercises the real adapter argument marshalling
 * and scratch-arena lifecycle without requiring a board.
 *
 *   host_cmsis_validate <plan.tgrs> [input_i8.bin]
 *
 * The optional input file supplies raw bytes for the first model input.
 * Otherwise every input byte is filled with 1.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tigris.h"
#include "tigris_executor.h"
#include "tigris_kernels_cmsis_nn.h"
#include "tigris_kernels_s8.h"
#include "tigris_loader.h"
#include "tigris_mem.h"

/* Match the benchmark's accepted one-LSB requantization nudge. */
#define INT8_TOL 1
#define HOST_SLOW_ARENA_BYTES (1u << 20)
#define HOST_ALIGNMENT 32u

static size_t align_up(size_t value)
{
    return (value + HOST_ALIGNMENT - 1u) & ~(HOST_ALIGNMENT - 1u);
}

static uint8_t *read_file(const char *path, long *len_out)
{
    FILE *file = fopen(path, "rb");
    long len;
    uint8_t *buffer;

    if (file == NULL) {
        fprintf(stderr, "cannot open %s\n", path);
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0
            || (len = ftell(file)) < 0
            || fseek(file, 0, SEEK_SET) != 0) {
        fprintf(stderr, "cannot size %s\n", path);
        fclose(file);
        return NULL;
    }
    buffer = aligned_alloc(HOST_ALIGNMENT, align_up((size_t)len));
    if (buffer == NULL
            || fread(buffer, 1, (size_t)len, file) != (size_t)len) {
        fprintf(stderr, "cannot read %s\n", path);
        free(buffer);
        fclose(file);
        return NULL;
    }
    fclose(file);
    *len_out = len;
    return buffer;
}

static int model_output_index(
    const tigris_plan_t *plan, uint16_t *output_index)
{
    if (plan->header->num_model_outputs == 0u) {
        fprintf(stderr, "plan declares no model output\n");
        return 0;
    }
    *output_index = plan->model_outputs[0];
    return 1;
}

static int run_backend(
    const tigris_plan_t *plan,
    tigris_kernel_fn dispatch,
    int prepare_cmsis,
    const char *input_path,
    int8_t **output_copy,
    uint32_t *output_size)
{
    uint32_t fast_size = prepare_cmsis
        ? tigris_cmsis_nn_fast_arena_required(plan)
        : tigris_fast_arena_required(plan);
    uint8_t *fast = aligned_alloc(HOST_ALIGNMENT, align_up(fast_size));
    uint8_t *slow = aligned_alloc(
        HOST_ALIGNMENT, align_up(HOST_SLOW_ARENA_BYTES));
    void **tensor_ptrs = calloc(plan->header->num_tensors, sizeof(void *));
    tigris_mem_t mem;
    tigris_exec_stats_t stats;
    uint16_t output_index;
    int cmsis_prepared = 0;
    int ok = 0;

    if (fast == NULL || slow == NULL || tensor_ptrs == NULL) {
        fprintf(stderr, "arena allocation failed\n");
        goto cleanup;
    }
    if (tigris_mem_init(
            &mem, tensor_ptrs, plan->header->num_tensors,
            fast, fast_size, slow, HOST_SLOW_ARENA_BYTES) != TIGRIS_MEM_OK) {
        fprintf(stderr, "memory initialization failed\n");
        goto cleanup;
    }
    if (prepare_cmsis && tigris_cmsis_nn_prepare(plan, &mem) != 0) {
        fprintf(stderr, "CMSIS-NN scratch preparation failed\n");
        goto cleanup;
    }
    cmsis_prepared = prepare_cmsis;

    for (uint8_t i = 0; i < plan->header->num_model_inputs; ++i) {
        uint16_t tensor_index = plan->model_inputs[i];
        uint32_t size = plan->tensors[tensor_index].size_bytes;

        if (tigris_mem_alloc_slow(&mem, tensor_index, size) != TIGRIS_MEM_OK) {
            fprintf(stderr, "input allocation failed\n");
            goto cleanup;
        }
        if (input_path != NULL && i == 0u) {
            long input_len = 0;
            uint8_t *input = read_file(input_path, &input_len);
            if (input == NULL || (uint32_t)input_len != size) {
                fprintf(
                    stderr, "input size mismatch: plan=%u file=%ld\n",
                    size, input_len);
                free(input);
                goto cleanup;
            }
            memcpy(mem.tensor_ptrs[tensor_index], input, size);
            free(input);
        } else {
            memset(mem.tensor_ptrs[tensor_index], 1, size);
        }
    }

    if (tigris_run(plan, &mem, dispatch, NULL, &stats) != TIGRIS_EXEC_OK) {
        fprintf(stderr, "inference failed\n");
        goto cleanup;
    }
    if (!model_output_index(plan, &output_index)
            || mem.tensor_ptrs[output_index] == NULL) {
        fprintf(stderr, "model output is unavailable\n");
        goto cleanup;
    }

    *output_size = plan->tensors[output_index].size_bytes;
    *output_copy = malloc(*output_size);
    if (*output_copy == NULL) {
        fprintf(stderr, "output allocation failed\n");
        goto cleanup;
    }
    memcpy(*output_copy, mem.tensor_ptrs[output_index], *output_size);
    ok = 1;

cleanup:
    if (cmsis_prepared)
        (void)tigris_cmsis_nn_deinit(&mem);
    free(tensor_ptrs);
    free(slow);
    free(fast);
    return ok;
}

int main(int argc, char **argv)
{
    const char *input_path;
    long plan_len = 0;
    uint8_t *plan_buffer;
    tigris_plan_t plan;
    int8_t *reference = NULL;
    int8_t *cmsis = NULL;
    uint32_t reference_size = 0;
    uint32_t cmsis_size = 0;
    int max_abs = 0;
    int pass;

    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: %s <plan.tgrs> [input_i8.bin]\n", argv[0]);
        return 2;
    }
    input_path = argc == 3 ? argv[2] : NULL;
    plan_buffer = read_file(argv[1], &plan_len);
    if (plan_buffer == NULL)
        return 2;
    if (tigris_plan_load(
            plan_buffer, (uint32_t)plan_len, &plan) != TIGRIS_OK) {
        fprintf(stderr, "plan load failed\n");
        free(plan_buffer);
        return 2;
    }
    if (!run_backend(
            &plan, tigris_dispatch_kernel_s8, 0, input_path,
            &reference, &reference_size)
            || !run_backend(
                &plan, tigris_dispatch_kernel_cmsis_nn, 1, input_path,
                &cmsis, &cmsis_size)) {
        free(cmsis);
        free(reference);
        free(plan_buffer);
        return 2;
    }
    if (reference_size != cmsis_size) {
        fprintf(
            stderr, "output length mismatch: s8=%u cmsis=%u\n",
            reference_size, cmsis_size);
        free(cmsis);
        free(reference);
        free(plan_buffer);
        return 1;
    }

    for (uint32_t i = 0; i < reference_size; ++i) {
        int difference = cmsis[i] - reference[i];
        if (difference < 0)
            difference = -difference;
        if (difference > max_abs)
            max_abs = difference;
    }
    pass = max_abs <= INT8_TOL;
    printf(
        "CMSIS-NN host parity: ops=%u output_bytes=%u max_abs_diff=%d %s\n",
        plan.header->num_ops, reference_size, max_abs,
        pass ? "PASS" : "FAIL");

    free(cmsis);
    free(reference);
    free(plan_buffer);
    return pass ? 0 : 1;
}
