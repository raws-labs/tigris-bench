/*
 * Standalone bare-metal TiGrIS runner for an emulated Arm Cortex-M55
 * (QEMU mps3-an547 / Corstone-300). Loads a .tgrs plan from DDR (0x60000000,
 * placed there by QEMU -device loader, modelling external XIP flash), tiles it
 * with the s8_ref kernel, and reports the output checksum + arena usage via
 * ARM semihosting. Wall-clock latency is intentionally NOT reported: QEMU is
 * functional, not cycle-accurate.
 *
 * Build-time configuration (see build.sh):
 *   -DMODEL_NAME=\"...\"   banner label
 *   -DPLAN_LEN=N          byte length of the plan loaded at 0x60000000
 *   -DFAST_KB=N           fast tile arena, KiB (on-chip SRAM)
 *   -DSLOW_KB=N           slow tier, KiB (on-chip SRAM, unless -DSLOW_DDR)
 *   -DSLOW_DDR            put the slow tier in external DDR (0x64000000, 32 MiB)
 *                         instead of on-chip SRAM - for models whose cross-stage
 *                         tensors exceed on-chip SRAM (e.g. ResNet-50).
 */
#include <stdint.h>
#include <stddef.h>

#include "tigris.h"
#include "tigris_loader.h"
#include "tigris_mem.h"
#include "tigris_executor.h"
#include "tigris_kernels_s8.h"

/* ---- ARM semihosting output ---- */
static void sh_write0(const char *s)
{
    register int op __asm__("r0") = 0x04;           /* SYS_WRITE0 */
    register const char *a __asm__("r1") = s;
    __asm__ volatile("bkpt 0xAB" : "+r"(op) : "r"(a) : "memory");
}
static void sh_exit(void)
{
    register int op __asm__("r0") = 0x18;            /* SYS_EXIT */
    register int a __asm__("r1") = 0x20026;          /* ADP_Stopped_ApplicationExit */
    __asm__ volatile("bkpt 0xAB" : "+r"(op) : "r"(a) : "memory");
}
static char g_nbuf[16];
static const char *itoa_(long v)
{
    char *p = g_nbuf + 15;
    *p = 0;
    int neg = v < 0;
    unsigned long u = neg ? (unsigned long)(-v) : (unsigned long)v;
    if (!u) *--p = '0';
    while (u) { *--p = (char)('0' + u % 10); u /= 10; }
    if (neg) *--p = '-';
    return p;
}
static void P(const char *s)  { sh_write0(s); }
static void PI(long v)        { sh_write0(itoa_(v)); }

#ifndef MODEL_NAME
#define MODEL_NAME "model"
#endif
#ifndef PLAN_ADDR
#define PLAN_ADDR 0x60000000u
#endif
#ifndef PLAN_LEN
#define PLAN_LEN 0u
#endif
#ifndef FAST_KB
#define FAST_KB 192u
#endif
#ifndef SLOW_KB
#define SLOW_KB 256u
#endif

#define FAST_BYTES (FAST_KB * 1024u)
static uint8_t s_fast[FAST_BYTES] __attribute__((aligned(16)));
static void   *s_ptrs[512];
static uint8_t s_ws[64u * 1024u]  __attribute__((aligned(16)));

#ifdef SLOW_DDR
#define SLOW_ADDR  0x64000000u                  /* DDR, past the plan at 0x60000000 */
#define SLOW_BYTES (32u * 1024u * 1024u)
#else
#define SLOW_BYTES (SLOW_KB * 1024u)
static uint8_t s_slow[SLOW_BYTES] __attribute__((aligned(16)));  /* on-chip SRAM */
#endif

int main(void)
{
    P("\nBENCH_START " MODEL_NAME " @ Cortex-M55 (QEMU mps3-an547)\n");

    const unsigned char *pd = (const unsigned char *)(uintptr_t)PLAN_ADDR;
    tigris_plan_t plan;
    tigris_error_t perr = tigris_plan_load(pd, (uint32_t)PLAN_LEN, &plan);
    if (perr != TIGRIS_OK) { P("PLAN_LOAD_FAIL "); PI(perr); P("\n"); sh_exit(); return 1; }

    P("Ops: ");      PI(plan.header->num_ops);
    P("  Stages: "); PI(plan.header->num_stages);
    P("  Tensors: ");PI(plan.header->num_tensors);
    P("  Budget: "); PI(plan.header->budget); P(" bytes\n");

    uint32_t nt = plan.header->num_tensors;
    if (nt > (sizeof(s_ptrs) / sizeof(s_ptrs[0]))) { P("TOO_MANY_TENSORS\n"); sh_exit(); return 1; }

    uint32_t fast_size = plan.header->budget + tigris_weight_decompression_overhead(&plan);
    if (fast_size > FAST_BYTES) { P("FAST_ARENA_TOO_SMALL need "); PI(fast_size); P("\n"); sh_exit(); return 1; }
#ifdef SLOW_DDR
    uint8_t *slow_buf = (uint8_t *)(uintptr_t)SLOW_ADDR;
#else
    uint8_t *slow_buf = s_slow;
#endif
    uint32_t slow_size = SLOW_BYTES;

    tigris_mem_t mem;
    tigris_mem_error_t merr = tigris_mem_init(&mem, s_ptrs, nt,
                                              s_fast, fast_size, slow_buf, slow_size);
    if (merr != TIGRIS_MEM_OK) { P("MEM_INIT_FAIL "); PI(merr); P("\n"); sh_exit(); return 1; }

    /* Fill inputs with the same deterministic pattern as host tools/host_probe,
     * so the output checksum can be compared bit-for-bit against the host. */
    for (uint8_t i = 0; i < plan.header->num_model_inputs; i++) {
        uint16_t tidx = plan.model_inputs[i];
        uint32_t sz = plan.tensors[tidx].size_bytes;
        tigris_mem_alloc_slow(&mem, tidx, sz);
        int8_t *d = (int8_t *)s_ptrs[tidx];
        for (uint32_t j = 0; j < sz; j++)
            d[j] = (int8_t)(((j * 73u + 17u) % 201u) - 100u);
    }

    size_t wss = tigris_executor_workspace_required(&plan);
    if (wss > sizeof(s_ws)) { P("WORKSPACE_TOO_SMALL need "); PI((long)wss); P("\n"); sh_exit(); return 1; }

    tigris_exec_stats_t stats;
    P("Running (tiled, s8_ref)...\n");
    tigris_exec_error_t eerr = tigris_run_with_workspace_buffer(
        &plan, &mem, tigris_dispatch_kernel_s8, NULL, &stats, s_ws, wss);
    if (eerr != TIGRIS_EXEC_OK) { P("EXEC_FAIL "); PI(eerr); P("\n"); sh_exit(); return 1; }

    /* Output = last op's first output tensor (matches host tools/host_probe). */
    uint16_t last_op_idx = (uint16_t)(plan.header->num_ops - 1);
    const tigris_op_t *last_op = &plan.ops[last_op_idx];
    const uint16_t *last_outs = tigris_op_outputs(&plan, last_op);
    uint16_t otidx = last_outs[0];
    const int8_t *out = (const int8_t *)s_ptrs[otidx];
    uint32_t n = plan.tensors[otidx].size_bytes;

    uint32_t csum = 0;
    long mn = 127, mx = -128, nonmin = 0;
    for (uint32_t j = 0; j < n; j++) {
        int v = out[j];
        csum += (uint32_t)(uint8_t)v * (j + 1u);
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        if (v != -128) nonmin++;
    }
    P("OUTPUT n="); PI(n); P(" checksum="); PI((long)csum);
    P(" min="); PI(mn); P(" max="); PI(mx); P(" nonmin="); PI(nonmin); P("\n");
    P("ARENA fast_peak="); PI(mem.fast_peak);
    P(" slow_used="); PI(mem.slow_used);
    P(" total_sram="); PI((long)mem.fast_peak + (long)mem.slow_used);
    P(" bytes (fast_cap="); PI(fast_size); P(" slow_cap="); PI(slow_size); P(")\n");
    P("BENCH_DONE\n");
    sh_exit();
    return 0;
}
