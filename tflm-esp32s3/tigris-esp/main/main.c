/*
 * TiGrIS ESP32-S3 Benchmark Harness
 *
 * Loads a .tgrs plan from the "plan" flash partition, runs warmup + N timed
 * inference runs, reports min/max/mean/stdev latency.
 *
 * Kernel selection via build define:
 *   -DBENCH_KERNEL=esp_nn   tigris_dispatch_kernel_esp_nn (int8 + ESP-NN)
 *   -DBENCH_KERNEL=s8       tigris_dispatch_kernel_s8     (int8 reference)
 *   default                 tigris_dispatch_kernel         (f32 reference)
 *
 * Machine-parseable output:
 *   BENCH_RESULT:latency_mean_ms=X,latency_min_ms=X,latency_max_ms=X,...
 *   BENCH_DONE
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>

#include "esp_partition.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "hal/wdt_hal.h"
#include "soc/timer_group_reg.h"

#include "tigris.h"
#include "tigris_loader.h"
#include "tigris_mem.h"
#include "tigris_executor.h"
#include "tigris_kernels.h"
#include "tigris_kernels_s8.h"
#ifdef TIGRIS_HAS_ESP_NN
#include "tigris_kernels_esp_nn.h"
#endif

static const char *TAG = "bench";

#define WARMUP_RUNS 3
#define BENCH_RUNS  10

/* Watchdog suppression: background task that continuously feeds and disables
 * all WDTs. This handles cases where FreeRTOS or bootloader re-enables
 * TG0 WDT during long-running inference. */
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "soc/rtc_cntl_reg.h"

static int progress_count = 0;

static void feed_and_disable_all_wdts(void) {
    wdt_hal_context_t ctx;

    /* TG0 MWDT: feed and disable */
    REG_WRITE(TIMG_WDTWPROTECT_REG(0), TIMG_WDT_WKEY_VALUE);
    REG_WRITE(TIMG_WDTFEED_REG(0), 1);
    REG_WRITE(TIMG_WDTWPROTECT_REG(0), 0);

    wdt_hal_init(&ctx, WDT_MWDT0, 0, false);
    wdt_hal_write_protect_disable(&ctx);
    wdt_hal_set_flashboot_en(&ctx, false);
    wdt_hal_disable(&ctx);
    wdt_hal_write_protect_enable(&ctx);

    /* TG1 MWDT: feed and disable */
    REG_WRITE(TIMG_WDTWPROTECT_REG(1), TIMG_WDT_WKEY_VALUE);
    REG_WRITE(TIMG_WDTFEED_REG(1), 1);
    REG_WRITE(TIMG_WDTWPROTECT_REG(1), 0);

    wdt_hal_init(&ctx, WDT_MWDT1, 0, false);
    wdt_hal_write_protect_disable(&ctx);
    wdt_hal_set_flashboot_en(&ctx, false);
    wdt_hal_disable(&ctx);
    wdt_hal_write_protect_enable(&ctx);

    /* RTC WDT */
    wdt_hal_init(&ctx, WDT_RWDT, 0, false);
    wdt_hal_write_protect_disable(&ctx);
    wdt_hal_set_flashboot_en(&ctx, false);
    wdt_hal_disable(&ctx);
    wdt_hal_write_protect_enable(&ctx);

    /* Super WDT: disable auto feed to prevent it from resetting */
    REG_WRITE(RTC_CNTL_SWD_WPROTECT_REG, RTC_CNTL_SWD_WKEY_VALUE);
    SET_PERI_REG_MASK(RTC_CNTL_SWD_CONF_REG, RTC_CNTL_SWD_AUTO_FEED_EN);
    REG_WRITE(RTC_CNTL_SWD_WPROTECT_REG, 0);
}

/* Background task: feed+disable all WDTs every 500ms */
static void wdt_killer_task(void *arg) {
    (void)arg;
    for (;;) {
        feed_and_disable_all_wdts();
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}

static void start_wdt_killer(void) {
    xTaskCreatePinnedToCore(wdt_killer_task, "wdt_kill", 2048, NULL,
                            configMAX_PRIORITIES - 1, NULL, 1);
}

void tigris_feed_wdt(void) {
    progress_count++;
    feed_and_disable_all_wdts();
}

/* Select kernel based on build defines */
static const char *kernel_name(void) {
#if defined(BENCH_KERNEL_ESP_NN)
    return "esp_nn";
#elif defined(BENCH_KERNEL_S8)
    return "s8_ref";
#else
    return "f32_ref";
#endif
}

static tigris_kernel_fn select_kernel(int is_quantized) {
#if defined(BENCH_KERNEL_ESP_NN)
    (void)is_quantized;
#ifdef TIGRIS_HAS_ESP_NN
    return tigris_dispatch_kernel_esp_nn;
#else
    ESP_LOGE(TAG, "ESP-NN requested but not compiled in");
    return NULL;
#endif
#elif defined(BENCH_KERNEL_S8)
    (void)is_quantized;
    return tigris_dispatch_kernel_s8;
#else
    (void)is_quantized;
    return tigris_dispatch_kernel;
#endif
}

/* Run a single inference, return latency in microseconds. Re-inits mem state.
 * If out_stats is non-NULL, the exec stats from this run are copied out. */
static int64_t run_once(tigris_plan_t *plan, tigris_mem_t *mem,
                        tigris_kernel_fn dispatch, int is_quantized,
                        tigris_exec_stats_t *out_stats) {
    /* Re-init memory state (reset both arenas + tensor_ptrs) */
    tigris_mem_init(mem, mem->tensor_ptrs, mem->num_tensors,
                    mem->fast_base, mem->fast_size,
                    mem->slow_base, mem->slow_size);

    /* Allocate + fill inputs */
    for (uint8_t i = 0; i < plan->header->num_model_inputs; i++) {
        uint16_t tidx = plan->model_inputs[i];
        uint32_t sz = plan->tensors[tidx].size_bytes;
        tigris_mem_alloc_slow(mem, tidx, sz);

        if (is_quantized) {
            /* Fill with int8 value 1, matches prepare.py reference generator */
            memset(mem->tensor_ptrs[tidx], 1, sz);
        } else {
            float *data = (float *)mem->tensor_ptrs[tidx];
            uint32_t n = sz / sizeof(float);
            for (uint32_t j = 0; j < n; j++)
                data[j] = 1.0f;
        }
    }

    progress_count = 0;

    tigris_exec_stats_t stats;
    int64_t t0 = esp_timer_get_time();
    tigris_exec_error_t err = tigris_run(plan, mem, dispatch, NULL, &stats);
    int64_t t1 = esp_timer_get_time();

    if (err != TIGRIS_EXEC_OK) {
        ESP_LOGE(TAG, "inference failed: %s", tigris_exec_error_str(err));
        return -1;
    }

    if (out_stats)
        *out_stats = stats;

    return t1 - t0;
}

void app_main(void) {
    feed_and_disable_all_wdts();
    start_wdt_killer();

    printf("\nTiGrIS Benchmark (kernel=%s)\n\n", kernel_name());

    /* 1. Load plan from flash partition */
    const esp_partition_t *part = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, 0x40, "plan");
    if (!part) {
        ESP_LOGE(TAG, "partition 'plan' not found");
        return;
    }

    const void *mapped_ptr = NULL;
    esp_partition_mmap_handle_t mmap_handle;
    esp_err_t err = esp_partition_mmap(
        part, 0, part->size, ESP_PARTITION_MMAP_DATA, &mapped_ptr, &mmap_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "mmap failed: %s", esp_err_to_name(err));
        return;
    }

    const tigris_file_header_t *raw_hdr = (const tigris_file_header_t *)mapped_ptr;
    uint32_t plan_size = raw_hdr->file_size;
    if (plan_size < sizeof(tigris_file_header_t) || plan_size > part->size) {
        ESP_LOGE(TAG, "bad plan file_size: %lu", (unsigned long)plan_size);
        esp_partition_munmap(mmap_handle);
        return;
    }

    tigris_plan_t plan;
    tigris_error_t perr = tigris_plan_load(
        (const uint8_t *)mapped_ptr, plan_size, &plan);
    if (perr != TIGRIS_OK) {
        ESP_LOGE(TAG, "plan load failed: %s", tigris_error_str(perr));
        esp_partition_munmap(mmap_handle);
        return;
    }

    const char *model_name = tigris_model_name(&plan);
    uint8_t input_dtype = plan.tensors[plan.model_inputs[0]].dtype;
    int is_quantized = (input_dtype == 3);

    printf("Model:   %s\n", model_name);
    printf("Type:    %s\n", is_quantized ? "int8" : "f32");
    printf("Kernel:  %s\n", kernel_name());
    printf("Ops:     %u\n", plan.header->num_ops);
    printf("Stages:  %u\n", plan.header->num_stages);
    printf("Budget:  %lu bytes\n", (unsigned long)plan.header->budget);

    /* 2. Select kernel */
    tigris_kernel_fn dispatch = select_kernel(is_quantized);
    if (!dispatch) {
        esp_partition_munmap(mmap_handle);
        return;
    }

    /* 3. Allocate buffers */
    uint32_t fast_size = plan.header->budget;
    if (fast_size == 0)
        fast_size = 64 * 1024;
    fast_size += tigris_weight_decompression_overhead(&plan);

    /* Cap at available internal SRAM (leave 16 KB for stack/heap) */
    uint32_t avail_internal = heap_caps_get_largest_free_block(
                                  MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (avail_internal > 16 * 1024)
        avail_internal -= 16 * 1024;
    if (fast_size > avail_internal) {
        printf("Note: capping fast_size %luK -> %luK (internal SRAM limit)\n",
               (unsigned long)(fast_size / 1024),
               (unsigned long)(avail_internal / 1024));
        fast_size = avail_internal;
    }

    uint32_t slow_size;
#if CONFIG_SPIRAM
    slow_size = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);
    if (slow_size > 2 * 1024 * 1024)
        slow_size -= 2 * 1024 * 1024;  /* leave headroom for ESP-NN scratch */
    else if (slow_size > 64 * 1024)
        slow_size -= 16 * 1024;
#else
    slow_size = heap_caps_get_largest_free_block(
                    MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (slow_size > 32 * 1024)
        slow_size -= 16 * 1024;
#endif
    if (slow_size < 64 * 1024)
        slow_size = 64 * 1024;

    void *fast_buf = heap_caps_malloc(fast_size,
                                      MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!fast_buf) {
        ESP_LOGE(TAG, "alloc fast_buf failed (%lu)", (unsigned long)fast_size);
        esp_partition_munmap(mmap_handle);
        return;
    }

#if CONFIG_SPIRAM
    void *slow_buf = heap_caps_malloc(slow_size, MALLOC_CAP_SPIRAM);
#else
    void *slow_buf = heap_caps_malloc(slow_size,
                                      MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
#endif
    if (!slow_buf) {
        ESP_LOGE(TAG, "alloc slow_buf failed (%lu)", (unsigned long)slow_size);
        heap_caps_free(fast_buf);
        esp_partition_munmap(mmap_handle);
        return;
    }

    uint16_t num_t = plan.header->num_tensors;
#if CONFIG_SPIRAM
    void **tensor_ptrs = heap_caps_calloc(num_t, sizeof(void *),
                                          MALLOC_CAP_SPIRAM);
#else
    void **tensor_ptrs = heap_caps_calloc(num_t, sizeof(void *),
                                          MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
#endif
    if (!tensor_ptrs) {
        ESP_LOGE(TAG, "alloc tensor_ptrs failed");
        heap_caps_free(slow_buf);
        heap_caps_free(fast_buf);
        esp_partition_munmap(mmap_handle);
        return;
    }

    tigris_mem_t mem;
    tigris_mem_error_t merr = tigris_mem_init(
        &mem, tensor_ptrs, num_t,
        fast_buf, fast_size,
        slow_buf, slow_size);
    if (merr != TIGRIS_MEM_OK) {
        ESP_LOGE(TAG, "mem init failed: %s", tigris_mem_error_str(merr));
        goto cleanup;
    }

    /* 3b. Pre-allocate ESP-NN scratch buffers from arena top */
#if defined(TIGRIS_HAS_ESP_NN) && defined(BENCH_KERNEL_ESP_NN)
    if (tigris_esp_nn_prepare(&plan, &mem) != 0) {
        ESP_LOGE(TAG, "esp_nn_prepare failed (arena too small for scratch)");
        goto cleanup;
    }
    ESP_LOGI(TAG, "ESP-NN scratch carved from arena, fast_size reduced to %lu",
             (unsigned long)mem.fast_size);
#endif

    printf("\nSRAM budget: %lu KB\n", (unsigned long)(fast_size / 1024));
    printf("Slow buffer: %lu KB\n\n", (unsigned long)(slow_size / 1024));

    /* 4. Warmup runs */
    printf("Warmup (%d runs)...\n", WARMUP_RUNS);
    for (int i = 0; i < WARMUP_RUNS; i++) {
        int64_t us = run_once(&plan, &mem, dispatch, is_quantized, NULL);
        if (us < 0) goto cleanup;
        printf("  warmup[%d]: %.2f ms\n", i, (float)us / 1000.0f);
    }

    /* 5. Timed runs */
    printf("\nBenchmark (%d runs)...\n", BENCH_RUNS);
    float latencies[BENCH_RUNS];
    tigris_exec_stats_t last_stats = {0};
    for (int i = 0; i < BENCH_RUNS; i++) {
        tigris_exec_stats_t *sp = (i == BENCH_RUNS - 1) ? &last_stats : NULL;
        int64_t us = run_once(&plan, &mem, dispatch, is_quantized, sp);
        if (us < 0) goto cleanup;
        latencies[i] = (float)us / 1000.0f;
        printf("  run[%d]: %.2f ms\n", i, latencies[i]);
    }

    /* 6. Compute statistics */
    float sum = 0, min_lat = latencies[0], max_lat = latencies[0];
    for (int i = 0; i < BENCH_RUNS; i++) {
        sum += latencies[i];
        if (latencies[i] < min_lat) min_lat = latencies[i];
        if (latencies[i] > max_lat) max_lat = latencies[i];
    }
    float mean = sum / BENCH_RUNS;

    float var_sum = 0;
    for (int i = 0; i < BENCH_RUNS; i++) {
        float diff = latencies[i] - mean;
        var_sum += diff * diff;
    }
    float stdev = sqrtf(var_sum / BENCH_RUNS);

    uint32_t sram_budget = plan.header->budget;
    uint32_t sram_actual = fast_size;
    uint32_t plan_flash = plan_size;

    /* 7. Print output values (for accuracy validation) */
    printf("\nOutput values:\n");
    if (plan.header->num_model_outputs > 0) {
        for (uint8_t i = 0; i < plan.header->num_model_outputs; i++) {
            uint16_t tidx = plan.model_outputs[i];
            void *out_ptr = mem.tensor_ptrs[tidx];
            if (!out_ptr) continue;

            if (is_quantized) {
                int8_t *out = (int8_t *)out_ptr;
                uint32_t n = plan.tensors[tidx].size_bytes;
                printf("  OUTPUT_I8:");
                for (uint32_t j = 0; j < n; j++)
                    printf(" %d", (int)out[j]);
                printf("\n");
            } else {
                float *out = (float *)out_ptr;
                uint32_t n = plan.tensors[tidx].size_bytes / sizeof(float);
                printf("  OUTPUT_F32:");
                for (uint32_t j = 0; j < n; j++)
                    printf(" %.6f", out[j]);
                printf("\n");
            }
        }
    } else {
        /* Fallback: scan for small non-null tensors that look like outputs */
        uint16_t last_op_idx = plan.header->num_ops - 1;
        const tigris_op_t *last_op = &plan.ops[last_op_idx];
        const uint16_t *last_outs = tigris_op_outputs(&plan, last_op);
        uint16_t tidx = last_outs[0];
        void *out_ptr = mem.tensor_ptrs[tidx];
        if (out_ptr) {
            if (is_quantized) {
                int8_t *out = (int8_t *)out_ptr;
                uint32_t n = plan.tensors[tidx].size_bytes;
                printf("  OUTPUT_I8:");
                for (uint32_t j = 0; j < n; j++)
                    printf(" %d", (int)out[j]);
                printf("\n");
            } else {
                float *out = (float *)out_ptr;
                uint32_t n = plan.tensors[tidx].size_bytes / sizeof(float);
                printf("  OUTPUT_F32:");
                for (uint32_t j = 0; j < n; j++)
                    printf(" %.6f", out[j]);
                printf("\n");
            }
        }
    }

    /* 8. Machine-parseable result line */
    printf("\nBENCH_RESULT:"
           "framework=tigris,"
           "kernel=%s,"
           "dtype=%s,"
           "model=%s,"
           "latency_mean_ms=%.2f,"
           "latency_min_ms=%.2f,"
           "latency_max_ms=%.2f,"
           "latency_stdev_ms=%.2f,"
           "sram_budget_kb=%lu,"
           "sram_actual_kb=%lu,"
           "plan_flash_kb=%lu,"
           "runs=%d,"
           "stages_normal=%u,"
           "stages_tiled=%u,"
           "stages_chain=%u,"
           "total_tiles=%u\n",
           kernel_name(),
           is_quantized ? "int8" : "f32",
           model_name,
           mean, min_lat, max_lat, stdev,
           (unsigned long)(sram_budget / 1024),
           (unsigned long)(sram_actual / 1024),
           (unsigned long)(plan_flash / 1024),
           BENCH_RUNS,
           (unsigned)last_stats.stages_normal,
           (unsigned)last_stats.stages_tiled,
           (unsigned)last_stats.stages_chain,
           (unsigned)last_stats.total_tiles);

    printf("BENCH_DONE\n");

cleanup:
    heap_caps_free(tensor_ptrs);
    heap_caps_free(slow_buf);
    heap_caps_free(fast_buf);
    esp_partition_munmap(mmap_handle);
}
