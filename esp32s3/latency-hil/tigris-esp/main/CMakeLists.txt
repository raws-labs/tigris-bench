idf_component_register(
    SRCS "main.c"
    INCLUDE_DIRS "."
    REQUIRES tigris_runtime
    PRIV_REQUIRES esp_partition esp_timer heap
)

# Kernel selection: -DBENCH_KERNEL=esp_nn | s8 | f32 (default)
if(BENCH_KERNEL STREQUAL "esp_nn")
    target_compile_definitions(${COMPONENT_LIB} PRIVATE BENCH_KERNEL_ESP_NN)
elseif(BENCH_KERNEL STREQUAL "s8")
    target_compile_definitions(${COMPONENT_LIB} PRIVATE BENCH_KERNEL_S8)
endif()

# Optional run-count overrides (e.g. -DWARMUP_RUNS=0 -DBENCH_RUNS=1 for QEMU
# functional runs, where wall-clock latency is meaningless).
if(DEFINED WARMUP_RUNS)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE WARMUP_RUNS=${WARMUP_RUNS})
endif()
if(DEFINED BENCH_RUNS)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE BENCH_RUNS=${BENCH_RUNS})
endif()
