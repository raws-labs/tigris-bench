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
