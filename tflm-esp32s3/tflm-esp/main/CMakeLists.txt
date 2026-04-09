idf_component_register(
    SRCS "main.cc"
    INCLUDE_DIRS "."
    PRIV_REQUIRES esp_timer heap
)

# Model selection: -DBENCH_WIDE=1 | -DBENCH_INT8=1 | default (f32)
if(BENCH_WIDE)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE BENCH_WIDE)
elseif(BENCH_INT8)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE BENCH_INT8)
endif()
