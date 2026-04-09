if(NOT DEFINED TIGRIS_RUNTIME_DIR)
    # Default: tigris-runtime is a sibling of tigris-bench.
    # From this file: components/ -> tigris-esp/ -> tflm-esp32s3/ -> tigris-bench/ -> parent/
    set(TIGRIS_RUNTIME_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../../tigris-runtime"
        CACHE PATH "Path to tigris-runtime root")
endif()

if(NOT EXISTS "${TIGRIS_RUNTIME_DIR}/include/tigris.h")
    message(FATAL_ERROR
        "tigris-runtime not found at ${TIGRIS_RUNTIME_DIR}\n"
        "Set -DTIGRIS_RUNTIME_DIR=/path/to/tigris-runtime")
endif()

idf_component_register(
    SRCS "${TIGRIS_RUNTIME_DIR}/src/tigris_loader.c"
         "${TIGRIS_RUNTIME_DIR}/src/tigris_mem.c"
         "${TIGRIS_RUNTIME_DIR}/src/tigris_executor.c"
         "${TIGRIS_RUNTIME_DIR}/src/tigris_kernels.c"
         "${TIGRIS_RUNTIME_DIR}/src/tigris_kernels_s8.c"
         "${TIGRIS_RUNTIME_DIR}/src/tigris_kernels_esp_nn.c"
         "${TIGRIS_RUNTIME_DIR}/src/tigris_lz4.c"
    INCLUDE_DIRS "${TIGRIS_RUNTIME_DIR}/include"
    PRIV_REQUIRES esp_system esp_timer
)

# Enable ESP-NN backend when esp-nn component is available
# (managed component name is espressif__esp-nn, local is esp-nn)
idf_build_get_property(components BUILD_COMPONENTS)
if("espressif__esp-nn" IN_LIST components)
    set(_ESP_NN_NAME "espressif__esp-nn")
elseif("esp-nn" IN_LIST components)
    set(_ESP_NN_NAME "esp-nn")
endif()
if(DEFINED _ESP_NN_NAME)
    target_compile_definitions(${COMPONENT_LIB} PUBLIC TIGRIS_HAS_ESP_NN)
    idf_component_get_property(esp_nn_lib ${_ESP_NN_NAME} COMPONENT_LIB)
    target_link_libraries(${COMPONENT_LIB} PUBLIC ${esp_nn_lib})
endif()
