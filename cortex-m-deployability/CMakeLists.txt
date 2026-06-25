cmake_minimum_required(VERSION 3.16)

# ---- configuration (override on the cmake command line) ----
set(TIGRIS_BOARD "nucleo_h753zi" CACHE STRING "target board (boards/<name>)")
set(BENCH_FRAMEWORK "tigris" CACHE STRING "framework under test: tigris | tflm")
set(BENCH_KERNEL "cmsis_nn" CACHE STRING "TiGrIS kernel backend: cmsis_nn | s8_ref")
set(TFLM_MODEL "ds_cnn" CACHE STRING "TFLM model under test; expects <name>_tflite_i8.h in MODELS_DIR")
set(TFLM_ARENA_BYTES "131072" CACHE STRING "TFLM tensor arena size (shrink for small-RAM boards, e.g. F446)")
set(TIGRIS_PLAN
    "${CMAKE_CURRENT_SOURCE_DIR}/../tflm-esp32s3/models/output/ds_cnn_i8.tgrs"
    CACHE FILEPATH "the .tgrs plan to embed in flash")
set(TIGRIS_FAST_ARENA_BYTES "131072" CACHE STRING "fast arena size (bytes)")
set(TIGRIS_SLOW_ARENA_BYTES "262144" CACHE STRING "slow arena size (bytes)")
# Optimization level for ALL benchmark code (TiGrIS runtime, CMSIS-NN, harness, TFLM
# glue). MUST be consistent across boards and frameworks for a fair comparison: TFLM's
# own kernels build at -O2, and without this the clean-CMake path defaults to -O0 (no
# CMAKE_BUILD_TYPE set), while the pico-sdk path defaults to -O3. -O2 matches TFLM.
set(BENCH_OPT_LEVEL "-O2" CACHE STRING "optimization level for all benchmark code")

project(cortexm_bench C CXX ASM)
set(CMAKE_C_STANDARD 11)

# ---- vendored CMSIS locations (populated by third_party/fetch.sh) ----
set(CMSIS_NN_DIR        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/CMSIS-NN)
set(CMSIS_CORE_DIR      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/CMSIS_6/CMSIS/Core)
set(CMSIS_DEVICE_H7_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cmsis-device-h7)
set(CMSIS_DEVICE_F4_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cmsis-device-f4)

# ---- board: sets BOARD_MCU_FLAGS, BOARD_STARTUP/SYSTEM/BSP/LINKER_SCRIPT,
#      BOARD_DEFINES, BOARD_INCLUDES ----
include(${CMAKE_CURRENT_SOURCE_DIR}/boards/${TIGRIS_BOARD}/board.cmake)

# Apply the per-core ABI flags + the consistent optimization level to EVERY target.
add_compile_options(${BOARD_MCU_FLAGS} ${BENCH_OPT_LEVEL} -ffunction-sections -fdata-sections)
add_link_options(
    ${BOARD_MCU_FLAGS}
    -T${BOARD_LINKER_SCRIPT}
    --specs=nano.specs
    -u _printf_float
    -Wl,--gc-sections
    -Wl,--print-memory-usage
    -Wl,-Map=${CMAKE_BINARY_DIR}/bench.map)

if(BENCH_FRAMEWORK STREQUAL "tigris")
    # ======================= TiGrIS firmware ============================
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    add_subdirectory(third_party)                          # cmsis-nn static lib

    set(TIGRIS_HAS_CMSIS_NN ON CACHE BOOL "" FORCE)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../tigris-runtime
                     ${CMAKE_BINARY_DIR}/tigris_runtime)

    if(BENCH_KERNEL STREQUAL "cmsis_nn")
        set(KERNEL_DEFINE BENCH_KERNEL_CMSIS_NN)
    elseif(BENCH_KERNEL STREQUAL "s8_ref")
        set(KERNEL_DEFINE BENCH_KERNEL_S8)
    else()
        message(FATAL_ERROR "BENCH_KERNEL must be 'cmsis_nn' or 's8_ref'")
    endif()

    set(PLAN_BLOB ${CMAKE_BINARY_DIR}/model_blob.c)
    add_custom_command(
        OUTPUT ${PLAN_BLOB}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/bin2c.py
                ${TIGRIS_PLAN} ${PLAN_BLOB} --symbol g_tigris_plan
        DEPENDS ${TIGRIS_PLAN} ${CMAKE_CURRENT_SOURCE_DIR}/tools/bin2c.py
        COMMENT "Embedding plan: ${TIGRIS_PLAN}")

    add_executable(tigris_bench
        harness/main.c ${PLAN_BLOB}
        ${BOARD_BSP} ${BOARD_SYSTEM} ${BOARD_STARTUP})
    target_include_directories(tigris_bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/harness ${BOARD_INCLUDES})
    target_compile_definitions(tigris_bench PRIVATE
        ${KERNEL_DEFINE} ${BOARD_DEFINES}
        TIGRIS_FAST_ARENA_BYTES=${TIGRIS_FAST_ARENA_BYTES}
        TIGRIS_SLOW_ARENA_BYTES=${TIGRIS_SLOW_ARENA_BYTES})
    target_link_libraries(tigris_bench PRIVATE tigris_runtime cmsis-nn)
    set_target_properties(tigris_bench PROPERTIES OUTPUT_NAME tigris_bench SUFFIX ".elf")
    add_custom_command(TARGET tigris_bench POST_BUILD
        COMMAND ${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:tigris_bench> ${CMAKE_BINARY_DIR}/tigris_bench.bin
        COMMAND ${CMAKE_SIZE} $<TARGET_FILE:tigris_bench>
        COMMENT "tigris_bench.bin + size")

elseif(BENCH_FRAMEWORK STREQUAL "tflm")
    # ===================== TFLite Micro baseline ========================
    # Uses the prebuilt libtensorflow-microlite.a (built by the user; see
    # BUILD.md). Must match the lib's flags: TF_LITE_STATIC_MEMORY + no
    # rtti/exceptions/threadsafe-statics.
    set(TFLM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tflite-micro)
    # Pick the prebuilt microlite lib for this board's core (BOARD_TFLM_ARCH set
    # in board.cmake). Build it with third_party/tflite-micro's make, e.g.
    #   make ... TARGET=cortex_m_generic TARGET_ARCH=cortex-m4+fp ... microlite
    if(NOT DEFINED BOARD_TFLM_ARCH)
        message(FATAL_ERROR "board.cmake must set BOARD_TFLM_ARCH (e.g. cortex-m7+fp)")
    endif()
    # TFLM build type goes in the gen/ path. release_with_logs (TFLM's recommended
    # benchmark build: -DNDEBUG, keeps error strings) removes the debug asserts/
    # checks that the 'default' build leaves on - those make 'default' ~5-10% slower
    # and would unfairly handicap the TFLM latency baseline.
    set(TFLM_BUILD_TYPE "release_with_logs" CACHE STRING "TFLM microlite build type (matches gen/ path)")
    set(TFLM_LIB ${TFLM_DIR}/gen/cortex_m_generic_${BOARD_TFLM_ARCH}_${TFLM_BUILD_TYPE}_cmsis_nn_gcc/lib/libtensorflow-microlite.a)
    set(TFLM_DL  ${TFLM_DIR}/tensorflow/lite/micro/tools/make/downloads)
    if(NOT EXISTS ${TFLM_LIB})
        message(FATAL_ERROR "TFLM lib not found: ${TFLM_LIB}\nBuild it (see BUILD.md): "
                "make ... TARGET_ARCH=${BOARD_TFLM_ARCH} BUILD_TYPE=${TFLM_BUILD_TYPE} ... microlite")
    endif()
    set(MODELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../tflm-esp32s3/models/output)

    add_executable(tflm_bench
        tflm/main.cc
        ${BOARD_BSP} ${BOARD_SYSTEM} ${BOARD_STARTUP})
    set_target_properties(tflm_bench PROPERTIES
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
        OUTPUT_NAME tflm_bench SUFFIX ".elf")
    target_compile_options(tflm_bench PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti -fno-exceptions -fno-threadsafe-statics>)
    target_compile_definitions(tflm_bench PRIVATE
        TF_LITE_STATIC_MEMORY ${BOARD_DEFINES} TFLM_ARENA_BYTES=${TFLM_ARENA_BYTES}
        TFLM_MODEL_HEADER="${TFLM_MODEL}_tflite_i8.h"
        TFLM_MODEL_SYM=${TFLM_MODEL}_tflite_i8
        TFLM_MODEL_NAME="${TFLM_MODEL}")
    target_include_directories(tflm_bench SYSTEM PRIVATE
        ${TFLM_DIR}
        ${TFLM_DL}/flatbuffers/include
        ${TFLM_DL}/gemmlowp
        ${TFLM_DL}/ruy)
    target_include_directories(tflm_bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tflm
        ${CMAKE_CURRENT_SOURCE_DIR}/harness
        ${MODELS_DIR} ${BOARD_INCLUDES})
    target_link_libraries(tflm_bench PRIVATE ${TFLM_LIB})
    add_custom_command(TARGET tflm_bench POST_BUILD
        COMMAND ${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:tflm_bench> ${CMAKE_BINARY_DIR}/tflm_bench.bin
        COMMAND ${CMAKE_SIZE} $<TARGET_FILE:tflm_bench>
        COMMENT "tflm_bench.bin + size")

else()
    message(FATAL_ERROR "BENCH_FRAMEWORK must be 'tigris' or 'tflm'")
endif()
