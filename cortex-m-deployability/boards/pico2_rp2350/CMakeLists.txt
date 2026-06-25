# Pico 2 (RP2350, Cortex-M33) build of the TiGrIS benchmark.
#
# Unlike the STM32 boards (driven by the top-level clean-CMake + CMSIS BSPs), the
# RP2350 needs the pico-sdk for its boot/XIP/USB-CDC, so it is a SEPARATE pico-sdk
# project that links the same TiGrIS runtime + CMSIS-NN and reuses the harness.
#
# Build (needs PICO_SDK_PATH + a built picotool):
#   PICO_SDK_PATH=~/pico/pico-sdk cmake -S boards/pico2_rp2350 -B build/pico_ts_cmsis \
#     -Dpicotool_DIR=~/pico/picotool/install/lib/cmake/picotool \
#     -DBENCH_KERNEL=cmsis_nn -DTIGRIS_PLAN=<abs path to .tgrs>
#   cmake --build build/pico_ts_cmsis -j      # -> tigris_pico_bench.uf2

cmake_minimum_required(VERSION 3.13)
set(PICO_BOARD pico2)   # RP2350; pico2_w would pull the cyw43 submodule (no wifi used)
# Pin the optimization level to match the STM32 clean-CMake path and TFLM (-O2). The
# pico-sdk defaults CMAKE_BUILD_TYPE=Release (-O3); override its Release flags BEFORE
# pico_sdk_init so the whole build (incl. the TiGrIS runtime + CMSIS-NN) is -O2.
set(BENCH_OPT_LEVEL "-O2" CACHE STRING "optimization level for all benchmark code")
set(CMAKE_C_FLAGS_RELEASE   "${BENCH_OPT_LEVEL} -DNDEBUG" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "${BENCH_OPT_LEVEL} -DNDEBUG" CACHE STRING "" FORCE)
include($ENV{PICO_SDK_PATH}/pico_sdk_init.cmake)
project(tigris_pico_bench C CXX ASM)
pico_sdk_init()

set(BENCH_ROOT ${CMAKE_CURRENT_LIST_DIR}/../..)
set(RT_DIR     ${BENCH_ROOT}/../../tigris-runtime)
set(NN_DIR     ${BENCH_ROOT}/third_party/CMSIS-NN)

set(BENCH_KERNEL "cmsis_nn" CACHE STRING "TiGrIS backend: cmsis_nn | s8_ref")
set(TIGRIS_PLAN  "" CACHE FILEPATH "the .tgrs plan embedded in flash")
set(TIGRIS_FAST_ARENA_BYTES "65536" CACHE STRING "fast arena (RP2350 has 520 KB SRAM)")
set(TIGRIS_SLOW_ARENA_BYTES "131072" CACHE STRING "slow arena")
set(WARMUP_RUNS "3" CACHE STRING "warmup runs (reduce for slow tiled models)")
set(BENCH_RUNS "30" CACHE STRING "timed runs (reduce for slow tiled models)")

if(NOT TIGRIS_PLAN)
    message(FATAL_ERROR "set -DTIGRIS_PLAN=<abs path to a .tgrs>")
endif()

# TiGrIS runtime + vendored CMSIS-NN (v7 is standalone: no CMSIS-Core/DSP dep).
# Compiled with the pico-sdk's M33 flags; CMSIS-NN auto-selects the DSP path from
# __ARM_FEATURE_DSP (cortex-m33 has it; no Helium/MVE on RP2350).
file(GLOB        RT_SRC ${RT_DIR}/src/*.c)
file(GLOB_RECURSE NN_SRC ${NN_DIR}/Source/*.c)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(PLAN_BLOB ${CMAKE_BINARY_DIR}/model_blob.c)
add_custom_command(
    OUTPUT ${PLAN_BLOB}
    COMMAND ${Python3_EXECUTABLE} ${BENCH_ROOT}/tools/bin2c.py
            ${TIGRIS_PLAN} ${PLAN_BLOB} --symbol g_tigris_plan
    DEPENDS ${TIGRIS_PLAN} ${BENCH_ROOT}/tools/bin2c.py)

add_executable(tigris_pico_bench
    ${BENCH_ROOT}/harness/main.c
    ${CMAKE_CURRENT_LIST_DIR}/platform_pico.c
    ${PLAN_BLOB}
    ${RT_SRC} ${NN_SRC})

target_include_directories(tigris_pico_bench PRIVATE
    ${BENCH_ROOT}/harness
    ${RT_DIR}/include
    ${NN_DIR}/Include)

if(BENCH_KERNEL STREQUAL "cmsis_nn")
    target_compile_definitions(tigris_pico_bench PRIVATE BENCH_KERNEL_CMSIS_NN TIGRIS_HAS_CMSIS_NN)
elseif(BENCH_KERNEL STREQUAL "s8_ref")
    target_compile_definitions(tigris_pico_bench PRIVATE BENCH_KERNEL_S8)
else()
    message(FATAL_ERROR "BENCH_KERNEL must be cmsis_nn or s8_ref")
endif()
target_compile_definitions(tigris_pico_bench PRIVATE
    TIGRIS_FAST_ARENA_BYTES=${TIGRIS_FAST_ARENA_BYTES}
    TIGRIS_SLOW_ARENA_BYTES=${TIGRIS_SLOW_ARENA_BYTES}
    WARMUP_RUNS=${WARMUP_RUNS}
    BENCH_RUNS=${BENCH_RUNS})
# vendor code (CMSIS-NN): silence its warnings, don't fail our build on them
set_source_files_properties(${NN_SRC} PROPERTIES COMPILE_OPTIONS "-w")

target_link_libraries(tigris_pico_bench pico_stdlib)
pico_enable_stdio_usb(tigris_pico_bench 1)
pico_enable_stdio_uart(tigris_pico_bench 0)
pico_add_extra_outputs(tigris_pico_bench)
