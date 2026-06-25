# Builds the `cmsis-nn` static library from vendored ARM CMSIS-NN and exposes
# the CMSIS-Core + ST device header locations to the rest of the build.
#
# Run third_party/fetch.sh first to populate these directories.

# CMSIS_NN_DIR / CMSIS_CORE_DIR / CMSIS_DEVICE_H7_DIR are set by the parent
# CMakeLists (so the same paths are visible to board.cmake).

if(NOT EXISTS ${CMSIS_NN_DIR}/Include/arm_nnfunctions.h)
    message(FATAL_ERROR
        "CMSIS-NN not vendored at ${CMSIS_NN_DIR}.\n"
        "Run: third_party/fetch.sh")
endif()
if(NOT EXISTS ${CMSIS_DEVICE_H7_DIR}/Include/stm32h753xx.h)
    message(FATAL_ERROR
        "cmsis-device-h7 not vendored at ${CMSIS_DEVICE_H7_DIR}.\n"
        "Run: third_party/fetch.sh")
endif()

file(GLOB_RECURSE CMSIS_NN_SRCS ${CMSIS_NN_DIR}/Source/*.c)
add_library(cmsis-nn STATIC ${CMSIS_NN_SRCS})

# Headers as SYSTEM so the runtime's -Wpedantic -Werror does not trip on them.
target_include_directories(cmsis-nn SYSTEM PUBLIC
    ${CMSIS_NN_DIR}/Include
    ${CMSIS_CORE_DIR}/Include)

# Vendor code: optimize, do not enforce our warning policy. CMSIS-NN's own
# CMakeLists defaults to -Ofast and -O3 is the obvious "go faster" lever, BUT
# both were measured ~2.5% SLOWER than -O2 on the M7 here (on-device @480 MHz:
# -O2 5.539 Mcyc / 11.54 ms vs -O3 5.675 Mcyc / 11.82 ms; -Ofast is byte-
# identical to -O3 - -ffast-math is dead for integer kernels). -O3/-Ofast's
# extra inlining/unrolling bloats .text +7.8% (86.5->93.2 KB) and evicts the hot
# kernels from the 16 KB I-cache; the penalty is in the cycle count itself, so
# it is not a wait-state artifact and holds at both 64 and 480 MHz. -O2 is the
# measured sweet spot for Cortex-M7. Override CMSIS_NN_OPT to re-measure. The
# per-core flags (-mcpu/-mfpu/...) are inherited from the parent's
# add_compile_options, so the M7 DSP paths (__ARM_FEATURE_DSP) auto-select.
set(CMSIS_NN_OPT "-O2" CACHE STRING "optimization level for the CMSIS-NN kernels")
target_compile_options(cmsis-nn PRIVATE ${CMSIS_NN_OPT} -w)
