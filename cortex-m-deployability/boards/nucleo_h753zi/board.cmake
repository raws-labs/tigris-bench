# Board config for NUCLEO-H753ZI (STM32H753ZI, Cortex-M7, double-precision FPU).
#
# Included by the top-level CMakeLists after add_subdirectory(third_party), so
# CMSIS_CORE_DIR / CMSIS_DEVICE_H7_DIR (set there) are visible.

set(BOARD_MCU_FLAGS
    -mcpu=cortex-m7
    -mfpu=fpv5-d16
    -mfloat-abi=hard
    -mthumb)

set(BOARD_DEFINES STM32H753xx)

set(BOARD_STARTUP       ${CMSIS_DEVICE_H7_DIR}/Source/Templates/gcc/startup_stm32h753xx.s)
set(BOARD_SYSTEM        ${CMSIS_DEVICE_H7_DIR}/Source/Templates/system_stm32h7xx.c)
set(BOARD_BSP           ${CMAKE_CURRENT_LIST_DIR}/bsp.c)
set(BOARD_LINKER_SCRIPT ${CMAKE_CURRENT_LIST_DIR}/STM32H753ZITx_FLASH.ld)

set(BOARD_INCLUDES
    ${CMSIS_CORE_DIR}/Include
    ${CMSIS_DEVICE_H7_DIR}/Include)

# TFLM prebuilt-lib arch (gen/cortex_m_generic_<arch>_default_cmsis_nn_gcc).
set(BOARD_TFLM_ARCH "cortex-m7+fp")
