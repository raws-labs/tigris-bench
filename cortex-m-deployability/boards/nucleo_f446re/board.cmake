# Board config for NUCLEO-F446RE (STM32F446RE, Cortex-M4F, single-precision FPU).
#
# Included by the top-level CMakeLists after the CMSIS dirs are set, so
# CMSIS_CORE_DIR / CMSIS_DEVICE_F4_DIR are visible.

set(BOARD_MCU_FLAGS
    -mcpu=cortex-m4
    -mfpu=fpv4-sp-d16
    -mfloat-abi=hard
    -mthumb)

set(BOARD_DEFINES STM32F446xx)

set(BOARD_STARTUP       ${CMSIS_DEVICE_F4_DIR}/Source/Templates/gcc/startup_stm32f446xx.s)
set(BOARD_SYSTEM        ${CMSIS_DEVICE_F4_DIR}/Source/Templates/system_stm32f4xx.c)
set(BOARD_BSP           ${CMAKE_CURRENT_LIST_DIR}/bsp.c)
set(BOARD_LINKER_SCRIPT ${CMAKE_CURRENT_LIST_DIR}/STM32F446RETx_FLASH.ld)

set(BOARD_INCLUDES
    ${CMSIS_CORE_DIR}/Include
    ${CMSIS_DEVICE_F4_DIR}/Include)

# TFLM prebuilt-lib arch (gen/cortex_m_generic_<arch>_default_cmsis_nn_gcc).
set(BOARD_TFLM_ARCH "cortex-m4+fp")
