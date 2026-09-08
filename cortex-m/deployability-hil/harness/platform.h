/*
 * Board contract for the Cortex-M deployability harness.
 *
 * The board bring-up (clock, a UART printf retargets to, a cycle counter, the
 * linker and syscalls) now comes from the tigris-cortex-m target-support
 * package, whose HAL is tigris_hal.h. This shim maps the harness's historic
 * platform_* calls onto the package's tigris_hal_* implementation, so main.c
 * and the TFLM baseline are unchanged while the board code is single-sourced in
 * the package (and thereby dogfooded by this bench).
 */
#ifndef CORTEXM_BENCH_PLATFORM_H
#define CORTEXM_BENCH_PLATFORM_H

#include "tigris_hal.h"

#define platform_init        tigris_hal_init
#define platform_cpu_hz      tigris_hal_cpu_hz
#define platform_cycles      tigris_hal_cycles
#define platform_board_name  tigris_hal_board_name
#define platform_clock_diag  tigris_hal_clock_diag
#define platform_halt        tigris_hal_halt

#endif /* CORTEXM_BENCH_PLATFORM_H */
