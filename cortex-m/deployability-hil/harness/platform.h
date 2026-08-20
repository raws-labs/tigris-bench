/*
 * Board abstraction for the Cortex-M deployability benchmark.
 *
 * The benchmark in main.c is board-independent: it talks to the hardware only
 * through this interface. Each board under boards/<board>/ implements it
 * (clock setup, a UART that stdio printf is retargeted to, and the cycle
 * counter used for timing).
 */
#ifndef CORTEXM_BENCH_PLATFORM_H
#define CORTEXM_BENCH_PLATFORM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bring up clocks, the result UART (printf is retargeted to it), the cycle
 * counter, and any caches. Call once at the very top of main(). */
void platform_init(void);

/* Core clock in Hz - used to convert cycle counts to wall-clock time. */
uint32_t platform_cpu_hz(void);

/* Free-running monotonic cycle counter (DWT->CYCCNT on Cortex-M7/M4/M33).
 * 32-bit; wraps after 2^32 cycles, which is far longer than one inference. */
uint32_t platform_cycles(void);

/* Short human-readable board name for the results line (e.g. "nucleo_h753zi"). */
const char *platform_board_name(void);

/* Optional bring-up diagnostics: a board may expose an 8-word snapshot of the
 * clock/power state captured at the end of clock setup (NULL if unsupported).
 * Lets the benchmark confirm the target clock was reached, and diagnose a
 * silent clock fallback, without a debugger. */
const volatile uint32_t *platform_clock_diag(void);

/* Disable interrupts and spin forever. Called once the benchmark is done so
 * the final serial output is not disturbed by a reset loop. Never returns. */
void platform_halt(void);

#ifdef __cplusplus
}
#endif

#endif /* CORTEXM_BENCH_PLATFORM_H */
