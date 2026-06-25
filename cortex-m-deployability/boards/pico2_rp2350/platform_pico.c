/*
 * platform.h implementation for the Raspberry Pi Pico 2 (RP2350, Cortex-M33)
 * on top of the pico-sdk. Unlike the STM32 boards (hand-written CMSIS BSPs),
 * the RP2350 has no on-board debugger or VCP, so the SDK provides the clock
 * setup, the flash/XIP boot, and stdio over USB-CDC (printf is retargeted to it).
 *
 * Timing uses the RP2350 hardware microsecond timer (time_us), NOT the Armv8-M
 * DWT cycle counter: the DWT CYCCNT on RP2350 under-counts (it disagreed with
 * the wall clock by ~6x for memory-heavy code while matching for compute-heavy
 * code - it appears not to count XIP-stall cycles). time_us is the trustworthy
 * wall clock; we report clock-equivalent "cycles" = us * MHz so the harness's
 * cycles/cpu_hz gives the real ms. The benchmark runs once at boot, so
 * platform_init waits (bounded) for the USB-CDC host before running.
 */

#include <stdint.h>

#include "platform.h"

#include "pico/stdlib.h"
#include "hardware/clocks.h"
#include "pico/time.h"

static uint32_t s_mhz = 150;
static volatile uint32_t s_clk_diag[8];
const volatile uint32_t *platform_clock_diag(void) { return s_clk_diag; }

void platform_init(void)
{
    /* RP2350 default sys clock is 150 MHz; set it explicitly for clarity. */
    set_sys_clock_khz(150000, true);
    s_mhz = (uint32_t)(clock_get_hz(clk_sys) / 1000000u);

    stdio_init_all();   /* USB-CDC; printf retargets here */
    /* The benchmark prints once and halts, so wait (bounded ~6 s) for the host
     * to open the CDC port before we run, otherwise the output is lost. */
    for (int i = 0; i < 600 && !stdio_usb_connected(); i++)
        sleep_ms(10);
    sleep_ms(150);

    s_clk_diag[0] = 5;            /* stage 5 = clock up (matches STM32 CLOCK_DIAG) */
    s_clk_diag[1] = (uint32_t)clock_get_hz(clk_sys);
}

uint32_t platform_cpu_hz(void)        { return (uint32_t)clock_get_hz(clk_sys); }
/* Wall-clock cycles = elapsed microseconds * MHz (see header note on the DWT). */
uint32_t platform_cycles(void)        { return (uint32_t)(time_us_64() * (uint64_t)s_mhz); }
const char *platform_board_name(void) { return "pico2_rp2350"; }

void platform_halt(void)
{
    for (;;)
        tight_loop_contents();
}
