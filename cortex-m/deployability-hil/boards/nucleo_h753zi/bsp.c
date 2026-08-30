/*
 * Board support for NUCLEO-H753ZI (STM32H753ZI, Cortex-M7).
 *
 * Implements the platform.h interface plus the minimal newlib syscalls so the
 * benchmark's printf reaches the ST-LINK virtual COM port.
 *
 * Bring-up runs the core from HSI at 64 MHz (the reset default): no PLL, no
 * flash-latency or voltage-scaling change needed, which keeps first-light
 * robust. Reaching 480 MHz (VOS0 + ODEN, 4 flash wait states) is done in
 * clock_init(); see the note there. Cycle counts are the primary, clock-
 * independent metric, so a valid measurement does not depend on the clock.
 *
 * UART: USART3 on PD8 (TX) / PD9 (RX), AF7 - the Nucleo-144 ST-LINK VCP wiring.
 * Timing: DWT cycle counter (CYCCNT); the Cortex-M7 needs the DWT LAR unlock.
 */

#include <stdint.h>
#include <stddef.h>
#include <errno.h>
#include <sys/stat.h>

#include "stm32h753xx.h"

#include "platform.h"

#define BENCH_UART_BAUD 115200u

/* Actual clocks after clock_init(); default to the HSI reset values. */
static uint32_t s_sysclk_hz = 64000000u;  /* Cortex-M7 core clock (drives CYCCNT) */
static uint32_t s_apb1_hz   = 64000000u;  /* USART3 kernel clock = PCLK1          */

/* Some CMSIS H7 headers expose the FIFO-aware names only. */
#ifndef USART_ISR_TXE_TXFNF
#define USART_ISR_TXE_TXFNF USART_ISR_TXE
#endif

/* ---- UART (USART3 -> ST-LINK VCP) ---------------------------------------- */

static void uart_init(void)
{
    RCC->AHB4ENR  |= RCC_AHB4ENR_GPIODEN;
    RCC->APB1LENR |= RCC_APB1LENR_USART3EN;
    (void)RCC->APB1LENR;  /* read-back: ensure the clock is on before config */

    /* PD8, PD9 -> alternate-function mode (MODER = 0b10). */
    GPIOD->MODER &= ~((3u << (8 * 2)) | (3u << (9 * 2)));
    GPIOD->MODER |=  ((2u << (8 * 2)) | (2u << (9 * 2)));
    /* High output speed. */
    GPIOD->OSPEEDR |= (3u << (8 * 2)) | (3u << (9 * 2));
    /* AF7 (USART3) on PD8/PD9 - both in the high alternate-function register. */
    GPIOD->AFR[1] &= ~((0xFu << ((8 - 8) * 4)) | (0xFu << ((9 - 8) * 4)));
    GPIOD->AFR[1] |=  ((7u   << ((8 - 8) * 4)) | (7u   << ((9 - 8) * 4)));

    /* USART3 kernel clock = PCLK1 = 64 MHz (reset prescalers are /1).
     * OVER16: BRR = round(fck / baud). */
    USART3->CR1 = 0;
    USART3->BRR = (s_apb1_hz + BENCH_UART_BAUD / 2) / BENCH_UART_BAUD;
    USART3->CR1 = USART_CR1_TE | USART_CR1_UE;
    while (!(USART3->ISR & USART_ISR_TEACK)) { }
}

static void uart_putc(char c)
{
    while (!(USART3->ISR & USART_ISR_TXE_TXFNF)) { }
    USART3->TDR = (uint8_t)c;
}

/* ---- DWT cycle counter --------------------------------------------------- */

#define DWT_LAR_ADDR 0xE0001FB0u   /* DWT lock access register */

static void dwt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    *((volatile uint32_t *)DWT_LAR_ADDR) = 0xC5ACCE55u;  /* M7 requires unlock */
    DWT->CYCCNT = 0;
    DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;
}

/* ---- platform.h ---------------------------------------------------------- */

/* ---- clock: 480 MHz off PLL1, with a safe fallback to HSI 64 MHz --------- */

/* Poll a status flag with a bounded timeout. Returns 1 if set, 0 on timeout. */
static int wait_flag(volatile uint32_t *reg, uint32_t mask)
{
    for (uint32_t i = 0; i < 4000000u; i++)
        if (*reg & mask)
            return 1;
    return 0;
}

/* Bring-up diagnostics, printed once after the UART is up. Snapshots the bail
 * stage and the power/clock registers so a silent HSI fallback is diagnosable
 * without a debugger (mirrors the HardFault register dump). */
static volatile uint32_t s_clk_diag[8];   /* state at bail/success */
#define CLK_DIAG_STAGE 0
#define CLK_DIAG_CR3   1
#define CLK_DIAG_CSR1  2
#define CLK_DIAG_D3CR  3
#define CLK_DIAG_RCCCR 4
#define CLK_DIAG_CFGR  5
#define CLK_DIAG_IDC   6
#define CLK_DIAG_ACR   7

static void clk_snap(volatile uint32_t *d, uint32_t stage)
{
    d[CLK_DIAG_STAGE] = stage;
    d[CLK_DIAG_CR3]   = PWR->CR3;
    d[CLK_DIAG_CSR1]  = PWR->CSR1;
    d[CLK_DIAG_D3CR]  = PWR->D3CR;
    d[CLK_DIAG_RCCCR] = RCC->CR;
    d[CLK_DIAG_CFGR]  = RCC->CFGR;
    d[CLK_DIAG_IDC]   = DBGMCU->IDCODE;   /* rev_id in [31:16] */
    d[CLK_DIAG_ACR]   = FLASH->ACR;
}

const volatile uint32_t *platform_clock_diag(void) { return s_clk_diag; }

/* HSI(64) / DIVM1=8 -> 8 MHz ref, xN=120 -> 960 MHz VCO, /P=2 -> 480 MHz.
 * Needs VOS0 (VOS1 + SYSCFG overdrive). Any timeout leaves the core on the HSI
 * 64 MHz reset clock, so a bad config degrades rather than bricking the board.
 * Bus tree: HCLK = SYSCLK/2 = 240 MHz (AXI max), APB = HCLK/2 = 120 MHz (USART3
 * kernel clock). Flash wait states are set from the FLASH interface clock
 * (= HCLK = 240 MHz, NOT the 480 MHz core), which is 4 WS at VOS0 per RM0433
 * Table 17 - this matches what CubeMX generates for a 480 MHz/VOS0 config. */
static void clock_init(void)
{
    RCC->APB4ENR |= RCC_APB4ENR_SYSCFGEN;     /* SYSCFG clock for the ODEN bit */
    (void)RCC->APB4ENR;

    /* H7 voltage scaling only completes once the power-supply configuration in
     * PWR->CR3 has been committed (written once after reset) and the regulator
     * reports ready via ACTVOSRDY. Re-write CR3's current (board-correct, reset)
     * supply selection to commit it, then wait. Skipping this leaves VOSRDY
     * permanently low and silently drops the core to the 64 MHz HSI fallback. */
    PWR->CR3 = PWR->CR3;                       /* commit reset/board supply config */
    if (!wait_flag(&PWR->CSR1, PWR_CSR1_ACTVOSRDY)) {
        clk_snap(s_clk_diag, 11);             /* supply never reached ready */
        return;
    }

    PWR->D3CR |= PWR_D3CR_VOS;                 /* 0b11 = VOS1 */
    if (!wait_flag(&PWR->D3CR, PWR_D3CR_VOSRDY)) {
        clk_snap(s_clk_diag, 10);             /* VOSRDY#1 (VOS1) timeout */
        return;
    }
    SYSCFG->PWRCR |= SYSCFG_PWRCR_ODEN;        /* overdrive -> VOS0 */
    if (!wait_flag(&PWR->D3CR, PWR_D3CR_VOSRDY)) {
        clk_snap(s_clk_diag, 20);                     /* VOSRDY#2 (VOS0/ODEN) timeout */
        return;
    }

    FLASH->ACR = (FLASH->ACR & ~FLASH_ACR_LATENCY) | FLASH_ACR_LATENCY_4WS;

    RCC->PLLCKSELR = (RCC->PLLCKSELR & ~(RCC_PLLCKSELR_DIVM1 | RCC_PLLCKSELR_PLLSRC))
                   | (8u << RCC_PLLCKSELR_DIVM1_Pos);     /* PLLSRC=00 (HSI) */
    RCC->PLL1DIVR = ((120u - 1u) << RCC_PLL1DIVR_N1_Pos)
                  | ((  2u - 1u) << RCC_PLL1DIVR_P1_Pos)
                  | ((  2u - 1u) << RCC_PLL1DIVR_Q1_Pos)
                  | ((  2u - 1u) << RCC_PLL1DIVR_R1_Pos);
    RCC->PLLCFGR = (RCC->PLLCFGR & ~RCC_PLLCFGR_PLL1RGE)
                 | (0x3u << RCC_PLLCFGR_PLL1RGE_Pos)       /* ref 8-16 MHz */
                 | RCC_PLLCFGR_DIVP1EN;                    /* P output; VCOSEL=0 (wide) */

    RCC->CR |= RCC_CR_PLL1ON;
    if (!wait_flag(&RCC->CR, RCC_CR_PLL1RDY)) {
        clk_snap(s_clk_diag, 30);                     /* PLL1 never locked -> stay HSI */
        return;
    }

    /* Prescalers before the switch so HCLK never exceeds 240 MHz. */
    RCC->D1CFGR = (RCC->D1CFGR & ~(RCC_D1CFGR_HPRE | RCC_D1CFGR_D1PPRE | RCC_D1CFGR_D1CPRE))
                | RCC_D1CFGR_HPRE_DIV2 | RCC_D1CFGR_D1PPRE_DIV2;   /* D1CPRE=/1 */
    RCC->D2CFGR = (RCC->D2CFGR & ~(RCC_D2CFGR_D2PPRE1 | RCC_D2CFGR_D2PPRE2))
                | RCC_D2CFGR_D2PPRE1_DIV2 | RCC_D2CFGR_D2PPRE2_DIV2;
    RCC->D3CFGR = (RCC->D3CFGR & ~RCC_D3CFGR_D3PPRE) | RCC_D3CFGR_D3PPRE_DIV2;

    RCC->CFGR = (RCC->CFGR & ~RCC_CFGR_SW) | RCC_CFGR_SW_PLL1;
    for (uint32_t i = 0; i < 4000000u; i++) {
        if ((RCC->CFGR & RCC_CFGR_SWS) == RCC_CFGR_SWS_PLL1) {
            s_sysclk_hz = 480000000u;
            s_apb1_hz   = 120000000u;
            clk_snap(s_clk_diag, 5);                  /* success: running on PLL1 @480 */
            return;
        }
    }
    clk_snap(s_clk_diag, 40);                         /* PLL1 locked but SWS never switched */
}

void platform_init(void)
{
    SCB_EnableICache();
    SCB_EnableDCache();
    clock_init();   /* 480 MHz off PLL1, or HSI 64 MHz fallback */
    uart_init();
    dwt_init();
}

uint32_t platform_cpu_hz(void)         { return s_sysclk_hz; }
uint32_t platform_cycles(void)         { return DWT->CYCCNT; }
const char *platform_board_name(void)  { return "nucleo_h753zi"; }

void platform_halt(void)
{
    __disable_irq();
    for (;;) { }
}

/* Announce a fault over the UART instead of silently spinning in the startup
 * Default_Handler - invaluable during bring-up. Prints the configurable fault
 * status (CFSR: bit 24 = UNALIGNED usage fault) and BusFault address (BFAR). */
static void uart_puts(const char *s)
{
    for (; *s; s++)
        uart_putc(*s);
}

static void uart_puthex(uint32_t v)
{
    static const char hx[] = "0123456789abcdef";
    uart_puts("0x");
    for (int i = 28; i >= 0; i -= 4)
        uart_putc(hx[(v >> i) & 0xF]);
}

void hardfault_report(uint32_t *frame)
{
    uart_puts("\n!!HARDFAULT!! CFSR=");
    uart_puthex(SCB->CFSR);
    uart_puts(" PC=");
    uart_puthex(frame[6]);   /* stacked return address (faulting instruction) */
    uart_puts(" LR=");
    uart_puthex(frame[5]);
    uart_putc('\n');
    for (;;) { }
}

__attribute__((naked)) void HardFault_Handler(void)
{
    __asm volatile(
        "tst lr, #4         \n"
        "ite eq             \n"
        "mrseq r0, msp      \n"
        "mrsne r0, psp      \n"
        "b hardfault_report \n");
}

/* ---- minimal newlib syscalls --------------------------------------------- */

int _write(int file, char *ptr, int len)
{
    (void)file;
    for (int i = 0; i < len; i++)
        uart_putc(ptr[i]);
    return len;
}

extern char end;        /* start of heap (linker)  */
extern char _heap_end;  /* end of heap region (linker) */

void *_sbrk(int incr)
{
    static char *heap = NULL;
    if (heap == NULL)
        heap = &end;
    if (heap + incr > &_heap_end) {
        errno = ENOMEM;
        return (void *)-1;
    }
    char *prev = heap;
    heap += incr;
    return prev;
}

int _read(int file, char *ptr, int len)   { (void)file; (void)ptr; (void)len; return 0; }
int _close(int file)                       { (void)file; return -1; }
int _lseek(int file, int off, int dir)     { (void)file; (void)off; (void)dir; return 0; }
int _fstat(int file, struct stat *st)      { (void)file; st->st_mode = S_IFCHR; return 0; }
int _isatty(int file)                      { (void)file; return 1; }
int _getpid(void)                          { return 1; }
int _kill(int pid, int sig)                { (void)pid; (void)sig; errno = EINVAL; return -1; }
void _exit(int code)                       { (void)code; __disable_irq(); for (;;) { } }
