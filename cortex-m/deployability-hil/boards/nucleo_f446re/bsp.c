/*
 * Board support for NUCLEO-F446RE (STM32F446RE, Cortex-M4F).
 *
 * Implements the platform.h interface plus the minimal newlib syscalls so the
 * benchmark's printf reaches the ST-LINK virtual COM port.
 *
 * Bring-up runs the core from HSI at 16 MHz (the reset default). clock_init()
 * raises it to the rated 180 MHz off PLL (HSI/8 -> 2 MHz, xN=180 -> 360 MHz VCO,
 * /P=2 -> 180 MHz), which needs VOS scale 1 + the PWR over-drive and 5 flash
 * wait states. Any timeout leaves the core on the 16 MHz HSI reset clock, so a
 * bad config degrades rather than bricking. Cycle counts are the primary,
 * clock-independent metric, so a valid measurement does not depend on the clock.
 *
 * UART: USART2 on PA2 (TX) / PA3 (RX), AF7 - the Nucleo-64 ST-LINK VCP wiring.
 * Timing: DWT cycle counter (CYCCNT); M4 does not need the M7 LAR unlock.
 */

#include <stdint.h>
#include <stddef.h>
#include <errno.h>
#include <sys/stat.h>

#include "stm32f446xx.h"

#include "platform.h"

#define BENCH_UART_BAUD 115200u

/* Actual clocks after clock_init(); default to the HSI reset values. */
static uint32_t s_sysclk_hz = 16000000u;  /* Cortex-M4 core clock (drives CYCCNT) */
static uint32_t s_apb1_hz   = 16000000u;  /* USART2 kernel clock = PCLK1          */

/* ---- UART (USART2 -> ST-LINK VCP) ---------------------------------------- */

static void uart_init(void)
{
    RCC->AHB1ENR  |= RCC_AHB1ENR_GPIOAEN;
    RCC->APB1ENR  |= RCC_APB1ENR_USART2EN;
    (void)RCC->APB1ENR;  /* read-back: ensure the clock is on before config */

    /* PA2, PA3 -> alternate-function mode (MODER = 0b10). */
    GPIOA->MODER &= ~((3u << (2 * 2)) | (3u << (3 * 2)));
    GPIOA->MODER |=  ((2u << (2 * 2)) | (2u << (3 * 2)));
    /* High output speed. */
    GPIOA->OSPEEDR |= (3u << (2 * 2)) | (3u << (3 * 2));
    /* AF7 (USART2) on PA2/PA3 - both in the low alternate-function register. */
    GPIOA->AFR[0] &= ~((0xFu << (2 * 4)) | (0xFu << (3 * 4)));
    GPIOA->AFR[0] |=  ((7u   << (2 * 4)) | (7u   << (3 * 4)));

    /* OVER16: BRR = round(fck / baud) in 12.4 fixed point (== fck/baud). */
    USART2->CR1 = 0;
    USART2->BRR = (s_apb1_hz + BENCH_UART_BAUD / 2) / BENCH_UART_BAUD;
    USART2->CR1 = USART_CR1_TE | USART_CR1_UE;
}

static void uart_putc(char c)
{
    while (!(USART2->SR & USART_SR_TXE)) { }
    USART2->DR = (uint8_t)c;
}

/* ---- DWT cycle counter --------------------------------------------------- */

static void dwt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;
}

/* ---- bring-up diagnostics (mirrors the H753 CLOCK_DIAG) ------------------ */

static volatile uint32_t s_clk_diag[8];
static void clk_snap(uint32_t stage)
{
    s_clk_diag[0] = stage;
    s_clk_diag[1] = PWR->CR;
    s_clk_diag[2] = PWR->CSR;
    s_clk_diag[3] = RCC->CR;
    s_clk_diag[4] = RCC->CFGR;
    s_clk_diag[5] = RCC->PLLCFGR;
    s_clk_diag[6] = DBGMCU->IDCODE;
    s_clk_diag[7] = FLASH->ACR;
}
const volatile uint32_t *platform_clock_diag(void) { return s_clk_diag; }

/* Poll a status flag with a bounded timeout. Returns 1 if set, 0 on timeout. */
static int wait_flag(volatile uint32_t *reg, uint32_t mask)
{
    for (uint32_t i = 0; i < 2000000u; i++)
        if (*reg & mask)
            return 1;
    return 0;
}

/* ---- clock: 180 MHz off PLL, with a safe fallback to HSI 16 MHz ---------- */

static void clock_init(void)
{
    /* PWR clock on; select VOS scale 1 (required for 180 MHz). */
    RCC->APB1ENR |= RCC_APB1ENR_PWREN;
    (void)RCC->APB1ENR;
    PWR->CR |= PWR_CR_VOS;                 /* 0b11 = scale 1 */

    /* HSI is already the reset clock; make sure it is ready. */
    RCC->CR |= RCC_CR_HSION;
    if (!wait_flag(&RCC->CR, RCC_CR_HSIRDY)) { clk_snap(10); return; }

    /* PLL: HSI(16) / M=8 -> 2 MHz, xN=180 -> 360 MHz VCO, /P=2 -> 180 MHz.
     * Q=8 (unused; USB not needed). PLLSRC = HSI. */
    RCC->PLLCFGR = (8u)                              /* PLLM   */
                 | (180u << RCC_PLLCFGR_PLLN_Pos)    /* PLLN   */
                 | (0u   << RCC_PLLCFGR_PLLP_Pos)    /* PLLP=/2 */
                 | (0u   << RCC_PLLCFGR_PLLSRC_Pos)  /* HSI    */
                 | (8u   << RCC_PLLCFGR_PLLQ_Pos);   /* PLLQ   */

    RCC->CR |= RCC_CR_PLLON;
    if (!wait_flag(&RCC->CR, RCC_CR_PLLRDY)) { clk_snap(30); return; }

    /* Over-drive to reach 180 MHz. */
    PWR->CR |= PWR_CR_ODEN;
    if (!wait_flag(&PWR->CSR, PWR_CSR_ODRDY))   { clk_snap(20); return; }
    PWR->CR |= PWR_CR_ODSWEN;
    if (!wait_flag(&PWR->CSR, PWR_CSR_ODSWRDY)) { clk_snap(21); return; }

    /* Flash: 5 wait states + prefetch + ART (I/D cache) for 180 MHz @ VOS1. */
    FLASH->ACR = FLASH_ACR_LATENCY_5WS | FLASH_ACR_PRFTEN |
                 FLASH_ACR_ICEN | FLASH_ACR_DCEN;

    /* Bus prescalers: AHB/1 = 180, APB1/4 = 45 (max 45), APB2/2 = 90 (max 90). */
    RCC->CFGR = (RCC->CFGR & ~(RCC_CFGR_HPRE | RCC_CFGR_PPRE1 | RCC_CFGR_PPRE2))
              | RCC_CFGR_HPRE_DIV1 | RCC_CFGR_PPRE1_DIV4 | RCC_CFGR_PPRE2_DIV2;

    /* Switch SYSCLK to the PLL. */
    RCC->CFGR = (RCC->CFGR & ~RCC_CFGR_SW) | RCC_CFGR_SW_PLL;
    for (uint32_t i = 0; i < 2000000u; i++) {
        if ((RCC->CFGR & RCC_CFGR_SWS) == RCC_CFGR_SWS_PLL) {
            s_sysclk_hz = 180000000u;
            s_apb1_hz   = 45000000u;     /* USART2 kernel clock = PCLK1 */
            clk_snap(5);
            return;
        }
    }
    clk_snap(40);
}

/* ---- platform.h ---------------------------------------------------------- */

void platform_init(void)
{
    clock_init();   /* 180 MHz off PLL, or HSI 16 MHz fallback */
    uart_init();
    dwt_init();
}

uint32_t platform_cpu_hz(void)         { return s_sysclk_hz; }
uint32_t platform_cycles(void)         { return DWT->CYCCNT; }
const char *platform_board_name(void)  { return "nucleo_f446re"; }

void platform_halt(void)
{
    __disable_irq();
    for (;;) { }
}

/* ---- fault reporting + newlib syscalls (generic) ------------------------- */

static void uart_puts(const char *s) { for (; *s; s++) uart_putc(*s); }
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
    uart_puthex(frame[6]);
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

int _write(int file, char *ptr, int len)
{
    (void)file;
    for (int i = 0; i < len; i++)
        uart_putc(ptr[i]);
    return len;
}

extern char end;
extern char _heap_end;

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
