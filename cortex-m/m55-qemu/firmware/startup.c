/* Minimal Cortex-M55 startup for QEMU mps3-an547. Vector table at 0x0 (ITCM). */
#include <stdint.h>

extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _estack;
extern int main(void);

void Reset_Handler(void)
{
    /* copy .data (LMA in ITCM -> VMA in SRAM) */
    uint32_t *src = &_sidata, *dst = &_sdata;
    while (dst < &_edata) *dst++ = *src++;
    /* zero .bss */
    for (dst = &_sbss; dst < &_ebss; ) *dst++ = 0u;
    /* enable CP10/CP11 (FPU full access) via CPACR */
    *(volatile uint32_t *)0xE000ED88u |= (0xFu << 20);
    __asm__ volatile("dsb");
    __asm__ volatile("isb");
    (void)main();
    for (;;) { }
}

void Default_Handler(void) { for (;;) { } }

/* A handful of vectors is enough for a QEMU bare-metal run (SP + Reset +
 * fault handlers). The rest default. */
__attribute__((section(".isr_vector"), used))
void (*const g_pfnVectors[])(void) = {
    (void (*)(void))&_estack,   /* 0: initial SP            */
    Reset_Handler,              /* 1: reset                 */
    Default_Handler,            /* 2: NMI                   */
    Default_Handler,            /* 3: HardFault             */
    Default_Handler,            /* 4: MemManage             */
    Default_Handler,            /* 5: BusFault              */
    Default_Handler,            /* 6: UsageFault            */
    Default_Handler,            /* 7: SecureFault           */
    0, 0, 0,                    /* 8-10 reserved            */
    Default_Handler,            /* 11: SVCall               */
    Default_Handler,            /* 12: DebugMon             */
    0,                          /* 13 reserved              */
    Default_Handler,            /* 14: PendSV               */
    Default_Handler,            /* 15: SysTick              */
};
