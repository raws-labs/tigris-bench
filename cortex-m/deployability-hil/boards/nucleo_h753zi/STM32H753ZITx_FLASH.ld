/* Linker script for NUCLEO-H753ZI (STM32H753ZI, Cortex-M7).
 *
 * Layout for the TiGrIS benchmark:
 *   - code + rodata + embedded plan blob -> FLASH (2 MB @ 0x08000000)
 *   - stack                              -> DTCM (fast, dedicated; top down)
 *   - .data / .bss / TiGrIS arenas / heap-> AXI SRAM D1 (512 KB @ 0x24000000)
 *
 * The big arenas (s_fast_arena / s_slow_arena) are .bss, so they land in AXI
 * SRAM; `arm-none-eabi-size` then reports them as part of the RAM working set.
 */

ENTRY(Reset_Handler)

_estack = ORIGIN(DTCM) + LENGTH(DTCM);   /* top of DTCM */

_Min_Heap_Size  = 0x2000;   /* 8 KB  - newlib printf internals only */
_Min_Stack_Size = 0x4000;   /* 16 KB */

MEMORY
{
  ITCM    (xrw) : ORIGIN = 0x00000000, LENGTH = 64K
  FLASH   (rx)  : ORIGIN = 0x08000000, LENGTH = 2048K
  DTCM    (xrw) : ORIGIN = 0x20000000, LENGTH = 128K
  AXISRAM (xrw) : ORIGIN = 0x24000000, LENGTH = 512K
  RAM_D2  (xrw) : ORIGIN = 0x30000000, LENGTH = 288K
  RAM_D3  (xrw) : ORIGIN = 0x38000000, LENGTH = 64K
}

SECTIONS
{
  .isr_vector :
  {
    . = ALIGN(4);
    KEEP(*(.isr_vector))
    . = ALIGN(4);
  } >FLASH

  .text :
  {
    . = ALIGN(4);
    *(.text)
    *(.text*)
    *(.glue_7)
    *(.glue_7t)
    *(.eh_frame)
    KEEP(*(.init))
    KEEP(*(.fini))
    . = ALIGN(4);
    _etext = .;
  } >FLASH

  .rodata :
  {
    . = ALIGN(4);
    *(.rodata)
    *(.rodata*)
    . = ALIGN(4);
  } >FLASH

  .ARM.extab : { *(.ARM.extab* .gnu.linkonce.armextab.*) } >FLASH
  .ARM :
  {
    __exidx_start = .;
    *(.ARM.exidx*)
    __exidx_end = .;
  } >FLASH

  .preinit_array :
  {
    PROVIDE_HIDDEN(__preinit_array_start = .);
    KEEP(*(.preinit_array*))
    PROVIDE_HIDDEN(__preinit_array_end = .);
  } >FLASH
  .init_array :
  {
    PROVIDE_HIDDEN(__init_array_start = .);
    KEEP(*(SORT(.init_array.*)))
    KEEP(*(.init_array*))
    PROVIDE_HIDDEN(__init_array_end = .);
  } >FLASH
  .fini_array :
  {
    PROVIDE_HIDDEN(__fini_array_start = .);
    KEEP(*(SORT(.fini_array.*)))
    KEEP(*(.fini_array*))
    PROVIDE_HIDDEN(__fini_array_end = .);
  } >FLASH

  _sidata = LOADADDR(.data);
  .data :
  {
    . = ALIGN(4);
    _sdata = .;
    *(.data)
    *(.data*)
    . = ALIGN(4);
    _edata = .;
  } >AXISRAM AT> FLASH

  .bss :
  {
    . = ALIGN(4);
    _sbss = .;
    __bss_start__ = _sbss;
    *(.bss)
    *(.bss*)
    *(COMMON)
    . = ALIGN(4);
    _ebss = .;
    __bss_end__ = _ebss;
  } >AXISRAM

  ._user_heap :
  {
    . = ALIGN(8);
    PROVIDE ( end = . );
    PROVIDE ( _end = . );
    . = . + _Min_Heap_Size;
    . = ALIGN(8);
  } >AXISRAM

  /* Heap may grow to the end of AXI SRAM (used by _sbrk). */
  _heap_end = ORIGIN(AXISRAM) + LENGTH(AXISRAM);

  /* Stack lives in DTCM; make sure the minimum fits. */
  ASSERT(_Min_Stack_Size <= LENGTH(DTCM), "DTCM too small for the stack")

  .ARM.attributes 0 : { *(.ARM.attributes) }
}
