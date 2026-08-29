/* Linker script for NUCLEO-F446RE (STM32F446RE, Cortex-M4F).
 *
 * Layout for the TiGrIS benchmark:
 *   - code + rodata + embedded plan blob -> FLASH (512 KB @ 0x08000000)
 *   - .data / .bss / TiGrIS arenas / heap -> SRAM (128 KB @ 0x20000000)
 *   - stack                               -> top of SRAM (grows down)
 *
 * The F446 has a single contiguous 128 KB SRAM (no TCM/AXI split), so the arenas,
 * heap and stack all share it - the benchmark sizes the arenas to fit. The big
 * arenas (s_fast_arena / s_slow_arena) are .bss, so `arm-none-eabi-size` reports
 * them as part of the RAM working set.
 */

ENTRY(Reset_Handler)

_estack = ORIGIN(RAM) + LENGTH(RAM);   /* top of SRAM */

_Min_Heap_Size  = 0x2000;   /* 8 KB  - newlib printf internals only */
_Min_Stack_Size = 0x4000;   /* 16 KB */

MEMORY
{
  FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 512K
  RAM   (xrw) : ORIGIN = 0x20000000, LENGTH = 128K
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
  } >RAM AT> FLASH

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
  } >RAM

  ._user_heap :
  {
    . = ALIGN(8);
    PROVIDE ( end = . );
    PROVIDE ( _end = . );
    . = . + _Min_Heap_Size;
    . = ALIGN(8);
  } >RAM

  /* Heap (used by _sbrk) may grow up to the stack reservation at the top. */
  _heap_end = ORIGIN(RAM) + LENGTH(RAM) - _Min_Stack_Size;

  /* Make sure .data + .bss + min heap + min stack fit in the 128 KB SRAM. */
  ASSERT(_ebss + _Min_Heap_Size + _Min_Stack_Size <= ORIGIN(RAM) + LENGTH(RAM),
         "F446 SRAM overflow: shrink the TiGrIS arenas")

  .ARM.attributes 0 : { *(.ARM.attributes) }
}
