/* Cortex-M55 / QEMU mps3-an547 (Corstone-300) memory map.
 *   ITCM  0x00000000 512K  - vector table + code + rodata (+ .data LMA)
 *   DTCM  0x20000000 512K  - stack
 *   SRAM  0x21000000 4M    - .data / .bss / TiGrIS arenas (the on-chip SRAM
 *                            working set: fast tile arena + slow tier)
 * The 24 MB plan lives at DDR 0x60000000 (loaded by QEMU -device loader),
 * modelling external XIP flash; it is not part of this image.
 */
ENTRY(Reset_Handler)

_estack = 0x20000000 + 512K;     /* top of DTCM */

MEMORY
{
  ITCM (rx)  : ORIGIN = 0x00000000, LENGTH = 512K
  DTCM (xrw) : ORIGIN = 0x20000000, LENGTH = 512K
  SRAM (xrw) : ORIGIN = 0x21000000, LENGTH = 4096K
}

SECTIONS
{
  .isr_vector : { . = ALIGN(4); KEEP(*(.isr_vector)) . = ALIGN(4); } > ITCM

  .text :
  {
    . = ALIGN(4);
    *(.text) *(.text*)
    *(.rodata) *(.rodata*)
    *(.glue_7) *(.glue_7t)
    KEEP(*(.init)) KEEP(*(.fini))
    . = ALIGN(4);
    _etext = .;
  } > ITCM

  .ARM.extab : { *(.ARM.extab* .gnu.linkonce.armextab.*) } > ITCM
  .ARM       : { __exidx_start = .; *(.ARM.exidx*) __exidx_end = .; } > ITCM

  _sidata = LOADADDR(.data);
  .data :
  {
    . = ALIGN(4);
    _sdata = .;
    *(.data) *(.data*)
    . = ALIGN(4);
    _edata = .;
  } > SRAM AT> ITCM

  .bss :
  {
    . = ALIGN(4);
    _sbss = .;
    *(.bss) *(.bss*) *(COMMON)
    . = ALIGN(16);
    _ebss = .;
  } > SRAM

  /DISCARD/ : { *(.ARM.attributes) }
}
