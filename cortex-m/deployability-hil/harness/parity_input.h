/*
 * Seeded parity inputs, shared by the TiGrIS and TFLM harnesses so both fill
 * the model input with the same bytes. After the timed runs, each harness runs
 * BENCH_PARITY_VECTORS more inferences on consecutive xorshift32 bytes from
 * BENCH_PARITY_SEED and prints each output as an OUTPUT_I8_SEEDED line. The
 * timed runs and the reported RAM still use the constant input.
 */
#ifndef CORTEXM_BENCH_PARITY_INPUT_H
#define CORTEXM_BENCH_PARITY_INPUT_H

#include <stdint.h>

#ifndef BENCH_PARITY_VECTORS
#define BENCH_PARITY_VECTORS 8
#endif
#define BENCH_PARITY_SEED 0x2545F491u

/* Write the next size_bytes bytes of the stream whose state is *state. */
static inline void parity_fill(uint8_t *data, uint32_t size_bytes,
                               uint32_t *state)
{
    uint32_t s = *state;
    for (uint32_t i = 0; i < size_bytes; i++) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        data[i] = (uint8_t)(s >> 24);
    }
    *state = s;
}

#endif /* CORTEXM_BENCH_PARITY_INPUT_H */
