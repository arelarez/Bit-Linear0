## Abstraction 1.58 bit (BitLinear)

1.58 bit is an architecture with a pivoting structure between the values ```(-1, 0, 1)``` which requires only a logarithmic memory space of ```$\log_2(3) ≈ 1.58 bit\$``` The crucial addition of the number "0" in the Grover matrix acts as a filter or *connection breaker* (*sparsity*) which allows the model to be adaptive in scaling by filtering the window context in the facet process to reach the variant (probability) empathetically; remove important noise and unimportant noise.

#### BitLinear Layer

The BitNet calculation layer uses a special layer called **BitLinear** which considers its weights to be only -1, 0 or 1, heavy multiplication operations that are usually calculated with high-precision architectures such as (```FP32/FP16```) or standard computation layers completely removed and replaced by much lighter **linear addition and subtraction** operations

## The Prospection & Advantages for AI

Bit 1.58 is not conceptually similar to the scaling phase and the usual model compression training techniques *such as forced post-training quantization* This model is **natively trained** in the initial stage with the constraints of those 3 values

#### Extreme Memory Shrinkage as encouragement for efficiency

Models with 2 billion parameters (such as Microsoft's ```BitNet b1.58 2B4T```) went from Gigabyte to only taking up about **400 MB - 700 MB** of memory space.with the efficiency of power consumption during the *Inference* process (producing answers) this is drastically compressed up to **55-70%** (or can be 12× more efficient) due to the elimination of matrix multiplication operations.

#### Breaking GPU Tendency

The addition operation is essentially designed to be very friendly to CPU width compromises,This architecture ushers in the era of Edge AI, enabling models as large as LLM and sophisticated low-level models to now be run locally on hardware architectures (HBM) such as the ARM series.such as smartphones, standard laptops, and even Raspberry Pi IoT devices without the need for a massive graphics card (VRAM) or a cloud connection at all.
