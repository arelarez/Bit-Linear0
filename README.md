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

## Training model b1.58 

From scratch, the b1.58 model training is not carried out using conventional training method concepts such as the usual *Post-Training Quantization (PTQ), the model is essentially trained with *high precision* which is then forcefully pruned to values ```-1, 0, 1``` this will lead to the mathematical consequence of *performance collapse.* Therefore, BitNet is designed through the **Quantization-Aware Training (QAT)** paradigm which is integrated directly into the *forward-backward pass cycle.*

### Weight Quantity in Ternary Back-prop

Each linear layer is replaced with a ```BitLinear``` layer before matrix operations are applied on each *forward pass,* the high-order *floating-point* weights (```\W ~ \R```) d converted to Ternary weights (```W_q \in \{-1, 0, 1\}```) using the absmax (*absolute maximum*) scaling function:

#### Scaling Factor (\gamma):

The average absolute magnitude value is calculated from the weight matrix to obtain the normalized scale: 

```\gamma = \frac{1}{n} \sum \vert{}W\Vert{}```

#### Ternary Quantization Function with Threshold:

The weight is divided by ```\gamma```, then rounded to the nearest value between ```{-1, 0, 1}```:

```\W_q = \Round (\frac{W}{\gamma})```

Where the rounding function maps the values to the three points based on a certain threshold; such as the use of a modified sign function (*sign*) with a zero threshold.

### Activation Quantization (INT8)

The heavy matrix multiplication operation is *removed* and completely replaced with *addition*, act input activations (```X```) are also dynamically quantized into 8-bit (INT8) ranges per-token or per-block:

```\X_q = \Clip (X × {Q_b}{\max(\vert{}X\vert{})}, -Q_b, Q_b - 1)```

Where ```Q_b = 2^8-1 = 127```. With this representation, the complicated matrix multiplication is transformed into a very efficient integer matrix multiplication operation.

### Backpropagation Challenge: Straight-Through Estimator (STE) 

The quantization rounding function (```W_q = \Round \(W/\gamma\)```) has a derivative (*Gradient*) of zero at almost all points and infinity at the transition points.This makes standard backpropagation impossible to run.

To address this, BitNet uses a Straight-Through Estimator (STE) that ignores the rounding operation and passes the gradient directly to the original floating-point weights (```W```).

```\frac{\partial \mathcal{L}}{\partial W} \approx \frac{\partial \mathcal{L}}{\partial W_q}```

Operations like AdamW store high precision *Master Weight* like the neural networks (AI) architecture I will build here using (*FP16 or BF16*) y which is then updated by the *Back-prop* gradient, while the Ternary weights (```W_q```) are only used temporarily during the *forward pass* to calculate the *loss* and activation.

## The LLM with the basis {AbsMax (Absolute maximum) :: b1.58}

Quantization-Aware Training (QAT) architecture is structured on Continuous value mappings from Floating-point (FP16/BF16) interwoven in a very limited *discrete* domain; y namely ```{-1, 0, 1}``` for weights, and in this architecture the use of INT8 is used for activation, so that the weight vector does not lose the semantic information of the model.

#### AbsMax on weight

For the weight matrix ```W``` BitNet b1.58 using the BasMax variant in the form of Mean Absolute Value as the scaling factor (```\gamma```):

```\gamma = \frac{1}{n} \sum_{i=1}^{n} \vert{}w_i\vert{}```

the weight tensor is divided by ```\gamma``` so that its values are within the standardized range. After normalization, the weights are rounded to the nearest integer via the *threshold* function:

```\W_q = \Round (\frac{W}{\gamma})```

The result of the operation will produce a matrix value that purely contains discrete ```-1, 0, 1``` factors ```\gamma``` are stored separately as *scale factors* per-layer or per-block for dik to be multiplied again during the matrix scale dequantization process (scaling factor recovery) takes place.

#### AbsMax on Activation 

Unlike weights which are static after training is complete, activations (```X```) are dynamic every time a new token is processed. AbsMax scaling on Activation is done per-token or per-block to keep the range in the INT8 format ```Q_b = 127```:

```\gamma_x = \frac{\max(\vert{}X\vert{})}{Q_b}```
    ```\X_q = \Clip\(Round\(X·{Q_b/\max\(\vert\X\vert\)\), -Q_b, Q_b - 1)\```

The importance of the AbsMax function approach is preventing numerical overflow or underflow when large matrix multiplication operations are simplified to pure integer addition.


## Roadmap LLM b1.58

The Custom PyTorch Module integrated into the C++ programming language structure on the Ternary LLM model that I created uses an integer-based matrix multiplication operation or Straight-Through Estimator (STE) base where; my application Activation quantization is applied to INT8 then the weight quantization *Master Weight* is optimized with Ternary (```-1, 0, 1```) using the AbsMax function. This is all because rounding (```\Round```) has zero derivative at almost all points, that's why the backward pass function is optimized using Custom STE so that so that the gradient can flow directly to the high-precision master weights.

## The backbone with RMSNorm & SwiGLU

Inter-layer quantization via NLL and FFN is normalized using RMSNorm (Root Mean Square Normalization) because it is more stable combined with extreme quantization, Use of ```SwiGLU`` (Swish-Gated Linear Unit) in Feed-forward Network (FFN) is aimed at maximizing the capacity of non-linear representation.

- Then to adjust the long window context, it is applied using Rotary Position Embedding (RoPE).

### Optimizer & Master Weights Management (FP16, b1.58)

**Master Weight (FP16/BF16)**: Stored in the memory of the Optimizer (such as AdamW) to receive precise gradient divisions.

**Ternary Weights (1.57-bit)**: Calculated on-the-fly from the Master Weight at the start of each forward pass.

