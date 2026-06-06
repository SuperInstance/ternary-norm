# ternary-norm

**Ternary normalization layers for neural networks operating in {-1, 0, +1} space.**

[![Tests](https://img.shields.io/badge/tests-17%20passing-brightgreen)]()

Ternary neural networks constrain weights and/or activations to the set {-1, 0, +1}, enabling
extreme model compression and efficient integer-only arithmetic on edge devices. However,
standard normalization layers (BatchNorm, LayerNorm, etc.) produce continuous outputs that
break the ternary constraint. **ternary-norm** provides normalization primitives that compute
statistics over ternary distributions and re-ternarize outputs, keeping the entire network
in ternary space.

## Why Ternary Normalization?

In a ternary network, every activation and weight is in {-1, 0, +1}. Passing through a
standard BatchNorm produces floating-point values, requiring an additional quantization step
and introducing noise. Ternary normalization layers handle this end-to-end:

1. **Compute statistics** (mean, variance) over the ternary distribution
2. **Normalize** using standard formulas (subtract mean, divide by std dev)
3. **Apply learnable scale/shift** (gamma, beta)
4. **Re-ternarize** the output back to {-1, 0, +1}

This creates a closed ternary pipeline where information flows through the network without
ever leaving ternary space.

## Features

- **L1/L2/Max normalization** — standard norm operations on vectors
- **Ternary Batch Normalization** — per-feature statistics with re-ternarization and running stats
- **Layer Normalization** — per-sample normalization with optional ternarization
- **Group Normalization** — divide features into groups, normalize within each group
- **Instance Normalization** — per-sample normalization (equivalent to layer norm for 2D)
- **Threshold-based ternarization** — configurable thresholds for the {-1, 0, +1} decision boundary

## Quick Start

```rust
use ternary_norm::{Tensor2D, TernaryBatchNorm, layer_norm, group_norm, l2_normalize};

// Create a batch of ternary-valued inputs
let input = Tensor2D::new(
    vec![
         1.0, -1.0,  0.5,
        -1.0,  1.0, -0.5,
         0.5,  0.5,  1.0,
        -0.5, -0.5, -1.0,
    ],
    4, 3, // 4 samples, 3 features
);

// Ternary Batch Normalization
let mut tbn = TernaryBatchNorm::new(3);
let output = tbn.forward(&input);
// All output values are guaranteed to be in {-1, 0, +1}

// Layer normalization with ternarization
let gamma = vec![1.0; 3];
let beta = vec![0.0; 3];
let normed = layer_norm(&input, &gamma, &beta, 1e-5, true, 0.5);

// Group normalization (2 groups of 3 features = 6 features total)
let input_6 = Tensor2D::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1, 6);
let gamma_6 = vec![1.0; 6];
let beta_6 = vec![0.0; 6];
let gn_output = group_norm(&input_6, 2, &gamma_6, &beta_6, 1e-5, true, 0.5);

// Standard L2 normalization
let v = vec![3.0, 4.0];
let unit = l2_normalize(&v); // [0.6, 0.8], ||unit|| = 1
```

## Architecture

### Tensor2D

The core data structure — a row-major 2D tensor with `rows × cols` elements. Supports:

- Element access (`get`, `set`)
- Row slicing (`row`, `row_mut`)
- Built-in ternarization (`ternarize(threshold)`)

### TernaryBatchNorm

The flagship layer. During training:

1. Computes per-feature mean μ and variance σ² across the batch
2. Updates exponential moving averages for inference
3. Normalizes: x̂ = (x - μ) / √(σ² + ε)
4. Applies learnable affine transform: y = γx̂ + β
5. Ternarizes output: output = ternarize(y, threshold)

Configuration via `TernaryBatchNormConfig`:
- `momentum` — running statistics update rate (default: 0.1)
- `epsilon` — numerical stability constant (default: 1e-5)
- `threshold` — ternarization threshold (default: 0.5)

### Layer Normalization

Normalizes across all features within a single sample. Reduces internal covariate shift
without depending on batch size. Can optionally ternarize the output.

### Group Normalization

Splits features into groups and normalizes within each group independently. More flexible
than LayerNorm (which is GroupNorm with 1 group) and InstanceNorm (which is GroupNorm
with groups = features).

## Research Context

Ternary weight networks were introduced in:

- Li, F., et al. "Ternary weight networks." *NIPS Workshop* (2016).
- Zhu, C., et al. "Trained ternary quantization." *ICLR* (2017).

This crate provides the normalization infrastructure needed to build complete ternary
network architectures where normalization layers preserve the ternary constraint.

## Testing

```bash
cargo test
```

17 comprehensive tests covering:
- L1/L2/Max norm correctness and edge cases (zero vectors)
- Ternary batch norm output validation (all values in {-1, 0, +1})
- Running statistics updates
- Layer norm variance reduction
- Group norm with different group configurations
- Instance norm correctness
- Tensor ternarization

## License

MIT
