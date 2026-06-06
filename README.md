# ternary-norm

**Normalization that stays ternary — batch norm, layer norm, and group norm with re-ternarization.**

[![Tests](https://img.shields.io/badge/tests-17%20passing-brightgreen)]()
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## Why This Exists

Normalization is the unsung hero of deep learning. Without it, deep networks don't converge. BatchNorm, LayerNorm, GroupNorm — they stabilize training by keeping activations centered and scaled.

But here's the problem: normalization produces *continuous* outputs. Even if your inputs are perfectly ternary {-1, 0, +1}, running them through standard BatchNorm gives you floating-point values. You'd need to re-quantize, introducing noise, losing information, breaking the ternary invariant.

**ternary-norm** solves this by making re-ternarization a first-class operation. The normalization computes real statistics (mean, variance) — because ternary distributions have real moments — then projects the output back to {-1, 0, +1} via threshold rounding. The result: a closed ternary pipeline where information flows without ever leaving ternary space.

## The Key Insight

Ternary distributions have well-defined statistics. A batch of {-1, 0, +1} values has a real-valued mean and variance. You can normalize those. What you can't do is *stay* normalized in ternary space without a rounding step. The insight: **threshold-based ternarization after normalization is not a bug — it's a feature.** It acts as a regularizer, pushing activations toward clean ternary decisions rather than drifting into continuous noise.

## Quick Start

```toml
[dependencies]
ternary-norm = "0.1"
```

```rust
use ternary_norm::*;

// Create a batch: 4 samples, 3 features each
let input = Tensor2D::new(
    vec![1.0, -1.0, 0.5, -1.0, 1.0, -0.5, 0.5, 0.5, 1.0, -0.5, -0.5, -1.0],
    4, 3,
);

// Ternary Batch Normalization
let mut tbn = TernaryBatchNorm::new(3);
let output = tbn.forward(&input);
// All output values are guaranteed to be in {-1.0, 0.0, +1.0}

// Layer normalization (per-sample, with ternarization)
let gamma = vec![1.0; 3];
let beta = vec![0.0; 3];
let normed = layer_norm(&input, &gamma, &beta, 1e-5, true, 0.5);

// Group normalization (2 groups of 3 features = 6 features total)
let input_6 = Tensor2D::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1, 6);
let gn = group_norm(&input_6, 2, &vec![1.0; 6], &vec![0.0; 6], 1e-5, true, 0.5);

// Standard vector norms (for custom pipelines)
let unit = l2_normalize(&[3.0, 4.0]); // [0.6, 0.8], ||unit|| = 1
```

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Tensor2D                           │
│  (rows × cols, f64, with ternarize(threshold) method) │
└──────────────────────────┬───────────────────────────┘
                           │
        ┌──────────────────┼──────────────────────┐
        │                  │                      │
  ┌─────▼──────┐   ┌──────▼───────┐   ┌─────────▼────────┐
  │ TernaryBatch│   │  layer_norm  │   │   group_norm     │
  │    Norm     │   │  (per-sample)│   │  (per-group)     │
  │ (per-feature│   │              │   │                  │
  │  + running  │   │              │   │                  │
  │   stats)    │   │              │   │                  │
  └─────┬──────┘   └──────┬───────┘   └─────────┬────────┘
        │                  │                      │
        └──────────────────┼──────────────────────┘
                           │
                  ┌────────▼─────────┐
                  │   ternarize(t)    │
                  │  x > t  →  +1    │
                  │  x < -t →  -1    │
                  │  else   →   0    │
                  └──────────────────┘
```

## How Ternary Batch Normalization Works

During training:

1. **Compute statistics** — per-feature mean μ and variance σ² across the batch
2. **Update running averages** — exponential moving average for inference mode
3. **Normalize** — x̂ = (x - μ) / √(σ² + ε)
4. **Scale and shift** — y = γx̂ + β (learnable affine transform)
5. **Ternarize** — output = ternarize(y, threshold)

The ternarization threshold (default 0.5) controls how aggressively the output is snapped back to {-1, 0, +1}. Higher threshold → more zeros (more sparsity). Lower threshold → more ±1 (more signal).

```rust
let mut tbn = TernaryBatchNorm::with_config(64, TernaryBatchNormConfig {
    momentum: 0.1,       // running stats update rate
    epsilon: 1e-5,       // numerical stability
    threshold: 0.5,      // ternarization cutoff
});
```

## Normalization Family

| Layer | Normalizes Over | Batch Dependent? | Best For |
|-------|----------------|-----------------|----------|
| **BatchNorm** | Each feature across batch | Yes | Large, consistent batch sizes |
| **LayerNorm** | All features within a sample | No | Variable batch sizes, transformers |
| **GroupNorm** | Feature groups within a sample | No | Between LayerNorm and InstanceNorm |
| **InstanceNorm** | Each feature per sample | No | Style transfer, per-sample tasks |

GroupNorm generalizes the family: GroupNorm with 1 group = LayerNorm. GroupNorm with groups = features = InstanceNorm.

## API Reference

### Core Type

```rust
struct Tensor2D {
    pub data: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl Tensor2D {
    fn new(data: Vec<f64>, rows: usize, cols: usize) -> Self;
    fn zeros(rows: usize, cols: usize) -> Self;
    fn get(&self, r: usize, c: usize) -> f64;
    fn set(&mut self, r: usize, c: usize, v: f64);
    fn row(&self, r: usize) -> &[f64];
    fn ternarize(&self, threshold: f64) -> Tensor2D;
}
```

### Batch Normalization

```rust
struct TernaryBatchNorm {
    pub config: TernaryBatchNormConfig,
    pub gamma: Vec<f64>,           // learnable scale
    pub beta: Vec<f64>,            // learnable shift
    pub running_mean: Vec<f64>,    // updated during training
    pub running_var: Vec<f64>,     // updated during training
}

impl TernaryBatchNorm {
    fn new(num_features: usize) -> Self;
    fn with_config(num_features: usize, config: TernaryBatchNormConfig) -> Self;
    fn forward(&mut self, input: &Tensor2D) -> Tensor2D;
}
```

### Layer/Group/Instance Norm

```rust
fn layer_norm(input: &Tensor2D, gamma: &[f64], beta: &[f64],
              epsilon: f64, ternarize_output: bool, threshold: f64) -> Tensor2D;

fn group_norm(input: &Tensor2D, num_groups: usize, gamma: &[f64], beta: &[f64],
              epsilon: f64, ternarize_output: bool, threshold: f64) -> Tensor2D;

fn instance_norm(input: &Tensor2D, gamma: &[f64], beta: &[f64],
                 epsilon: f64, ternarize_output: bool, threshold: f64) -> Tensor2D;
```

### Vector Norms

```rust
fn l1_norm(x: &[f64]) -> f64;
fn l1_normalize(x: &[f64]) -> Vec<f64>;
fn l2_norm(x: &[f64]) -> f64;
fn l2_normalize(x: &[f64]) -> Vec<f64>;
fn max_norm(x: &[f64]) -> f64;
fn max_normalize(x: &[f64]) -> Vec<f64>;
fn max_norm_clip(x: &[f64], max_val: f64) -> Vec<f64>;
```

## Real-World Example: Ternary Transformer

You're building a text classifier that runs on a Raspberry Pi. The model uses a ternary transformer — all weights and activations in {-1, 0, +1}. Layer normalization is critical for transformer stability.

```rust
// One transformer layer
let attention_output = ternary_attention(&input, &weights);  // from ternary-matmul
let normed = layer_norm(&attention_output, &gamma, &beta, 1e-5, true, 0.5);
let ff_output = ternary_ffn(&normed, &ff_weights);
let final = layer_norm(&ff_output, &gamma2, &beta2, 1e-5, true, 0.5);
// final.data: all values in {-1.0, 0.0, 1.0}
```

Without `ternary-norm`, you'd need to dequantize → float layer norm → re-quantize at every layer. That round-trip introduces noise and defeats the purpose of ternary inference.

## Performance Characteristics

- **TernaryBatchNorm**: O(batch × features) per forward pass. Running stats update is O(features).
- **LayerNorm**: O(batch × features) — compute mean/var per sample.
- **GroupNorm**: O(batch × features) — same order, constant factor depends on group count.
- **Vector norms**: O(n) — single pass over the vector.

Memory: Tensor2D stores f64 (8 bytes per element). A batch of 128 samples × 512 features = 512 KB. The ternarized output could be packed to 2 bits per element, reducing to 16 KB.

## Ecosystem Connections

Normalization sits between layers in the ternary network:

- [`ternary-conv`](https://github.com/SuperInstance/ternary-conv) — produces feature maps needing normalization
- [`ternary-matmul`](https://github.com/SuperInstance/ternary-matmul) — linear layers producing activations to normalize
- [`ternary-activation`](https://github.com/SuperInstance/ternary-activation) — applied after normalization
- [`ternary-optimizer`](https://github.com/SuperInstance/ternary-optimizer) — updates the γ and β parameters

## Open Questions

- **Optimal threshold**: The ternarization threshold (0.5 default) is a hyperparameter. Is there a principled way to learn it during training? Initial experiments suggest threshold ∈ [0.3, 0.7] works well.
- **Skip ternarization during training**: Some evidence suggests normalizing in float during training and only ternarizing at inference improves accuracy. This crate supports both via the `ternarize_output` flag.
- **Fused norm+activation**: Combining ternary batch norm with ternary ReLU could skip the intermediate float representation entirely.

## Testing

```bash
cargo test
```

17 tests covering: L1/L2/Max norm correctness, zero-vector edge cases, batch norm output validation (all values in {-1, 0, +1}), running statistics updates, layer norm variance reduction, group norm with different group sizes (2, 3, 6), instance norm equivalence with layer norm, tensor ternarization, and balanced distribution preservation.

## License

MIT
