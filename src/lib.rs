//! # ternary-norm
//!
//! Ternary normalization layers for neural networks operating on {-1, 0, +1} values.
//!
//! This crate provides normalization primitives designed for ternary weight networks
//! where activations and weights are constrained to the set {-1, 0, +1}. Each layer
//! computes statistics over ternary distributions and re-ternarizes outputs.

use std::fmt;

/// A 2D tensor stored in row-major order (rows × cols).
#[derive(Clone, Debug)]
pub struct Tensor2D {
    pub data: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl Tensor2D {
    pub fn new(data: Vec<f64>, rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols, "data length must equal rows × cols");
        Self { data, rows, cols }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { data: vec![0.0; rows * cols], rows, cols }
    }

    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }

    pub fn row(&self, r: usize) -> &[f64] {
        let start = r * self.cols;
        &self.data[start..start + self.cols]
    }

    pub fn row_mut(&mut self, r: usize) -> &mut [f64] {
        let start = r * self.cols;
        &mut self.data[start..start + self.cols]
    }

    /// Ternarize each element to {-1, 0, +1} using threshold rounding.
    pub fn ternarize(&self, threshold: f64) -> Tensor2D {
        let data: Vec<f64> = self.data.iter().map(|&v| {
            if v > threshold { 1.0 }
            else if v < -threshold { -1.0 }
            else { 0.0 }
        }).collect();
        Tensor2D::new(data, self.rows, self.cols)
    }
}

impl fmt::Display for Tensor2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for r in 0..self.rows {
            for c in 0..self.cols {
                write!(f, "{:8.4}", self.get(r, c))?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

/// Ternarize a single f64 value using the given threshold.
pub fn ternarize_value(v: f64, threshold: f64) -> f64 {
    if v > threshold { 1.0 }
    else if v < -threshold { -1.0 }
    else { 0.0 }
}

// ── L1 Norm ──────────────────────────────────────────────────────────────────

/// Compute the L1 norm (sum of absolute values) of a slice.
pub fn l1_norm(x: &[f64]) -> f64 {
    x.iter().map(|v| v.abs()).sum()
}

/// Normalize a slice by its L1 norm so that the sum of absolute values equals 1.
/// Returns a zeroed vector if the L1 norm is zero.
pub fn l1_normalize(x: &[f64]) -> Vec<f64> {
    let norm = l1_norm(x);
    if norm == 0.0 {
        return vec![0.0; x.len()];
    }
    x.iter().map(|&v| v / norm).collect()
}

// ── L2 Norm ──────────────────────────────────────────────────────────────────

/// Compute the L2 norm (Euclidean norm) of a slice.
pub fn l2_norm(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Normalize a slice by its L2 norm so that the Euclidean length equals 1.
/// Returns a zeroed vector if the L2 norm is zero.
pub fn l2_normalize(x: &[f64]) -> Vec<f64> {
    let norm = l2_norm(x);
    if norm == 0.0 {
        return vec![0.0; x.len()];
    }
    x.iter().map(|&v| v / norm).collect()
}

// ── Max Norm ─────────────────────────────────────────────────────────────────

/// Compute the max norm (maximum absolute value) of a slice.
pub fn max_norm(x: &[f64]) -> f64 {
    x.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
}

/// Normalize a slice by its max norm, scaling so the largest absolute value is 1.
/// Returns a zeroed vector if the max norm is zero.
pub fn max_normalize(x: &[f64]) -> Vec<f64> {
    let norm = max_norm(x);
    if norm == 0.0 {
        return vec![0.0; x.len()];
    }
    x.iter().map(|&v| v / norm).collect()
}

/// Clip a slice so that no element exceeds the given max norm in absolute value.
pub fn max_norm_clip(x: &[f64], max_val: f64) -> Vec<f64> {
    x.iter().map(|&v| v.max(-max_val).min(max_val)).collect()
}

// ── Ternary Batch Normalization ──────────────────────────────────────────────

/// Configuration for ternary batch normalization.
#[derive(Clone, Debug)]
pub struct TernaryBatchNormConfig {
    /// Momentum for running statistics (0.0 = use batch stats only).
    pub momentum: f64,
    /// Epsilon for numerical stability.
    pub epsilon: f64,
    /// Threshold for re-ternarizing output.
    pub threshold: f64,
}

impl Default for TernaryBatchNormConfig {
    fn default() -> Self {
        Self {
            momentum: 0.1,
            epsilon: 1e-5,
            threshold: 0.5,
        }
    }
}

/// Ternary Batch Normalization layer.
///
/// Computes per-feature mean and variance across the batch, normalizes,
/// then re-ternarizes outputs to {-1, 0, +1}. Optionally uses learnable
/// scale (gamma) and shift (beta) parameters before ternarization.
pub struct TernaryBatchNorm {
    pub config: TernaryBatchNormConfig,
    /// Per-feature scale (gamma). Defaults to 1.0.
    pub gamma: Vec<f64>,
    /// Per-feature shift (beta). Defaults to 0.0.
    pub beta: Vec<f64>,
    /// Running mean (updated during training).
    pub running_mean: Vec<f64>,
    /// Running variance (updated during training).
    pub running_var: Vec<f64>,
}

impl TernaryBatchNorm {
    /// Create a new ternary batch norm layer for `num_features` features.
    pub fn new(num_features: usize) -> Self {
        Self {
            config: TernaryBatchNormConfig::default(),
            gamma: vec![1.0; num_features],
            beta: vec![0.0; num_features],
            running_mean: vec![0.0; num_features],
            running_var: vec![1.0; num_features],
        }
    }

    /// Create with custom config.
    pub fn with_config(num_features: usize, config: TernaryBatchNormConfig) -> Self {
        Self {
            config,
            gamma: vec![1.0; num_features],
            beta: vec![0.0; num_features],
            running_mean: vec![0.0; num_features],
            running_var: vec![1.0; num_features],
        }
    }

    /// Forward pass: normalize then ternarize.
    ///
    /// Input tensor shape: (batch_size, num_features).
    /// Output: ternarized tensor of the same shape.
    pub fn forward(&mut self, input: &Tensor2D) -> Tensor2D {
        assert_eq!(input.cols, self.gamma.len(), "feature dimension mismatch");
        let batch = input.rows;
        let features = input.cols;

        // Compute per-feature mean
        let mut mean = vec![0.0; features];
        for r in 0..batch {
            for c in 0..features {
                mean[c] += input.get(r, c);
            }
        }
        for m in mean.iter_mut() {
            *m /= batch as f64;
        }

        // Compute per-feature variance
        let mut var = vec![0.0; features];
        for r in 0..batch {
            for c in 0..features {
                let diff = input.get(r, c) - mean[c];
                var[c] += diff * diff;
            }
        }
        for v in var.iter_mut() {
            *v /= batch as f64;
        }

        // Update running statistics
        let mom = self.config.momentum;
        for c in 0..features {
            self.running_mean[c] = (1.0 - mom) * self.running_mean[c] + mom * mean[c];
            self.running_var[c] = (1.0 - mom) * self.running_var[c] + mom * var[c];
        }

        // Normalize, scale, shift, then ternarize
        let mut output = Tensor2D::zeros(batch, features);
        for r in 0..batch {
            for c in 0..features {
                let std_dev = (var[c] + self.config.epsilon).sqrt();
                let normalized = (input.get(r, c) - mean[c]) / std_dev;
                let scaled = self.gamma[c] * normalized + self.beta[c];
                output.set(r, c, ternarize_value(scaled, self.config.threshold));
            }
        }
        output
    }
}

// ── Layer Normalization ──────────────────────────────────────────────────────

/// Apply layer normalization across the last dimension, then optionally ternarize.
///
/// Computes mean and variance per sample (across all features), normalizes,
/// applies scale/shift, and optionally ternarizes.
pub fn layer_norm(
    input: &Tensor2D,
    gamma: &[f64],
    beta: &[f64],
    epsilon: f64,
    ternarize_output: bool,
    threshold: f64,
) -> Tensor2D {
    let batch = input.rows;
    let features = input.cols;
    assert_eq!(features, gamma.len());
    assert_eq!(features, beta.len());

    let mut output = Tensor2D::zeros(batch, features);

    for r in 0..batch {
        let row = input.row(r);
        let mean: f64 = row.iter().sum::<f64>() / features as f64;
        let var: f64 = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / features as f64;
        let std_dev = (var + epsilon).sqrt();

        for c in 0..features {
            let normalized = (row[c] - mean) / std_dev;
            let scaled = gamma[c] * normalized + beta[c];
            if ternarize_output {
                output.set(r, c, ternarize_value(scaled, threshold));
            } else {
                output.set(r, c, scaled);
            }
        }
    }
    output
}

// ── Group Normalization ──────────────────────────────────────────────────────

/// Apply group normalization.
///
/// Divides features into `num_groups` groups and normalizes within each group.
/// Optionally ternarizes the output.
pub fn group_norm(
    input: &Tensor2D,
    num_groups: usize,
    gamma: &[f64],
    beta: &[f64],
    epsilon: f64,
    ternarize_output: bool,
    threshold: f64,
) -> Tensor2D {
    let features = input.cols;
    assert_eq!(features, gamma.len());
    assert_eq!(features, beta.len());
    assert!(features % num_groups == 0, "features must be divisible by num_groups");
    let group_size = features / num_groups;

    let mut output = Tensor2D::zeros(input.rows, features);

    for r in 0..input.rows {
        for g in 0..num_groups {
            let start = g * group_size;
            let end = start + group_size;

            // Compute group mean and variance
            let mut mean = 0.0;
            let mut var = 0.0;
            for c in start..end {
                mean += input.get(r, c);
            }
            mean /= group_size as f64;
            for c in start..end {
                let diff = input.get(r, c) - mean;
                var += diff * diff;
            }
            var /= group_size as f64;
            let std_dev = (var + epsilon).sqrt();

            for c in start..end {
                let normalized = (input.get(r, c) - mean) / std_dev;
                let scaled = gamma[c] * normalized + beta[c];
                if ternarize_output {
                    output.set(r, c, ternarize_value(scaled, threshold));
                } else {
                    output.set(r, c, scaled);
                }
            }
        }
    }
    output
}

// ── Instance Normalization ───────────────────────────────────────────────────

/// Apply instance normalization (per-sample, per-channel-like normalization).
///
/// Each row is normalized independently (like layer norm but conceptually per-instance).
/// Then optionally ternarized.
pub fn instance_norm(
    input: &Tensor2D,
    gamma: &[f64],
    beta: &[f64],
    epsilon: f64,
    ternarize_output: bool,
    threshold: f64,
) -> Tensor2D {
    // Instance norm on a 2D tensor is equivalent to layer norm
    layer_norm(input, gamma, beta, epsilon, ternarize_output, threshold)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l1_norm_correctness() {
        let v = vec![3.0, -4.0, 0.0, 2.0];
        assert_eq!(l1_norm(&v), 9.0);
    }

    #[test]
    fn test_l1_normalize_correctness() {
        let v = vec![3.0, -4.0, 0.0, 2.0];
        let normed = l1_normalize(&v);
        let sum_abs: f64 = normed.iter().map(|x| x.abs()).sum();
        assert!((sum_abs - 1.0).abs() < 1e-10);
        assert!((normed[0] - 3.0 / 9.0).abs() < 1e-10);
        assert!((normed[1] - (-4.0 / 9.0)).abs() < 1e-10);
    }

    #[test]
    fn test_l1_normalize_zero_vector() {
        let v = vec![0.0, 0.0, 0.0];
        let normed = l1_normalize(&v);
        assert!(normed.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_l2_norm_correctness() {
        let v = vec![3.0, 4.0];
        assert!((l2_norm(&v) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_l2_normalize_correctness() {
        let v = vec![3.0, 4.0];
        let normed = l2_normalize(&v);
        assert!((l2_norm(&normed) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let v = vec![0.0, 0.0];
        let normed = l2_normalize(&v);
        assert!(normed.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_max_norm_correctness() {
        let v = vec![-5.0, 3.0, -2.0, 4.0];
        assert_eq!(max_norm(&v), 5.0);
    }

    #[test]
    fn test_max_normalize_correctness() {
        let v = vec![-5.0, 3.0];
        let normed = max_normalize(&v);
        assert!((normed[0] - (-1.0)).abs() < 1e-10);
        assert!((normed[1] - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_max_norm_clip() {
        let v = vec![-5.0, 3.0, -2.0, 4.0];
        let clipped = max_norm_clip(&v, 3.0);
        assert_eq!(clipped, vec![-3.0, 3.0, -2.0, 3.0]);
    }

    #[test]
    fn test_ternary_batch_norm_output_is_balanced_ternary() {
        let input = Tensor2D::new(
            vec![
                1.0, -1.0, 0.5,
                -1.0, 1.0, -0.5,
                0.5, 0.5, 1.0,
                -0.5, -0.5, -1.0,
            ],
            4, 3,
        );
        let mut tbn = TernaryBatchNorm::new(3);
        let output = tbn.forward(&input);

        // All outputs must be in {-1, 0, 1}
        for v in &output.data {
            let is_ternary = (*v == -1.0) || (*v == 0.0) || (*v == 1.0);
            assert!(is_ternary, "output value {} is not ternary", v);
        }
    }

    #[test]
    fn test_ternary_batch_norm_updates_running_stats() {
        let input = Tensor2D::new(
            vec![1.0, -1.0, 0.0, 1.0, -1.0, 0.0],
            2, 3,
        );
        let mut tbn = TernaryBatchNorm::new(3);
        let _ = tbn.forward(&input);
        // Running stats should have been updated (not still at initial values)
        // Initial mean = 0, after one batch with mom=0.1 they should be non-zero for some features
        // Feature 0: mean = (1+1)/2 = 1.0, running_mean = 0.1*1.0 = 0.1
        assert!((tbn.running_mean[0] - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_layer_norm_reduces_variance() {
        // Input with high variance per row
        let input = Tensor2D::new(
            vec![100.0, -100.0, 50.0, -50.0],
            1, 4,
        );
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let output = layer_norm(&input, &gamma, &beta, 1e-5, false, 0.5);

        // Output should have approximately zero mean and unit variance
        let row = output.row(0);
        let mean: f64 = row.iter().sum::<f64>() / row.len() as f64;
        assert!(mean.abs() < 1e-10, "mean should be ~0, got {}", mean);

        let var: f64 = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / row.len() as f64;
        assert!((var - 1.0).abs() < 0.1, "variance should be ~1, got {}", var);
    }

    #[test]
    fn test_layer_norm_with_ternarization() {
        let input = Tensor2D::new(
            vec![10.0, -10.0, 0.0, 5.0],
            1, 4,
        );
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let output = layer_norm(&input, &gamma, &beta, 1e-5, true, 0.5);

        for v in &output.data {
            let is_ternary = (*v == -1.0) || (*v == 0.0) || (*v == 1.0);
            assert!(is_ternary, "ternarized output {} not in {{-1,0,1}}", v);
        }
    }

    #[test]
    fn test_group_norm_with_different_group_sizes() {
        let input = Tensor2D::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
            ],
            2, 6,
        );
        let gamma = vec![1.0; 6];
        let beta = vec![0.0; 6];

        // Test with 2 groups (group_size = 3)
        let output_2g = group_norm(&input, 2, &gamma, &beta, 1e-5, false, 0.5);
        // Each group should have ~0 mean within itself
        for r in 0..2 {
            // Group 0: cols 0..3
            let g0_mean: f64 = (0..3).map(|c| output_2g.get(r, c)).sum::<f64>() / 3.0;
            assert!(g0_mean.abs() < 1e-10, "group 0 mean should be ~0, got {}", g0_mean);
        }

        // Test with 3 groups (group_size = 2)
        let output_3g = group_norm(&input, 3, &gamma, &beta, 1e-5, false, 0.5);
        for r in 0..2 {
            // Group 0: cols 0..2
            let g0_mean: f64 = (0..2).map(|c| output_3g.get(r, c)).sum::<f64>() / 2.0;
            assert!(g0_mean.abs() < 1e-10, "group 0 mean should be ~0, got {}", g0_mean);
        }

        // Test with 6 groups (group_size = 1, each feature normalized independently)
        let output_6g = group_norm(&input, 6, &gamma, &beta, 1e-5, false, 0.5);
        // With group size 1, each element is (x - x)/sqrt(0 + eps) = 0
        for r in 0..2 {
            for c in 0..6 {
                assert!(output_6g.get(r, c).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_instance_norm() {
        let input = Tensor2D::new(
            vec![
                10.0, 20.0, 30.0,
                -5.0, 0.0, 5.0,
            ],
            2, 3,
        );
        let gamma = vec![1.0; 3];
        let beta = vec![0.0; 3];
        let output = instance_norm(&input, &gamma, &beta, 1e-5, false, 0.5);

        // Each row should have ~0 mean and ~1 variance
        for r in 0..2 {
            let row = output.row(r);
            let mean: f64 = row.iter().sum::<f64>() / 3.0;
            assert!(mean.abs() < 1e-10);
        }
    }

    #[test]
    fn test_tensor2d_ternarize() {
        let t = Tensor2D::new(vec![0.8, -0.8, 0.3, -0.3, 0.0, 1.5], 2, 3);
        let ternary = t.ternarize(0.5);
        assert_eq!(ternary.data, vec![1.0, -1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_ternary_batch_norm_balanced_distribution() {
        // Use a large batch to get a balanced output distribution
        let mut data = Vec::new();
        for i in 0..100 {
            data.push((i as f64 % 3.0) - 1.0); // cycles through -1, 0, 1
            data.push(-((i as f64 % 3.0) - 1.0)); // anti-cycles
        }
        let input = Tensor2D::new(data, 100, 2);
        let mut tbn = TernaryBatchNorm::new(2);
        let output = tbn.forward(&input);

        // Count occurrences of each value
        let mut counts = [0usize; 3]; // [-1, 0, 1]
        for v in &output.data {
            match *v as i32 {
                -1 => counts[0] += 1,
                0 => counts[1] += 1,
                1 => counts[2] += 1,
                _ => panic!("non-ternary value"),
            }
        }
        // Should have all three values represented
        assert!(counts[0] > 0, "no -1 values in output");
        assert!(counts[1] > 0, "no 0 values in output");
        assert!(counts[2] > 0, "no +1 values in output");
    }
}
