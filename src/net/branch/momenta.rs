use super::gradient::{BranchLogDensityGradient, BranchLogDensityGradientJoint};
use super::step_sizes::StepSizes;
use arrayfire::Array;

/// Momenta w.r.t. weights, bias and precision dimensions
pub struct BranchMomentaJoint {
    pub wrt_weights: Vec<Array<f32>>,
    pub wrt_biases: Vec<Array<f32>>,
    pub wrt_weight_precisions: Vec<Array<f32>>,
    pub wrt_bias_precisions: Vec<Array<f32>>,
    pub wrt_error_precision: Array<f32>,
}

impl BranchMomentaJoint {
    pub fn half_step(&mut self, step_sizes: &StepSizes, grad: &BranchLogDensityGradientJoint) {
        self.step(step_sizes, grad, 0.5)
    }

    pub fn full_step(&mut self, step_sizes: &StepSizes, grad: &BranchLogDensityGradientJoint) {
        self.step(step_sizes, grad, 1.0)
    }

    fn step(
        &mut self,
        step_sizes: &StepSizes,
        grad: &BranchLogDensityGradientJoint,
        fraction: f32,
    ) {
        for i in 0..self.wrt_weights.len() {
            self.wrt_weights[i] += fraction * &step_sizes.wrt_weights[i] * &grad.wrt_weights[i];
        }
        for i in 0..self.wrt_biases.len() {
            self.wrt_biases[i] += &step_sizes.wrt_biases[i] * fraction * &grad.wrt_biases[i];
        }
        for i in 0..self.wrt_weight_precisions.len() {
            self.wrt_biases[i] += &step_sizes.wrt_weight_precisions.as_ref().unwrap()[i]
                * fraction
                * &grad.wrt_weight_precisions[i];
        }
        for i in 0..self.wrt_bias_precisions.len() {
            self.wrt_biases[i] += &step_sizes.wrt_bias_precisions.as_ref().unwrap()[i]
                * fraction
                * &grad.wrt_bias_precisions[i];
        }
        self.wrt_error_precision +=
            step_sizes.wrt_error_precision.as_ref().unwrap() * fraction * &grad.wrt_error_precision;
    }

    // This is K(p) = p^T p / 2
    pub fn log_density(&self) -> f32 {
        let mut log_density: f32 = 0.;
        for i in 0..self.wrt_weights.len() {
            log_density += arrayfire::sum_all(&(&self.wrt_weights[i] * &self.wrt_weights[i])).0;
        }
        for i in 0..self.wrt_biases.len() {
            log_density += arrayfire::sum_all(&(&self.wrt_biases[i] * &self.wrt_biases[i])).0;
        }
        for i in 0..self.wrt_weight_precisions.len() {
            log_density += arrayfire::sum_all(
                &(&self.wrt_weight_precisions[i] * &self.wrt_weight_precisions[i]),
            )
            .0;
        }
        for i in 0..self.wrt_bias_precisions.len() {
            log_density +=
                arrayfire::sum_all(&(&self.wrt_bias_precisions[i] * &self.wrt_bias_precisions[i]))
                    .0;
        }
        log_density +=
            arrayfire::sum_all(&(&self.wrt_error_precision * &self.wrt_error_precision)).0;
        0.5 * log_density
    }

    pub fn wrt_weights(&self, index: usize) -> &Array<f32> {
        &self.wrt_weights[index]
    }

    pub fn wrt_biases(&self, index: usize) -> &Array<f32> {
        &self.wrt_biases[index]
    }

    pub fn wrt_weight_precisions(&self, index: usize) -> &Array<f32> {
        &self.wrt_weight_precisions[index]
    }

    pub fn wrt_bias_precisions(&self, index: usize) -> &Array<f32> {
        &self.wrt_bias_precisions[index]
    }

    pub fn wrt_error_precision(&self) -> &Array<f32> {
        &self.wrt_error_precision
    }
}

/// Momenta w.r.t. weights and bias dimensions
#[derive(Clone)]
pub struct BranchMomenta {
    pub wrt_weights: Vec<Array<f32>>,
    pub wrt_biases: Vec<Array<f32>>,
}

impl BranchMomenta {
    pub fn half_step(&mut self, step_sizes: &StepSizes, grad: &BranchLogDensityGradient) {
        self.step(step_sizes, grad, 0.5)
    }

    pub fn full_step(&mut self, step_sizes: &StepSizes, grad: &BranchLogDensityGradient) {
        self.step(step_sizes, grad, 1.0)
    }

    pub fn step(&mut self, step_sizes: &StepSizes, grad: &BranchLogDensityGradient, fraction: f32) {
        for i in 0..self.wrt_weights.len() {
            self.wrt_weights[i] += fraction * &step_sizes.wrt_weights[i] * &grad.wrt_weights[i];
        }
        for i in 0..self.wrt_biases.len() {
            self.wrt_biases[i] += &step_sizes.wrt_biases[i] * fraction * &grad.wrt_biases[i];
        }
    }

    // This is K(p) = p^T p / 2
    pub fn log_density(&self) -> f32 {
        let mut log_density: f32 = 0.;
        for i in 0..self.wrt_weights.len() {
            log_density += arrayfire::sum_all(&(&self.wrt_weights[i] * &self.wrt_weights[i])).0;
        }
        for i in 0..self.wrt_biases.len() {
            log_density += arrayfire::sum_all(&(&self.wrt_biases[i] * &self.wrt_biases[i])).0;
        }
        0.5 * log_density
    }

    pub fn wrt_weights(&self, index: usize) -> &Array<f32> {
        &self.wrt_weights[index]
    }

    pub fn wrt_biases(&self, index: usize) -> &Array<f32> {
        &self.wrt_biases[index]
    }
}
