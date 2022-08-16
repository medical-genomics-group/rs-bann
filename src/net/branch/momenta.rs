use super::base_branch::BranchLogDensityGradient;
use super::step_sizes::StepSizes;
use arrayfire::Array;

#[derive(Clone)]
pub(crate) struct BranchMomenta {
    pub wrt_weights: Vec<Array<f64>>,
    pub wrt_biases: Vec<Array<f64>>,
}

impl BranchMomenta {
    pub fn half_step(&mut self, step_sizes: &StepSizes, grad: &BranchLogDensityGradient) {
        for i in 0..self.wrt_weights.len() {
            self.wrt_weights[i] += &step_sizes.wrt_weights[i] * 0.5 * &grad.wrt_weights[i];
        }
        for i in 0..self.wrt_biases.len() {
            self.wrt_biases[i] += &step_sizes.wrt_biases[i] * 0.5 * &grad.wrt_biases[i];
        }
    }

    pub fn full_step(&mut self, step_sizes: &StepSizes, grad: &BranchLogDensityGradient) {
        for i in 0..self.wrt_weights.len() {
            self.wrt_weights[i] += &step_sizes.wrt_weights[i] * &grad.wrt_weights[i];
        }
        for i in 0..self.wrt_biases.len() {
            self.wrt_biases[i] += &step_sizes.wrt_biases[i] * &grad.wrt_biases[i];
        }
    }

    pub fn log_density(&self) -> f64 {
        let mut log_density: f64 = 0.;
        for i in 0..self.wrt_weights.len() {
            log_density += arrayfire::sum_all(&(&self.wrt_weights[i] * &self.wrt_weights[i])).0;
        }
        for i in 0..self.wrt_biases.len() {
            log_density += arrayfire::sum_all(&(&self.wrt_biases[i] * &self.wrt_biases[i])).0;
        }
        log_density
    }

    pub fn wrt_weights(&self, index: usize) -> &Array<f64> {
        &self.wrt_weights[index]
    }

    pub fn wrt_biases(&self, index: usize) -> &Array<f64> {
        &self.wrt_biases[index]
    }
}
