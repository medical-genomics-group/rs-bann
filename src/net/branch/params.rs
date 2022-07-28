use super::momenta::BranchMomenta;
use super::step_sizes::StepSizes;
use arrayfire::{dim4, Array};
use std::fmt;

#[derive(Clone)]
pub(crate) struct BranchHyperparams {
    pub weight_precisions: Vec<f64>,
    pub bias_precisions: Vec<f64>,
    pub error_precision: f64,
}

/// Weights and biases
#[derive(Clone)]
pub(crate) struct BranchParams {
    pub weights: Vec<Array<f64>>,
    pub biases: Vec<Array<f64>>,
}

impl fmt::Debug for BranchParams {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.param_vec())
    }
}

impl BranchParams {
    pub fn from_param_vec(
        param_vec: &Vec<f64>,
        layer_widths: &Vec<usize>,
        num_markers: usize,
    ) -> Self {
        let mut weights: Vec<Array<f64>>;
        let mut biases: Vec<Array<f64>>;
        let mut prev_width = num_markers;
        let mut read_ix: usize = 0;
        for width in layer_widths {
            let num_weights = prev_width * width;
            weights.push(Array::new(
                &param_vec[read_ix..read_ix + num_weights],
                dim4!(prev_width as u64, *width as u64, 1, 1),
            ));
            prev_width = *width;
            read_ix += num_weights;
        }
        for width in layer_widths {
            let num_biases = width;
            Array::new(
                &param_vec[read_ix..read_ix + num_biases],
                dim4!(1, *width as u64, 1, 1),
            );
            read_ix += num_biases;
        }
        Self { weights, biases }
    }

    pub fn param_vec(&self) -> Vec<f64> {
        let mut host_vec = Vec::new();
        host_vec.resize(self.num_params(), 0.);
        let mut insert_ix: usize = 0;
        for i in 0..self.weights.len() {
            let len = self.weights[i].elements();
            self.weights[i].host(&mut host_vec[insert_ix..insert_ix + len]);
            insert_ix += len;
        }
        for i in 0..self.biases.len() {
            let len = self.biases[i].elements();
            self.biases[i].host(&mut host_vec[insert_ix..insert_ix + len]);
            insert_ix += len;
        }
        host_vec
    }

    fn num_params(&self) -> usize {
        let mut res: usize = 0;
        for i in 0..self.weights.len() {
            res += self.weights[i].elements();
        }
        for i in 0..self.biases.len() {
            res += self.biases[i].elements();
        }
        res
    }

    pub fn full_step(&mut self, step_sizes: &StepSizes, mom: &BranchMomenta) {
        for i in 0..self.weights.len() {
            self.weights[i] += &step_sizes.wrt_weights[i] * &mom.wrt_weights[i];
        }
        for i in 0..self.biases.len() {
            self.biases[i] += &step_sizes.wrt_biases[i] * &mom.wrt_biases[i];
        }
    }

    pub fn log_density(&self, hyperparams: &BranchHyperparams, rss: f64) -> f64 {
        let mut log_density: f64 = -0.5 * hyperparams.error_precision * rss;
        for i in 0..self.weights.len() {
            log_density -= hyperparams.weight_precisions[i]
                * 0.5
                * arrayfire::sum_all(&(&self.weights[i] * &self.weights[i])).0;
        }
        for i in 0..self.biases.len() {
            log_density -= hyperparams.bias_precisions[i]
                * 0.5
                * arrayfire::sum_all(&(&self.biases[i] * &self.biases[i])).0;
        }
        log_density
    }
}

#[cfg(test)]
mod tests {
    use super::BranchParams;
    use arrayfire::{dim4, Array};

    #[test]
    fn test_param_vec() {
        let params = BranchParams {
            weights: vec![
                Array::new(&[0.1, 0.2], dim4![2, 1, 1, 1]),
                Array::new(&[0.3], dim4![1, 1, 1, 1]),
            ],
            biases: vec![Array::new(&[0.4], dim4![1, 1, 1, 1])],
        };
        let exp = vec![0.1, 0.2, 0.3, 0.4];
        assert_eq!(params.param_vec(), exp);
    }
}