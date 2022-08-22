use super::momenta::BranchMomenta;
use super::step_sizes::StepSizes;
use crate::to_host;
use arrayfire::{dim4, Array};
use serde::ser::{Serialize, SerializeStruct, Serializer};
use std::fmt;

#[derive(Clone)]
pub struct BranchHyperparams {
    pub weight_precisions: Vec<Array<f64>>,
    pub bias_precisions: Vec<f64>,
    pub error_precision: f64,
}

impl BranchHyperparams {
    fn weight_precisions_to_host(&self) -> Vec<Vec<f64>> {
        let mut res = Vec::with_capacity(self.weight_precisions.len());
        for arr in &self.weight_precisions {
            res.push(to_host(arr));
        }
        res
    }
}

impl Serialize for BranchHyperparams {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("BranchHyperparams", 3)?;
        // 3 is the number of fields in the struct.
        state.serialize_field("weight_precisions", &self.weight_precisions_to_host())?;
        state.serialize_field("bias_precisions", &self.bias_precisions)?;
        state.serialize_field("error_precision", &self.error_precision)?;
        state.end()
    }
}

/// Weights and biases
#[derive(Clone)]
pub struct BranchParams {
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
        let mut weights: Vec<Array<f64>> = vec![];
        let mut biases: Vec<Array<f64>> = vec![];
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
        for width in &layer_widths[..layer_widths.len() - 1] {
            let num_biases = width;
            biases.push(Array::new(
                &param_vec[read_ix..read_ix + num_biases],
                dim4!(1, *width as u64, 1, 1),
            ));
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

    pub fn weights(&self, index: usize) -> &Array<f64> {
        &self.weights[index]
    }

    pub fn biases(&self, index: usize) -> &Array<f64> {
        &self.biases[index]
    }
}

#[cfg(test)]
mod tests {
    use super::BranchParams;
    use crate::to_host;
    use arrayfire::{dim4, Array};

    fn test_params() -> BranchParams {
        BranchParams {
            weights: vec![
                Array::new(&[0.1, 0.2], dim4![2, 1, 1, 1]),
                Array::new(&[0.3], dim4![1, 1, 1, 1]),
            ],
            biases: vec![Array::new(&[0.4], dim4![1, 1, 1, 1])],
        }
    }

    #[test]
    fn test_param_vec() {
        let params = test_params();
        let exp = vec![0.1, 0.2, 0.3, 0.4];
        assert_eq!(params.param_vec(), exp);
    }

    #[test]
    fn test_from_param_vec() {
        let params = test_params();
        let param_vec = params.param_vec();
        let params_loaded = BranchParams::from_param_vec(&param_vec, &vec![1, 1], 2);
        assert_eq!(params.weights.len(), params_loaded.weights.len());
        for ix in 0..params.weights.len() {
            assert_eq!(
                to_host(&params.weights[ix]),
                to_host(&params_loaded.weights[ix])
            );
        }
        assert_eq!(params.biases.len(), params_loaded.biases.len());
        for ix in 0..params.biases.len() {
            assert_eq!(
                to_host(&params.biases[ix]),
                to_host(&params_loaded.biases[ix])
            );
        }
    }
}
