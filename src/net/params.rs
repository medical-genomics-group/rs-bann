use super::branch::gradient::BranchLogDensityGradient;
use super::branch::momentum::BranchMomentumJoint;
use super::branch::step_sizes::StepSizes;
use super::branch::{branch::BranchCfg, momentum::Momentum};
use crate::af_helpers::to_host;
use arrayfire::{dim4, Array};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Branch specific model hyperparameters
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct BranchHyperparameters {
    pub(crate) num_params: usize,
    pub(crate) num_markers: usize,
    pub(crate) layer_widths: Vec<usize>,
}

impl BranchHyperparameters {
    pub(crate) fn from_cfg(cfg: &BranchCfg) -> Self {
        Self {
            num_params: cfg.num_params,
            num_markers: cfg.num_markers,
            layer_widths: cfg.layer_widths.clone(),
        }
    }
}

/// All hyperparameters of the model
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct ModelHyperparameters {
    pub(crate) branch_hyperparams: Vec<BranchHyperparameters>,
    pub(crate) precision_hyperparams: NetworkPrecisionHyperparameters,
}

impl ModelHyperparameters {
    pub(crate) fn new(
        precision_hyperparams: &NetworkPrecisionHyperparameters,
        branch_cfgs: &[BranchCfg],
    ) -> Self {
        Self {
            branch_hyperparams: branch_cfgs
                .iter()
                .map(BranchHyperparameters::from_cfg)
                .collect(),
            precision_hyperparams: precision_hyperparams.clone(),
        }
    }
}

/// Hyperparameters of a prior distribution of precision parameters.
/// The precisions always have Gamma priors, which are parametrized by
/// shape and scale hyperparameters.
#[derive(Clone, Serialize, Deserialize)]
pub struct PrecisionHyperparameters {
    pub shape: f32,
    pub scale: f32,
}

impl PrecisionHyperparameters {
    pub fn new(shape: f32, scale: f32) -> Self {
        Self { shape, scale }
    }
}

impl Default for PrecisionHyperparameters {
    fn default() -> Self {
        Self {
            shape: 1.0,
            scale: 1.0,
        }
    }
}

/// All hyperparameters of precision prior distributions for the model.
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct NetworkPrecisionHyperparameters {
    /// Hyperparams of precisions in the dense layers
    pub dense: PrecisionHyperparameters,
    /// Hyperparams of precisions in the summary layer
    pub summary: PrecisionHyperparameters,
    /// Hyperparams of precisions in the output layer
    pub output: PrecisionHyperparameters,
}

impl NetworkPrecisionHyperparameters {
    /// (shape, scale) of dense, summary or output layer, depending on index
    pub fn layer_prior_hyperparams(&self, layer_index: usize, num_layers: usize) -> (f32, f32) {
        if layer_index == num_layers - 1 {
            (
                self.output_layer_prior_shape(),
                self.output_layer_prior_scale(),
            )
        } else if layer_index == num_layers - 2 {
            (
                self.summary_layer_prior_shape(),
                self.summary_layer_prior_scale(),
            )
        } else {
            (
                self.dense_layer_prior_shape(),
                self.dense_layer_prior_scale(),
            )
        }
    }

    pub fn dense_layer_prior_shape(&self) -> f32 {
        self.dense.shape
    }

    pub fn dense_layer_prior_scale(&self) -> f32 {
        self.dense.scale
    }

    pub fn summary_layer_prior_shape(&self) -> f32 {
        self.summary.shape
    }

    pub fn summary_layer_prior_scale(&self) -> f32 {
        self.summary.scale
    }

    pub fn output_layer_prior_shape(&self) -> f32 {
        self.output.shape
    }

    pub fn output_layer_prior_scale(&self) -> f32 {
        self.output.scale
    }
}

/// Precision parameters stored on Host.
#[derive(Clone, Serialize, Deserialize)]
pub struct BranchPrecisionsHost {
    // One Array per layer, possibly scalars
    pub weight_precisions: Vec<Vec<f32>>,
    // scalars
    pub bias_precisions: Vec<Vec<f32>>,
    // scalar
    pub error_precision: Vec<f32>,
}

impl BranchPrecisionsHost {
    pub fn set_output_layer_precision(&mut self, precision: f32) {
        *self
            .weight_precisions
            .last_mut()
            .expect("Branch weight precisions is empty!") = vec![precision];
    }
}

/// Precision parameters stored on Device.
#[derive(Clone, Serialize, Deserialize)]
pub struct BranchPrecisions {
    // One Array per layer, possibly scalars
    pub weight_precisions: Vec<Array<f32>>,
    // scalars
    pub bias_precisions: Vec<Array<f32>>,
    // scalar
    pub error_precision: Array<f32>,
}

impl BranchPrecisions {
    pub fn from_host(host: &BranchPrecisionsHost) -> Self {
        Self {
            weight_precisions: host
                .weight_precisions
                .iter()
                .map(|v| Array::new(v, dim4!(v.len() as u64)))
                .collect(),
            bias_precisions: host
                .bias_precisions
                .iter()
                .map(|v| Array::new(v, dim4!(1)))
                .collect(),
            error_precision: Array::new(&host.error_precision, dim4!(1)),
        }
    }

    pub fn to_host(&self) -> BranchPrecisionsHost {
        BranchPrecisionsHost {
            weight_precisions: self
                .weight_precisions
                .iter()
                .map(|arr| to_host(arr))
                .collect(),
            bias_precisions: self
                .bias_precisions
                .iter()
                .map(|arr| to_host(arr))
                .collect(),
            error_precision: to_host(&self.error_precision),
        }
    }

    pub fn param_vec(&self) -> Vec<f32> {
        let mut host_vec = Vec::new();
        host_vec.resize(self.num_params(), 0.);
        let mut insert_ix: usize = 0;
        for i in 0..self.weight_precisions.len() {
            let len = self.weight_precisions[i].elements();
            self.weight_precisions[i].host(&mut host_vec[insert_ix..insert_ix + len]);
            insert_ix += len;
        }
        for i in 0..self.bias_precisions.len() {
            let len = self.bias_precisions[i].elements();
            self.bias_precisions[i].host(&mut host_vec[insert_ix..insert_ix + len]);
            insert_ix += len;
        }
        self.error_precision
            .host(&mut host_vec[insert_ix..insert_ix + 1]);
        host_vec
    }

    fn num_params(&self) -> usize {
        let mut res: usize = 1;
        for i in 0..self.weight_precisions.len() {
            res += self.weight_precisions[i].elements();
        }
        for i in 0..self.bias_precisions.len() {
            res += self.bias_precisions[i].elements();
        }
        res
    }

    pub fn set_output_layer_precision(&mut self, precision: f32) {
        *self
            .weight_precisions
            .last_mut()
            .expect("Branch weight precisions is empty!") = Array::new(&[precision], dim4!(1));
    }

    pub fn num_precisions(&self) -> usize {
        1 + self.bias_precisions.len()
            + self
                .weight_precisions
                .iter()
                .map(|arr| arr.elements())
                .sum::<usize>()
    }

    pub fn layer_weight_precisions(&self, layer_index: usize) -> &Array<f32> {
        &self.weight_precisions[layer_index]
    }

    pub fn layer_bias_precision(&self, layer_index: usize) -> &Array<f32> {
        &self.bias_precisions[layer_index]
    }

    pub fn layer_weight_precisions_mut(&mut self, layer_index: usize) -> &mut Array<f32> {
        &mut self.weight_precisions[layer_index]
    }

    pub fn layer_bias_precision_mut(&mut self, layer_index: usize) -> &mut Array<f32> {
        &mut self.bias_precisions[layer_index]
    }

    pub fn error_precision_mut(&mut self) -> &mut Array<f32> {
        &mut self.error_precision
    }

    pub fn full_step(&mut self, step_sizes: &StepSizes, mom: &BranchMomentumJoint) {
        for i in 0..self.weight_precisions.len() {
            self.weight_precisions[i] += &step_sizes.wrt_weight_precisions.as_ref().unwrap()[i]
                * &mom.wrt_weight_precisions[i];
        }
        for i in 0..self.bias_precisions.len() {
            self.bias_precisions[i] +=
                &step_sizes.wrt_bias_precisions.as_ref().unwrap()[i] * &mom.wrt_bias_precisions[i];
        }
        self.error_precision +=
            step_sizes.wrt_error_precision.as_ref().unwrap() * &mom.wrt_error_precision;
    }
}

/// Weights and biases
#[derive(Clone)]
pub struct BranchParams {
    pub weights: Vec<Array<f32>>,
    pub biases: Vec<Array<f32>>,
}

impl fmt::Debug for BranchParams {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.param_vec())
    }
}

impl BranchParams {
    pub fn from_param_vec(
        param_vec: &[f32],
        layer_widths: &Vec<usize>,
        num_markers: usize,
    ) -> Self {
        let mut weights: Vec<Array<f32>> = vec![];
        let mut biases: Vec<Array<f32>> = vec![];
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

    pub fn load_param_vec(
        &mut self,
        param_vec: &[f32],
        layer_widths: &Vec<usize>,
        num_markers: usize,
    ) {
        let mut prev_width = num_markers;
        let mut read_ix: usize = 0;
        for (lix, width) in layer_widths.iter().enumerate() {
            let num_weights = prev_width * width;
            self.weights[lix] = Array::new(
                &param_vec[read_ix..read_ix + num_weights],
                dim4!(prev_width as u64, *width as u64, 1, 1),
            );
            prev_width = *width;
            read_ix += num_weights;
        }
        for (lix, width) in layer_widths[..layer_widths.len() - 1].iter().enumerate() {
            let num_biases = width;
            self.biases[lix] = Array::new(
                &param_vec[read_ix..read_ix + num_biases],
                dim4!(1, *width as u64, 1, 1),
            );
            read_ix += num_biases;
        }
    }

    pub fn param_vec(&self) -> Vec<f32> {
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

    pub fn full_step<T>(&mut self, step_sizes: &StepSizes, mom: &T)
    where
        T: Momentum,
    {
        for i in 0..self.weights.len() {
            self.weights[i] += &step_sizes.wrt_weights[i] * mom.wrt_layer_weights(i);
        }
        for i in 0..self.biases.len() {
            self.biases[i] += &step_sizes.wrt_biases[i] * mom.wrt_layer_biases(i);
        }
    }

    pub fn descent_gradient(&mut self, step_size: f32, gradient: &BranchLogDensityGradient) {
        for i in 0..self.weights.len() {
            self.weights[i] += step_size * &gradient.wrt_weights[i];
        }
        for i in 0..self.biases.len() {
            self.biases[i] += step_size * &gradient.wrt_biases[i];
        }
    }

    pub fn layer_weights(&self, index: usize) -> &Array<f32> {
        &self.weights[index]
    }

    pub fn layer_biases(&self, index: usize) -> &Array<f32> {
        &self.biases[index]
    }

    pub fn layer_weights_mut(&mut self, index: usize) -> &mut Array<f32> {
        &mut self.weights[index]
    }

    pub fn layer_biases_mut(&mut self, index: usize) -> &mut Array<f32> {
        &mut self.biases[index]
    }
}

#[cfg(test)]
mod tests {
    use super::BranchParams;
    use crate::af_helpers::to_host;
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
