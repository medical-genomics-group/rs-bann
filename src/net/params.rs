use super::branch::gradient::{NetParamGradient, NetPrecisionGradient};
use super::branch::momentum::BranchMomentumJoint;
use super::branch::step_sizes::StepSizes;
use super::branch::{branch::BranchCfg, momentum::Momentum};
use crate::af_helpers::{af_scalar, scalar_to_host, to_host};
use arrayfire::{dim4, Array};
use rand::Rng;
use rand_distr::Distribution;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Copy, Deserialize, Serialize)]
pub struct GlobalParams {
    pub error_precision: f32,
    pub output_layer_precision: f32,
    pub output_weight_summary_stats: OutputWeightSummaryStatsHost,
}

impl GlobalParams {
    pub fn error_precision(&self) -> f32 {
        self.error_precision
    }

    pub fn output_layer_precision(self) -> f32 {
        self.output_layer_precision
    }

    pub fn output_weight_summary_stats(&self) -> OutputWeightSummaryStatsHost {
        self.output_weight_summary_stats
    }

    pub fn set_error_precision(&mut self, val: f32) {
        self.error_precision = val;
    }

    pub fn set_output_layer_precision(&mut self, val: f32) {
        self.output_layer_precision = val;
    }

    pub fn update_from_branch_cfg(&mut self, cfg: &BranchCfg) {
        self.error_precision = cfg.error_precision();
        self.output_layer_precision = cfg.output_layer_precision();
        self.output_weight_summary_stats = cfg.output_weight_summary_stats();
    }

    // fn update_from_branch(&mut self, branch: &impl Branch) {
    //     self.error_precision = scalar_to_host(branch.error_precision());
    //     self.output_layer_precision = scalar_to_host(branch.output_layer_precision());
    //     self.output_weight_summary_stats = branch.out
    // }
}

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
    pub fn set_error_precision(&mut self, precision: f32) {
        self.error_precision[0] = precision;
    }

    pub fn set_output_layer_precision(&mut self, precision: f32) {
        *self
            .weight_precisions
            .last_mut()
            .expect("Branch weight precisions is empty!") = vec![precision];
    }

    pub fn output_layer_precision(&self) -> f32 {
        self.weight_precisions
            .last()
            .expect("Branch weight precisions is empty!")[0]
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
            weight_precisions: self.weight_precisions.iter().map(to_host).collect(),
            bias_precisions: self.bias_precisions.iter().map(to_host).collect(),
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

    pub fn output_layer_precision(&self) -> &Array<f32> {
        self.weight_precisions
            .last()
            .expect("Branch weight precisions is empty!")
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

    pub fn descend_gradient<T: NetPrecisionGradient>(&mut self, step_size: f32, gradient: &T) {
        let gwwp = gradient.wrt_weight_precisions();
        for i in 0..self.weight_precisions.len() {
            self.weight_precisions[i] += step_size * &gwwp[i];
        }
        let gwbp = gradient.wrt_bias_precisions();
        for i in 0..self.bias_precisions.len() {
            self.bias_precisions[i] += step_size * &gwbp[i];
        }
        self.error_precision += step_size * gradient.wrt_error_precision();
    }
}

#[derive(Clone, Deserialize, Debug, PartialEq, Serialize, Copy)]
pub struct OutputWeightSummaryStatsHost {
    reg_sum: f32,
    num_params: usize,
}

impl Default for OutputWeightSummaryStatsHost {
    fn default() -> Self {
        Self {
            reg_sum: 0.0,
            num_params: 0,
        }
    }
}

impl OutputWeightSummaryStatsHost {
    pub fn incr_num_params(&mut self, num: usize) {
        self.num_params += num;
    }

    pub fn incr_reg_sum(&mut self, by: f32) {
        self.reg_sum += by;
    }
}

#[derive(Clone)]
pub struct OutputWeightSummaryStats {
    reg_sum: Array<f32>,
    num_params: Array<f32>,
}

impl Default for OutputWeightSummaryStats {
    fn default() -> Self {
        Self {
            reg_sum: af_scalar(0.0),
            num_params: af_scalar(0.0),
        }
    }
}

impl OutputWeightSummaryStats {
    pub fn new_single_branch(num_output_weights: usize) -> Self {
        Self {
            reg_sum: af_scalar(0.0),
            num_params: af_scalar(num_output_weights as f32),
        }
    }
}

impl OutputWeightSummaryStats {
    pub fn from_host(host: OutputWeightSummaryStatsHost) -> Self {
        Self {
            reg_sum: af_scalar(host.reg_sum),
            num_params: af_scalar(host.num_params as f32),
        }
    }

    pub fn to_host(&self) -> OutputWeightSummaryStatsHost {
        OutputWeightSummaryStatsHost {
            reg_sum: scalar_to_host(&self.reg_sum),
            num_params: scalar_to_host(&self.num_params) as usize,
        }
    }

    pub fn num_params(&self) -> &Array<f32> {
        &self.num_params
    }

    pub fn reg_sum(&self) -> &Array<f32> {
        &self.reg_sum
    }

    pub fn num_params_host(&self) -> usize {
        scalar_to_host(&self.num_params) as usize
    }

    pub fn reg_sum_host(&self) -> f32 {
        scalar_to_host(&self.reg_sum)
    }
}

#[derive(Clone, Deserialize, Debug, PartialEq, Serialize)]
pub struct BranchParamsHost {
    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<Vec<f32>>,
    pub layer_widths: Vec<usize>,
    pub num_markers: usize,
    /// Sum of abs or squares of output weights of other branches,
    /// needed for correct precision updates in joint sampling methods.
    pub output_weight_summary_stats: OutputWeightSummaryStatsHost,
}

impl BranchParamsHost {
    pub fn new(layer_widths: Vec<usize>, num_markers: usize) -> Self {
        let mut weights = Vec::new();
        let mut prev_width = num_markers;
        for curr_width in &layer_widths {
            weights.push(vec![0.0; prev_width * curr_width]);

            prev_width = *curr_width;
        }
        let biases = layer_widths
            .iter()
            .take(layer_widths.len() - 1)
            .map(|l| vec![0.0; *l])
            .collect();

        Self {
            weights,
            biases,
            layer_widths,
            num_markers,
            output_weight_summary_stats: OutputWeightSummaryStatsHost::default(),
        }
    }

    pub fn num_layers(&self) -> usize {
        self.layer_widths.len()
    }

    pub fn set_layer_weights(&mut self, layer_ix: usize, weights: Vec<f32>) {
        self.weights[layer_ix] = weights;
    }

    pub fn set_layer_biases(&mut self, layer_ix: usize, weights: Vec<f32>) {
        self.weights[layer_ix] = weights;
    }

    pub fn set_all_weights_to_constant(&mut self, value: f32) {
        for w in &mut self.weights {
            for v in w {
                *v = value;
            }
        }
    }

    pub fn set_all_biases_to_constant(&mut self, value: f32) {
        for b in &mut self.biases {
            for v in b {
                *v = value;
            }
        }
    }

    pub fn set_layer_weights_from_distribution(
        &mut self,
        layer_ix: usize,
        dist: &impl Distribution<f32>,
        rng: &mut impl Rng,
    ) {
        for v in &mut self.weights[layer_ix] {
            *v = dist.sample(rng);
        }
    }

    pub fn set_layer_biases_from_distribution(
        &mut self,
        layer_ix: usize,
        dist: &impl Distribution<f32>,
        rng: &mut impl Rng,
    ) {
        for v in &mut self.biases[layer_ix] {
            *v = dist.sample(rng);
        }
    }

    pub fn set_all_weights_from_distribution(
        &mut self,
        dist: &impl Distribution<f32>,
        rng: &mut impl Rng,
    ) {
        for layer_ix in 0..self.num_layers() {
            self.set_layer_weights_from_distribution(layer_ix, dist, rng);
        }
    }

    pub fn set_all_biases_from_distribution(
        &mut self,
        dist: &impl Distribution<f32>,
        rng: &mut impl Rng,
    ) {
        for layer_ix in 0..self.num_layers() - 1 {
            self.set_layer_biases_from_distribution(layer_ix, dist, rng);
        }
    }

    pub fn num_weights_in_layer(&self, layer_ix: usize) -> usize {
        self.weights[layer_ix].len()
    }

    pub fn num_biases_in_layer(&self, layer_ix: usize) -> usize {
        self.biases[layer_ix].len()
    }

    pub fn set_marker_effects_to_zero(&mut self, marker_ix: usize) {
        self.weights[0]
            .iter_mut()
            .skip(marker_ix)
            .step_by(self.num_markers)
            .for_each(|v| *v = 0.0);
    }

    pub fn ard_group(&self, layer_ix: usize, ard_group_ix: usize) -> Vec<f32> {
        let num_ard_groups = if layer_ix == 0 {
            self.num_markers
        } else {
            self.layer_widths[layer_ix - 1]
        };
        self.weights[layer_ix]
            .iter()
            .skip(ard_group_ix)
            .step_by(num_ard_groups)
            .cloned()
            .collect()
    }
}

/// Weights and biases
#[derive(Clone)]
pub struct BranchParams {
    pub weights: Vec<Array<f32>>,
    pub biases: Vec<Array<f32>>,
    pub layer_widths: Vec<usize>,
    pub num_markers: usize,
    pub output_weight_summary_stats: OutputWeightSummaryStats,
}

impl fmt::Debug for BranchParams {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.param_vec())
    }
}

impl BranchParams {
    pub fn from_host(host: &BranchParamsHost) -> Self {
        Self {
            weights: host
                .weights
                .iter()
                .enumerate()
                .map(|(ix, v)| {
                    let nrow = if ix == 0 {
                        host.num_markers
                    } else {
                        host.layer_widths[ix - 1]
                    };
                    let ncol = host.layer_widths[ix];
                    Array::new(v, dim4!(nrow as u64, ncol as u64))
                })
                .collect(),
            biases: host
                .biases
                .iter()
                .map(|v| Array::new(v, dim4!(1, v.len() as u64)))
                .collect(),
            layer_widths: host.layer_widths.clone(),
            num_markers: host.num_markers,
            output_weight_summary_stats: OutputWeightSummaryStats::from_host(
                host.output_weight_summary_stats,
            ),
        }
    }

    pub fn to_host(&self) -> BranchParamsHost {
        BranchParamsHost {
            weights: self.weights.iter().map(to_host).collect(),
            biases: self.biases.iter().map(to_host).collect(),
            layer_widths: self.layer_widths.clone(),
            num_markers: self.num_markers,
            output_weight_summary_stats: self.output_weight_summary_stats.to_host(),
        }
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

    pub fn descend_gradient<T: NetParamGradient>(&mut self, step_size: f32, gradient: &T) {
        let gww = gradient.wrt_weights();
        for i in 0..self.weights.len() {
            self.weights[i] += step_size * &gww[i];
        }
        let gwb = gradient.wrt_biases();
        for i in 0..self.biases.len() {
            self.biases[i] += step_size * &gwb[i];
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
    use arrayfire::{dim4, Array};

    fn make_test_params() -> BranchParams {
        BranchParams {
            weights: vec![
                Array::new(&[0.1, 0.2], dim4![2, 1, 1, 1]),
                Array::new(&[0.3], dim4![1, 1, 1, 1]),
            ],
            biases: vec![Array::new(&[0.4], dim4![1, 1, 1, 1])],
            layer_widths: vec![1, 1],
            num_markers: 2,
            output_weight_summary_stats: super::OutputWeightSummaryStats::default(),
        }
    }

    #[test]
    fn param_vec() {
        let params = make_test_params();
        let exp = vec![0.1, 0.2, 0.3, 0.4];
        assert_eq!(params.param_vec(), exp);
    }
}
