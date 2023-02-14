use super::{
    super::model_type::ModelType,
    super::params::{BranchParams, BranchPrecisions, OutputWeightSummaryStats},
    branch::{Branch, BranchCfg},
    branch_cfg_builder::BranchCfgBuilder,
    step_sizes::StepSizes,
    training_state::TrainingState,
};
use crate::af_helpers::{af_scalar, scalar_to_host, sum_of_squares};
use crate::net::mcmc_cfg::MCMCCfg;
use crate::net::params::NetworkPrecisionHyperparameters;
use arrayfire::{sqrt, Array};
use rand::prelude::ThreadRng;
use rand::thread_rng;

pub struct StdNormalBranch {
    pub(crate) num_params: usize,
    pub(crate) num_weights: usize,
    pub(crate) params: BranchParams,
    pub(crate) num_markers: usize,
    pub(crate) precisions: BranchPrecisions,
    pub(crate) layer_widths: Vec<usize>,
    pub(crate) num_layers: usize,
    pub(crate) rng: ThreadRng,
    pub(crate) training_state: TrainingState,
}

impl Branch for StdNormalBranch {
    fn summary_stat_fn_host(_: &[f32]) -> f32 {
        1.0
    }

    fn summary_stat_fn(&self, vals: &Array<f32>) -> Array<f32> {
        af_scalar(sum_of_squares(vals))
    }

    fn model_type() -> ModelType {
        ModelType::StdNormal
    }

    fn build_cfg(cfg_bld: BranchCfgBuilder) -> BranchCfg {
        cfg_bld.build_base()
    }

    /// Creates Branch on device with BranchCfg from host memory.
    fn from_cfg(cfg: &BranchCfg) -> Self {
        let mut res = Self {
            num_params: cfg.num_params,
            num_weights: cfg.num_weights,
            num_markers: cfg.num_markers,
            num_layers: cfg.layer_widths.len(),
            layer_widths: cfg.layer_widths.clone(),
            precisions: BranchPrecisions::from_host(&cfg.precisions),
            params: BranchParams::from_host(&cfg.params),
            rng: thread_rng(),
            training_state: TrainingState::default(),
        };

        res.subtract_output_weight_summary_stat_from_global();

        res
    }

    fn rng_mut(&mut self) -> &mut ThreadRng {
        &mut self.rng
    }

    fn output_weight_summary_stats(&self) -> &OutputWeightSummaryStats {
        &self.params.output_weight_summary_stats
    }

    fn output_weight_summary_stats_mut(&mut self) -> &mut OutputWeightSummaryStats {
        &mut self.params.output_weight_summary_stats
    }

    fn training_state(&self) -> &TrainingState {
        &self.training_state
    }

    fn training_state_mut(&mut self) -> &mut TrainingState {
        &mut self.training_state
    }

    fn num_weights(&self) -> usize {
        self.num_weights
    }

    fn precisions(&self) -> &BranchPrecisions {
        &self.precisions
    }

    fn precisions_mut(&mut self) -> &mut BranchPrecisions {
        &mut self.precisions
    }

    fn set_precisions(&mut self, precisions: &BranchPrecisions) {
        self.precisions = precisions.clone();
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn rng(&mut self) -> &mut ThreadRng {
        &mut self.rng
    }

    fn params_mut(&mut self) -> &mut BranchParams {
        &mut self.params
    }

    fn set_params(&mut self, params: &BranchParams) {
        self.params = params.clone();
    }

    fn params(&self) -> &BranchParams {
        &self.params
    }

    fn num_params(&self) -> usize {
        self.num_params
    }

    fn num_markers(&self) -> usize {
        self.num_markers
    }

    fn layer_widths(&self) -> &Vec<usize> {
        &self.layer_widths
    }

    fn layer_width(&self, index: usize) -> usize {
        self.layer_widths[index]
    }

    fn set_error_precision(&mut self, val: f32) {
        self.precisions.error_precision = af_scalar(val);
    }

    fn std_scaled_step_sizes(&self, mcmc_cfg: &MCMCCfg) -> StepSizes {
        let const_factor = mcmc_cfg.hmc_step_size_factor;
        let mut wrt_weights = Vec::with_capacity(self.num_layers());
        let mut wrt_biases = Vec::with_capacity(self.num_layers() - 1);

        for index in 0..self.num_layers() {
            wrt_weights.push(Array::new(
                &vec![
                    const_factor * (1. / scalar_to_host(self.weight_precisions(index))).sqrt();
                    self.layer_weights(index).elements()
                ],
                self.layer_weights(index).dims(),
            ));
        }
        for index in 0..self.num_layers() - 1 {
            wrt_biases.push(
                arrayfire::constant(1.0f32, self.layer_biases(index).dims())
                    * (const_factor * (1.0f32 / arrayfire::sqrt(self.bias_precision(index)))),
            );
        }

        StepSizes {
            wrt_weights,
            wrt_biases,
            wrt_weight_precisions: None,
            wrt_bias_precisions: None,
            wrt_error_precision: None,
        }
    }

    fn izmailov_step_sizes(&mut self, mcmc_cfg: &MCMCCfg) -> StepSizes {
        let integration_length = mcmc_cfg.hmc_integration_length;
        let mut wrt_weights: Vec<Array<f32>> = Vec::with_capacity(self.num_layers());
        let mut wrt_biases = Vec::with_capacity(self.num_layers() - 1);

        for index in 0..self.num_layers() {
            wrt_weights.push(
                std::f32::consts::PI
                    / (2f32
                        * sqrt(&self.precisions().weight_precisions[index])
                        * integration_length as f32),
            );
        }

        for index in 0..self.num_layers() - 1 {
            wrt_biases.push(
                arrayfire::constant(1.0f32, self.layer_biases(index).dims())
                    * (std::f32::consts::PI
                        / (2.0f32
                            * arrayfire::sqrt(&self.precisions().bias_precisions[index])
                            * integration_length as f32)),
            );
        }

        StepSizes {
            wrt_weights,
            wrt_biases,
            wrt_weight_precisions: None,
            wrt_bias_precisions: None,
            wrt_error_precision: None,
        }
    }

    fn log_density_joint_wrt_weights(
        &self,
        _params: &BranchParams,
        _precisions: &BranchPrecisions,
        _hyperparams: &NetworkPrecisionHyperparameters,
    ) -> Array<f32> {
        unimplemented!("Joint sampling is not implemented for std normal priors, since the precisions are fixed to 1.0");
    }

    fn log_density_wrt_weights(
        &self,
        params: &BranchParams,
        _precisions: &BranchPrecisions,
    ) -> Array<f32> {
        let mut log_density: Array<f32> = af_scalar(0.0);

        // weight terms
        for i in 0..self.num_layers() {
            log_density -= af_scalar(sum_of_squares(params.layer_weights(i)) / 2.0);
        }

        log_density
    }

    fn log_density(&self, params: &BranchParams, precisions: &BranchPrecisions, rss: f32) -> f32 {
        let mut log_density: f32 = scalar_to_host(&(-0.5f32 * &precisions.error_precision * rss));
        for i in 0..self.num_layers() {
            log_density -=
                0.5 * arrayfire::sum_all(&(params.layer_weights(i) * params.layer_weights(i))).0;
        }
        for i in 0..self.num_layers() - 1 {
            log_density -=
                0.5 * arrayfire::sum_all(&(params.layer_biases(i) * params.layer_biases(i))).0;
        }
        log_density
    }

    fn log_density_gradient_wrt_weights(&self) -> Vec<Array<f32>> {
        let mut ldg_wrt_weights: Vec<Array<f32>> = Vec::with_capacity(self.num_layers);
        for layer_index in 0..self.num_layers() {
            ldg_wrt_weights.push(
                -(self.error_precision() * self.layer_d_rss_wrt_weights(layer_index)
                    + self.layer_weights(layer_index)),
            );
        }
        ldg_wrt_weights
    }

    fn log_density_gradient_wrt_weight_precisions(
        &self,
        _hyperparams: &NetworkPrecisionHyperparameters,
    ) -> Vec<Array<f32>> {
        unimplemented!("Joint sampling is not implemented for std normal priors, since the precisions are fixed to 1.0");
    }

    fn precision_posterior_host(
        &mut self,
        _prior_shape: f32,
        _prior_scale: f32,
        _summary_stat: f32,
        _num_vals: usize,
    ) -> f32 {
        1.0
    }

    /// Samples precision values from their posterior distribution in a Gibbs step.
    fn sample_prior_precisions(&mut self, __hyperparams: &NetworkPrecisionHyperparameters) {}
}
