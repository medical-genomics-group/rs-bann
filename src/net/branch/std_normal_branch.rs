use super::{
    super::model_type::ModelType,
    super::params::{BranchParams, BranchPrecisions},
    branch_cfg::BranchCfg,
    branch_cfg_builder::BranchCfgBuilder,
    branch_sampler::BranchSampler,
    branch_struct::BranchStruct,
    step_sizes::StepSizes,
    training_state::TrainingState,
};
use crate::af_helpers::{af_scalar, scalar_to_host, sum_of_squares};
use crate::net::activation_functions::*;
use crate::net::mcmc_cfg::MCMCCfg;
use crate::net::params::NetworkPrecisionHyperparameters;
use arrayfire::{sqrt, Array};
use rand::prelude::ThreadRng;

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
    pub(crate) activation_function: ActivationFunction,
}

crate::net::activation_functions::has_activation_function!(StdNormalBranch);
super::branch_struct::branch_struct!(StdNormalBranch);

impl BranchSampler for StdNormalBranch {
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

    fn log_density_joint_wrt_local_weights(
        &self,
        _params: &BranchParams,
        _precisions: &BranchPrecisions,
        _hyperparams: &NetworkPrecisionHyperparameters,
    ) -> Array<f32> {
        unimplemented!("Joint sampling is not implemented for std normal priors, since the precisions are fixed to 1.0");
    }

    fn log_density_joint_wrt_output_weights(
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
