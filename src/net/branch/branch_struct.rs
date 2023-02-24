use crate::net::{
    activation_functions::ActivationFunction,
    branch::{branch_cfg::BranchCfg, training_state::TrainingState},
    params::{BranchParams, BranchPrecisions, OutputWeightSummaryStats},
};
use rand::rngs::ThreadRng;

macro_rules! branch_struct {
    ($t:ident) => {
        impl $crate::net::branch::branch_struct::BranchStruct for $t {
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
                    rng: $crate::rand::thread_rng(),
                    training_state: TrainingState::default(),
                    activation_function: cfg.activation_function,
                };

                res.subtract_output_weight_summary_stat_from_global();

                res
            }

            fn rng_mut(&mut self) -> &mut ThreadRng {
                &mut self.rng
            }

            fn output_weight_summary_stats(
                &self,
            ) -> &$crate::net::params::OutputWeightSummaryStats {
                &self.params.output_weight_summary_stats
            }

            fn output_weight_summary_stats_mut(
                &mut self,
            ) -> &mut $crate::net::params::OutputWeightSummaryStats {
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

            fn num_markers(&self) -> usize {
                self.num_markers
            }

            fn layer_widths(&self) -> &Vec<usize> {
                &self.layer_widths
            }

            fn precisions(&self) -> &BranchPrecisions {
                &self.precisions
            }

            fn precisions_mut(&mut self) -> &mut BranchPrecisions {
                &mut self.precisions
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

            fn set_precisions(&mut self, precisions: &BranchPrecisions) {
                self.precisions = precisions.clone();
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

            fn layer_width(&self, index: usize) -> usize {
                self.layer_widths[index]
            }

            fn set_error_precision(&mut self, val: f32) {
                self.precisions.error_precision = af_scalar(val);
            }

            fn activation_function(&self) -> ActivationFunction {
                self.activation_function
            }
        }
    };
}
pub(crate) use branch_struct;

pub trait BranchStruct {
    fn from_cfg(cfg: &BranchCfg) -> Self;
    fn rng_mut(&mut self) -> &mut ThreadRng;
    fn output_weight_summary_stats(&self) -> &OutputWeightSummaryStats;
    fn output_weight_summary_stats_mut(&mut self) -> &mut OutputWeightSummaryStats;
    fn training_state(&self) -> &TrainingState;
    fn training_state_mut(&mut self) -> &mut TrainingState;
    fn num_weights(&self) -> usize;
    fn num_markers(&self) -> usize;
    fn layer_widths(&self) -> &Vec<usize>;
    fn precisions(&self) -> &BranchPrecisions;
    fn precisions_mut(&mut self) -> &mut BranchPrecisions;
    fn num_layers(&self) -> usize;
    fn rng(&mut self) -> &mut ThreadRng;
    fn params_mut(&mut self) -> &mut BranchParams;
    fn set_precisions(&mut self, precisions: &BranchPrecisions);
    fn set_params(&mut self, params: &BranchParams);
    fn params(&self) -> &BranchParams;
    fn num_params(&self) -> usize;
    fn layer_width(&self, index: usize) -> usize;
    fn set_error_precision(&mut self, val: f32);
    fn activation_function(&self) -> ActivationFunction;
}
