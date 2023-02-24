use crate::net::{
    activation_functions::ActivationFunction,
    params::{BranchParamsHost, BranchPrecisionsHost, GlobalParams, OutputWeightSummaryStatsHost},
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct BranchCfg {
    pub(crate) num_params: usize,
    pub(crate) num_weights: usize,
    pub(crate) num_markers: usize,
    pub(crate) layer_widths: Vec<usize>,
    pub(crate) params: BranchParamsHost,
    pub(crate) precisions: BranchPrecisionsHost,
    pub(crate) activation_function: ActivationFunction,
}

impl BranchCfg {
    pub fn params(&self) -> &BranchParamsHost {
        &self.params
    }

    pub fn output_layer_weights(&self) -> &[f32] {
        self.params.weights.last().unwrap()
    }

    pub fn precisions(&self) -> &BranchPrecisionsHost {
        &self.precisions
    }

    pub fn set_output_layer_precision(&mut self, precision: f32) {
        self.precisions.set_output_layer_precision(precision);
    }

    pub fn set_error_precision(&mut self, precision: f32) {
        self.precisions.set_error_precision(precision);
    }

    pub fn output_weight_summary_stats_mut(&mut self) -> &mut OutputWeightSummaryStatsHost {
        &mut self.params.output_weight_summary_stats
    }

    pub fn set_output_weight_summary_stats(&mut self, sstats: OutputWeightSummaryStatsHost) {
        *self.output_weight_summary_stats_mut() = sstats;
    }

    pub fn output_layer_precision(&self) -> f32 {
        self.precisions.output_layer_precision()
    }

    pub fn output_weight_summary_stats(&self) -> OutputWeightSummaryStatsHost {
        self.params.output_weight_summary_stats
    }

    pub fn error_precision(&self) -> f32 {
        self.precisions.error_precision[0]
    }

    pub fn update_global_params(&mut self, gp: &GlobalParams) {
        self.set_error_precision(gp.error_precision());
        self.set_output_layer_precision(gp.output_layer_precision());
        self.set_output_weight_summary_stats(gp.output_weight_summary_stats());
    }

    pub fn perturb_params(&mut self, by: f32) {
        self.params.perturb(by)
    }

    pub fn perturb_precisions(&mut self, by: f32) {
        self.precisions.perturb(by)
    }
}
