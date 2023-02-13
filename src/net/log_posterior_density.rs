use arrayfire::Array;

use crate::af_helpers;

use super::params::NetworkPrecisionHyperparameters;

/// The log posterior density of a network model.
pub struct LogPosteriorDensity {
    /// single term for all branches
    wrt_rss_and_error_precision: f32,
    /// one term per branch
    wrt_params_and_param_precisions: Vec<f32>,
}

impl LogPosteriorDensity {
    fn new(num_branches: usize) -> Self {
        Self {
            wrt_rss_and_error_precision: f32::NEG_INFINITY,
            wrt_params_and_param_precisions: vec![f32::NEG_INFINITY; num_branches],
        }
    }

    fn update_rss_and_error_precision_term(
        &mut self,
        residual: &Array<f32>,
        error_precision: f32,
        hyperparams: &NetworkPrecisionHyperparameters,
    ) {
        let rss = af_helpers::sum_of_squares(residual);
        let num_individuals = residual.elements();
        self.wrt_rss_and_error_precision = error_precision.ln()
            * (hyperparams.output_layer_prior_shape() + (num_individuals as f32 - 2.0) / 2.0)
            - error_precision * (rss / 2.0 + 1.0 / hyperparams.output_layer_prior_scale());
    }

    fn update_params_and_param_precisions_term(&mut self, branch_ix: usize, log_density: f32) {
        self.wrt_params_and_param_precisions[branch_ix] = log_density;
    }

    fn log_density(&self) -> f32 {
        self.wrt_rss_and_error_precision + self.wrt_params_and_param_precisions.iter().sum::<f32>()
    }
}
