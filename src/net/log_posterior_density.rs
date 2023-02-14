use crate::af_helpers;
use arrayfire::Array;
use serde::{Deserialize, Serialize};

use super::{branch::branch::Branch, params::NetworkPrecisionHyperparameters};

#[derive(Serialize, Deserialize)]
/// The log posterior density of a network model.
pub struct LogPosteriorDensity {
    /// single term for all branches
    wrt_rss_and_error_precision: f32,
    /// shared log density term
    wrt_output_weights_and_precision: f32,
    /// log density terms that are branch specific, one per branch
    wrt_local_params: Vec<f32>,
}

impl LogPosteriorDensity {
    pub fn new(num_branches: usize) -> Self {
        Self {
            wrt_rss_and_error_precision: f32::NEG_INFINITY,
            wrt_output_weights_and_precision: f32::NEG_INFINITY,
            wrt_local_params: vec![f32::NEG_INFINITY; num_branches],
        }
    }

    pub fn update_from_branch(
        &mut self,
        branch_ix: usize,
        branch: &impl Branch,
        residual: &Array<f32>,
        hyperparams: &NetworkPrecisionHyperparameters,
    ) {
        let (wrt_out_w, wrt_local) =
            branch.log_density_joint_components_curr_internal_state(hyperparams);

        log::debug!(
            "Received from branch: wrt_out_w: {:?}, wrt_local: {:?}",
            wrt_out_w,
            wrt_local
        );

        self.wrt_local_params[branch_ix] = wrt_local;
        self.wrt_output_weights_and_precision = wrt_out_w;
        self.update_rss_and_error_precision_term(residual, branch.error_precision(), hyperparams);
    }

    fn update_rss_and_error_precision_term(
        &mut self,
        residual: &Array<f32>,
        error_precision: &Array<f32>,
        hyperparams: &NetworkPrecisionHyperparameters,
    ) {
        let rss = af_helpers::sum_of_squares(residual);
        let num_individuals = residual.elements();
        self.wrt_rss_and_error_precision = crate::af_helpers::scalar_to_host(
            &(arrayfire::log(error_precision)
                * (hyperparams.output_layer_prior_shape() + (num_individuals as f32 - 2.0) / 2.0)
                - error_precision * (rss / 2.0 + 1.0 / hyperparams.output_layer_prior_scale())),
        );
    }

    pub fn lpd(&self) -> f32 {
        self.wrt_rss_and_error_precision
            + self.wrt_output_weights_and_precision
            + self.wrt_local_params.iter().sum::<f32>()
    }
}
