use super::{super::mcmc_cfg::MCMCCfg, params::BranchHyperparams};
use arrayfire::Array;

pub trait Branch {
    fn from_cfg(cfg: &BranchCfg) -> Self;
    fn to_cfg(&self) -> BranchCfg;
    fn num_params(&self) -> usize;
    fn num_markers(&self) -> usize;
    fn num_layers(&self) -> usize;
    fn weights(&self, index: usize) -> &Array<f64>;
    fn biases(&self, index: usize) -> &Array<f64>;
    fn weight_precision(&self, index: usize) -> f64;
    fn bias_precision(&self, index: usize) -> f64;
    fn error_precision(&self) -> f64;
    fn layer_widths(&self, index: usize) -> usize;
    fn set_error_precision(&mut self, val: f64);
    fn rss(&self, x: &Array<f64>, y: &Array<f64>) -> f64;
    fn predict(&self, x: &Array<f64>) -> Array<f64>;
    fn sample_precisions(&mut self, prior_shape: f64, prior_scale: f64);
    fn hmc_step(
        &mut self,
        x_train: &Array<f64>,
        y_train: &Array<f64>,
        mcmc_cfg: &MCMCCfg,
    ) -> HMCStepResult;
}

#[derive(Clone)]
pub struct BranchCfg {
    pub(crate) num_params: usize,
    pub(crate) num_markers: usize,
    pub(crate) layer_widths: Vec<usize>,
    pub(crate) params: Vec<f64>,
    pub(crate) hyperparams: BranchHyperparams,
}

impl BranchCfg {
    pub fn params(&self) -> &Vec<f64> {
        &self.params
    }
}

pub enum HMCStepResult {
    RejectedEarly,
    Rejected,
    Accepted,
}
