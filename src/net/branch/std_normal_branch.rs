use super::{
    branch::{Branch, BranchCfg, BranchLogDensityGradient},
    branch_cfg_builder::BranchCfgBuilder,
    params::{BranchHyperparams, BranchParams},
    step_sizes::StepSizes,
};
use crate::scalar_to_host;
use arrayfire::Array;
use rand::prelude::ThreadRng;
use rand::thread_rng;

pub struct StdNormalBranch {
    pub(crate) num_params: usize,
    pub(crate) params: BranchParams,
    pub(crate) num_markers: usize,
    pub(crate) hyperparams: BranchHyperparams,
    pub(crate) layer_widths: Vec<usize>,
    pub(crate) num_layers: usize,
    pub(crate) rng: ThreadRng,
}

impl Branch for StdNormalBranch {
    fn build_cfg(cfg_bld: BranchCfgBuilder) -> BranchCfg {
        cfg_bld.build_base()
    }

    /// Creates Branch on device with BranchCfg from host memory.
    fn from_cfg(cfg: &BranchCfg) -> Self {
        Self {
            num_params: cfg.num_params,
            num_markers: cfg.num_markers,
            num_layers: cfg.layer_widths.len(),
            layer_widths: cfg.layer_widths.clone(),
            hyperparams: cfg.hyperparams.clone(),
            params: BranchParams::from_param_vec(&cfg.params, &cfg.layer_widths, cfg.num_markers),
            rng: thread_rng(),
        }
    }

    /// Dumps all branch info into a BranchCfg object stored in host memory.
    fn to_cfg(&self) -> BranchCfg {
        BranchCfg {
            num_params: self.num_params,
            num_markers: self.num_markers,
            layer_widths: self.layer_widths.clone(),
            params: self.params.param_vec(),
            hyperparams: self.hyperparams.clone(),
        }
    }

    fn hyperparams(&self) -> &BranchHyperparams {
        &self.hyperparams
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

    fn layer_widths(&self, index: usize) -> usize {
        self.layer_widths[index]
    }

    fn set_error_precision(&mut self, val: f64) {
        self.hyperparams.error_precision = val;
    }

    fn std_scaled_step_sizes(&self, const_factor: f64) -> StepSizes {
        let mut wrt_weights = Vec::with_capacity(self.num_layers());
        let mut wrt_biases = Vec::with_capacity(self.num_layers() - 1);

        for index in 0..self.num_layers() {
            wrt_weights.push(Array::new(
                &vec![
                    const_factor * (1. / scalar_to_host(&self.weight_precisions(index))).sqrt();
                    self.weights(index).elements()
                ],
                self.weights(index).dims(),
            ));
        }
        for index in 0..self.num_layers() - 1 {
            wrt_biases.push(Array::new(
                &vec![
                    const_factor * (1. / self.bias_precision(index)).sqrt();
                    self.biases(index).elements()
                ],
                self.biases(index).dims(),
            ));
        }

        StepSizes {
            wrt_weights,
            wrt_biases,
        }
    }

    fn log_density(&self, params: &BranchParams, hyperparams: &BranchHyperparams, rss: f64) -> f64 {
        let mut log_density: f64 = -0.5 * hyperparams.error_precision * rss;
        for i in 0..self.num_layers() {
            log_density -= 0.5 * arrayfire::sum_all(&(&(params.weights(i) * params.weights(i)))).0;
        }
        for i in 0..self.num_layers() - 1 {
            log_density -= 0.5 * arrayfire::sum_all(&(params.biases(i) * params.biases(i))).0;
        }
        log_density
    }

    fn log_density_gradient(
        &self,
        x_train: &Array<f64>,
        y_train: &Array<f64>,
    ) -> BranchLogDensityGradient {
        let (d_rss_wrt_weights, d_rss_wrt_biases) = self.backpropagate(x_train, y_train);
        let mut ldg_wrt_weights: Vec<Array<f64>> = Vec::with_capacity(self.num_layers);
        let mut ldg_wrt_biases: Vec<Array<f64>> = Vec::with_capacity(self.num_layers - 1);
        for layer_index in 0..self.num_layers() {
            ldg_wrt_weights.push(
                -(self.error_precision() * &d_rss_wrt_weights[layer_index]
                    + self.weights(layer_index)),
            );
        }
        for layer_index in 0..self.num_layers() - 1 {
            ldg_wrt_biases.push(-1. * self.biases(layer_index) - &d_rss_wrt_biases[layer_index]);
        }

        BranchLogDensityGradient {
            wrt_weights: ldg_wrt_weights,
            wrt_biases: ldg_wrt_biases,
        }
    }

    /// Samples precision values from their posterior distribution in a Gibbs step.
    fn sample_precisions(&mut self, _prior_shape: f64, _prior_scale: f64) {}
}
