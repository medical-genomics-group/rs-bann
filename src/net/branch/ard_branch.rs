use super::{
    super::gibbs_steps::multi_param_precision_posterior,
    branch::{Branch, BranchCfg, BranchLogDensityGradient},
    params::{BranchHyperparams, BranchParams},
    step_sizes::StepSizes,
};
use crate::to_host;
use arrayfire::{matmul, sum, sum_all, Array, MatProp};
use rand::prelude::ThreadRng;
use rand::thread_rng;
use rand_distr::{Distribution, Gamma};

pub struct ArdBranch {
    pub(crate) num_params: usize,
    pub(crate) num_markers: usize,
    pub(crate) params: BranchParams,
    pub(crate) hyperparams: BranchHyperparams,
    pub(crate) layer_widths: Vec<usize>,
    pub(crate) num_layers: usize,
    pub(crate) rng: ThreadRng,
}

// Weights in this branch are grouped by the node they
// are going out of.
impl Branch for ArdBranch {
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

    fn set_params(&mut self, params: &BranchParams) {
        self.params = params.clone();
    }

    fn params_mut(&mut self) -> &mut BranchParams {
        &mut self.params
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

    // TODO: actually make some step sizes here
    fn std_scaled_step_sizes(&self, const_factor: f64) -> StepSizes {
        let mut wrt_weights = Vec::with_capacity(self.num_layers());
        let mut wrt_biases = Vec::with_capacity(self.num_layers() - 1);

        StepSizes {
            wrt_weights,
            wrt_biases,
        }
    }

    fn log_density(&self, params: &BranchParams, hyperparams: &BranchHyperparams, rss: f64) -> f64 {
        let mut log_density: f64 = -0.5 * hyperparams.error_precision * rss;

        for i in 0..self.num_layers() {
            log_density -= 0.5
                * sum_all(&matmul(
                    &(params.weights(i) * params.weights(i)),
                    &self.weight_precisions(i),
                    MatProp::TRANS,
                    MatProp::NONE,
                ))
                .0;
        }
        for i in 0..self.num_layers() - 1 {
            log_density -= hyperparams.bias_precisions[i]
                * 0.5
                * sum_all(&(params.biases(i) * params.biases(i))).0;
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
                -self.weight_precisions(layer_index) * self.weights(layer_index)
                    - self.error_precision() * &d_rss_wrt_weights[layer_index],
            );
        }

        for layer_index in 0..self.num_layers - 1 {
            ldg_wrt_biases.push(
                -self.bias_precision(layer_index) * self.biases(layer_index)
                    - self.error_precision() * &d_rss_wrt_biases[layer_index],
            );
        }

        BranchLogDensityGradient {
            wrt_weights: ldg_wrt_weights,
            wrt_biases: ldg_wrt_biases,
        }
    }

    /// Samples precision values from their posterior distribution in a Gibbs step.
    fn sample_precisions(&mut self, prior_shape: f64, prior_scale: f64) {
        for i in 0..self.params.weights.len() {
            let posterior_shape = self.layer_widths(i) as f64 / 2. + prior_shape;
            // compute sums of squares of all rows
            self.hyperparams.weight_precisions[i] = Array::new(
                &to_host(&sum(&(self.params.weights[i] * self.params.weights[i]), 0))
                    .iter()
                    .map(|sum_squares| {
                        let posterior_scale = 2. * prior_scale / (2. + prior_scale * sum_squares);
                        Gamma::new(posterior_shape, posterior_scale)
                            .unwrap()
                            .sample(self.rng())
                    })
                    .collect::<Vec<f64>>(),
                self.hyperparams.weight_precisions[i].dims(),
            );
        }
        for i in 0..self.params.biases.len() {
            self.hyperparams.bias_precisions[i] = multi_param_precision_posterior(
                prior_shape,
                prior_scale,
                &self.params.biases[i],
                &mut self.rng,
            );
        }
    }
}
