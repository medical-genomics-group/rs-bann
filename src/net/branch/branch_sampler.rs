use super::gradient::{BranchLogDensityGradient, BranchLogDensityGradientJoint};
use super::momentum::{BranchMomentumJoint, Momentum};
use super::{
    super::{
        mcmc_cfg::{MCMCCfg, StepSizeMode},
        model_type::ModelType,
        params::BranchParams,
        params::BranchPrecisions,
    },
    branch_cfg::BranchCfg,
    branch_cfg_builder::BranchCfgBuilder,
    branch_struct::BranchStruct,
    momentum::BranchMomentum,
    step_sizes::StepSizes,
    trajectory::Trajectory,
};
use crate::af_helpers::{add_at_ix, af_scalar, scalar_to_host, subtract_at_ix, sum_of_squares};
use crate::net::activation_functions::HasActivationFunction;
use crate::net::gibbs_steps::ridge_multi_param_precision_posterior;
use crate::net::params::NetworkPrecisionHyperparameters;
use arrayfire::{diag_extract, dim4, dot, matmul, randu, sum, Array, MatProp};
use log::{debug, warn};
use rand::Rng;
use serde_json::to_writer;
use std::{
    fs::File,
    io::{BufWriter, Write},
};

const NUMERICAL_DELTA: f32 = 0.001;

pub trait BranchSampler: HasActivationFunction + BranchStruct {
    fn model_type() -> ModelType;

    fn build_cfg(cfg_bld: BranchCfgBuilder) -> BranchCfg;

    /// Branch type specific function that computes the value sum summary statistic
    /// from an array of parameter values. E.g. sum of squares, sum of abs
    fn summary_stat_fn_host(vals: &[f32]) -> f32;

    /// Branch type specific function that computes the value sum summary statistic
    /// from an array of parameter values. E.g. sum of squares, sum of abs
    fn summary_stat_fn(&self, vals: &Array<f32>) -> Array<f32>;

    fn precision_posterior_host(
        &mut self,
        // k
        prior_shape: f32,
        // s or theta
        prior_scale: f32,
        summary_stat: f32,
        num_vals: usize,
    ) -> f32;

    fn sample_prior_precisions(
        &mut self,
        precision_prior_hyperparams: &NetworkPrecisionHyperparameters,
    );

    fn log_density_gradient_wrt_weight_precisions(
        &self,
        hyperparams: &NetworkPrecisionHyperparameters,
    ) -> Vec<Array<f32>>;

    fn log_density_gradient_wrt_weights(&self) -> Vec<Array<f32>>;

    fn output_layer_precision(&self) -> &Array<f32> {
        self.precisions().output_layer_precision()
    }

    // This should be -U(q), e.g. log P(D | Theta)P(Theta)
    fn log_density(&self, params: &BranchParams, precisions: &BranchPrecisions, rss: f32) -> f32 {
        let wrt_w = self.log_density_wrt_weights(params, precisions);
        let wrt_e = self.log_density_wrt_rss(precisions, rss);
        let wrt_b = self.log_density_wrt_biases(params, precisions);

        scalar_to_host(&(wrt_w + wrt_b + wrt_e))
    }

    fn log_density_wrt_weights(
        &self,
        params: &BranchParams,
        precisions: &BranchPrecisions,
    ) -> Array<f32>;

    fn log_density_wrt_rss(&self, precisions: &BranchPrecisions, rss: f32) -> Array<f32> {
        -1.0f32 * &precisions.error_precision * (rss / 2.0)
    }

    /// Log density w.r.t. l2 regularized biases
    fn log_density_wrt_biases(
        &self,
        params: &BranchParams,
        precisions: &BranchPrecisions,
    ) -> Array<f32> {
        let mut log_density: Array<f32> = af_scalar(0.0);

        for i in 0..self.output_layer_index() {
            log_density -=
                precisions.layer_bias_precision(i) * (sum_of_squares(params.layer_biases(i)) / 2.0);
        }

        log_density
    }

    fn last_rss(&self) -> &Array<f32> {
        self.training_state().rss()
    }

    fn set_last_rss(&mut self, rss: &Array<f32>) {
        self.training_state_mut().set_rss(rss);
    }

    fn d_rss_wrt_weights(&self) -> &Vec<Array<f32>> {
        self.training_state().d_rss_wrt_weights()
    }

    fn d_rss_wrt_biases(&self) -> &Vec<Array<f32>> {
        self.training_state().d_rss_wrt_biases()
    }

    fn layer_d_rss_wrt_weights(&self, layer_index: usize) -> &Array<f32> {
        &self.training_state().d_rss_wrt_weights()[layer_index]
    }

    fn layer_d_rss_wrt_biases(&self, layer_index: usize) -> &Array<f32> {
        &self.training_state().d_rss_wrt_biases()[layer_index]
    }

    /// Dumps all branch info into a BranchCfg object stored in host memory.
    fn to_cfg(&mut self) -> BranchCfg {
        self.add_output_weight_summary_stat_to_global();

        let res = BranchCfg {
            num_params: self.num_params(),
            num_weights: self.num_weights(),
            num_markers: self.num_markers(),
            layer_widths: self.layer_widths().clone(),
            params: self.params().to_host(),
            precisions: self.precisions().to_host(),
            activation_function: self.activation_function(),
        };

        self.subtract_output_weight_summary_stat_from_global();

        res
    }

    fn sample_param_precisions(&mut self, hyperparams: &NetworkPrecisionHyperparameters) {
        self.sample_prior_precisions(hyperparams);
        self.sample_output_weight_precisions(hyperparams);
    }

    fn sample_output_weight_precisions(&mut self, hyperparams: &NetworkPrecisionHyperparameters) {
        self.add_output_weight_summary_stat_to_global();
        let precision = self.precision_posterior_host(
            hyperparams.output_layer_prior_shape(),
            hyperparams.output_layer_prior_scale(),
            self.output_weight_summary_stats().reg_sum_host(),
            self.output_weight_summary_stats().num_params_host(),
        );
        self.subtract_output_weight_summary_stat_from_global();
        self.precisions_mut().set_output_layer_precision(precision);
    }

    fn sample_error_precision(
        &mut self,
        residual: &Array<f32>,
        hyperparams: &NetworkPrecisionHyperparameters,
    ) {
        let precision = ridge_multi_param_precision_posterior(
            hyperparams.output_layer_prior_shape(),
            hyperparams.output_layer_prior_scale(),
            &residual,
            self.rng_mut(),
        );
        self.set_error_precision(precision);
    }

    fn summary_layer_index(&self) -> usize {
        self.num_layers() - 2
    }

    fn output_layer_index(&self) -> usize {
        self.num_layers() - 1
    }

    /// Portion of log density attributable to weights local to the branch and their precisions
    fn log_density_joint_wrt_local_weights(
        &self,
        params: &BranchParams,
        precisions: &BranchPrecisions,
        hyperparams: &NetworkPrecisionHyperparameters,
    ) -> Array<f32>;

    /// Portion of log density attributable to output weights and their precision
    fn log_density_joint_wrt_output_weights(
        &self,
        params: &BranchParams,
        precisions: &BranchPrecisions,
        hyperparams: &NetworkPrecisionHyperparameters,
    ) -> Array<f32>;

    /// Portion of log density attributable to weights and their precisions
    fn log_density_joint_wrt_weights(
        &self,
        params: &BranchParams,
        precisions: &BranchPrecisions,
        hyperparams: &NetworkPrecisionHyperparameters,
    ) -> Array<f32> {
        self.log_density_joint_wrt_local_weights(params, precisions, hyperparams)
            + self.log_density_joint_wrt_output_weights(params, precisions, hyperparams)
    }

    /// Portion of log density attributable to rss and the error precision
    fn log_density_joint_wrt_rss(
        &self,
        precisions: &BranchPrecisions,
        rss: f32,
        hyperparams: &NetworkPrecisionHyperparameters,
        num_individuals: usize,
    ) -> Array<f32> {
        let mut log_density: Array<f32> = af_scalar(0.0);

        // rss / error precision terms
        log_density += (hyperparams.output_layer_prior_shape()
            + (num_individuals as f32 - 2.0) / 2.0)
            * arrayfire::log(&precisions.error_precision);
        log_density -= &precisions.error_precision
            * (rss / 2.0 + 1.0 / hyperparams.output_layer_prior_scale());

        log_density
    }

    /// Portion of log density attributable to l2 regularized biases and their precisions
    fn log_density_joint_wrt_biases(
        &self,
        params: &BranchParams,
        precisions: &BranchPrecisions,
        hyperparams: &NetworkPrecisionHyperparameters,
    ) -> Array<f32> {
        let mut log_density: Array<f32> = af_scalar(0.0);

        for i in 0..self.num_layers() - 1 {
            // w.r.t. biases
            let (shape, scale) = hyperparams.layer_prior_hyperparams(i, self.num_layers());
            log_density -= precisions.layer_bias_precision(i)
                * (sum_of_squares(params.layer_biases(i)) / 2.0 + 1.0 / scale);
            let nvar = params.layer_biases(i).elements();
            log_density += (shape + (nvar as f32 - 2.0f32) / 2.0)
                * arrayfire::log(precisions.layer_bias_precision(i));
        }

        log_density
    }

    fn log_density_joint_wrt_local_params(
        &self,
        hyperparams: &NetworkPrecisionHyperparameters,
    ) -> f32 {
        let wrt_b =
            self.log_density_joint_wrt_biases(self.params(), self.precisions(), hyperparams);
        let wrt_local_w =
            self.log_density_joint_wrt_local_weights(self.params(), self.precisions(), hyperparams);
        scalar_to_host(&(wrt_b + wrt_local_w))
    }

    fn log_density_joint(
        &self,
        params: &BranchParams,
        precisions: &BranchPrecisions,
        rss: f32,
        hyperparams: &NetworkPrecisionHyperparameters,
        num_individuals: usize,
    ) -> f32 {
        let wrt_w = self.log_density_joint_wrt_weights(params, precisions, hyperparams);
        let wrt_e = self.log_density_joint_wrt_rss(precisions, rss, hyperparams, num_individuals);
        let wrt_b = self.log_density_joint_wrt_biases(params, precisions, hyperparams);

        scalar_to_host(&(wrt_w + wrt_b + wrt_e))
    }

    fn log_density_joint_components_curr_internal_state(
        &self,
        hyperparams: &NetworkPrecisionHyperparameters,
    ) -> (f32, f32) {
        let wrt_output_w = self.log_density_joint_wrt_output_weights(
            self.params(),
            self.precisions(),
            hyperparams,
        );
        let wrt_local = self.log_density_joint_wrt_local_params(hyperparams);
        (scalar_to_host(&wrt_output_w), wrt_local)
    }

    /// Gradient w.r.t l2 regularized biases
    fn log_density_gradient_wrt_biases(&self) -> Vec<Array<f32>> {
        let mut ldg_wrt_biases: Vec<Array<f32>> = Vec::with_capacity(self.num_layers() - 1);

        for layer_index in 0..self.output_layer_index() {
            ldg_wrt_biases.push(
                -1.0f32 * self.bias_precision(layer_index) * self.layer_biases(layer_index)
                    - self.error_precision() * self.layer_d_rss_wrt_biases(layer_index),
            );
        }

        ldg_wrt_biases
    }

    /// Gradient w.r.t precision params of l2 regularized biases
    fn log_density_gradient_wrt_bias_precisions(
        &self,
        hyperparams: &NetworkPrecisionHyperparameters,
    ) -> Vec<Array<f32>> {
        let mut ldg_wrt_bias_precisions: Vec<Array<f32>> =
            Vec::with_capacity(self.num_layers() - 1);
        for layer_index in 0..self.num_layers() - 1 {
            let precision = self.bias_precision(layer_index);
            let params = self.layer_biases(layer_index);
            let nvar = params.elements();
            let (shape, scale) =
                hyperparams.layer_prior_hyperparams(layer_index, self.num_layers());
            ldg_wrt_bias_precisions.push(
                (2.0 * shape + (nvar as f32 - 2.0)) / (2.0f32 * precision)
                    - (1.0f32 / scale)
                    - sum_of_squares(params) / 2.0f32,
            );
        }
        ldg_wrt_bias_precisions
    }

    fn log_density_gradient_wrt_error_precision(
        &self,
        y_train: &Array<f32>,
        hyperparams: &NetworkPrecisionHyperparameters,
    ) -> Array<f32> {
        (2.0 * hyperparams.output_layer_prior_shape() + y_train.elements() as f32 - 2.0)
            / (2.0f32 * self.error_precision())
            - 1.0f32 / hyperparams.output_layer_prior_scale()
            - self.last_rss() / 2.0f32
    }

    fn log_density_gradient(
        &mut self,
        x_train: &Array<f32>,
        y_train: &Array<f32>,
    ) -> BranchLogDensityGradient {
        self.backpropagate(x_train, y_train);

        BranchLogDensityGradient {
            wrt_weights: self.log_density_gradient_wrt_weights(),
            wrt_biases: self.log_density_gradient_wrt_biases(),
        }
    }

    fn log_density_gradient_joint(
        &mut self,
        x_train: &Array<f32>,
        y_train: &Array<f32>,
        hyperparams: &NetworkPrecisionHyperparameters,
    ) -> BranchLogDensityGradientJoint {
        let ldg_wrt_params = self.log_density_gradient(x_train, y_train);

        BranchLogDensityGradientJoint {
            wrt_weights: ldg_wrt_params.wrt_weights,
            wrt_biases: ldg_wrt_params.wrt_biases,
            wrt_weight_precisions: self.log_density_gradient_wrt_weight_precisions(hyperparams),
            wrt_bias_precisions: self.log_density_gradient_wrt_bias_precisions(hyperparams),
            wrt_error_precision: self
                .log_density_gradient_wrt_error_precision(y_train, hyperparams),
        }
    }

    fn incr_weight(&mut self, layer: usize, row: u32, col: u32, value: f32) {
        add_at_ix(self.layer_weights_mut(layer), row, col, value);
    }

    fn decr_weight(&mut self, layer: usize, row: u32, col: u32, value: f32) {
        subtract_at_ix(self.layer_weights_mut(layer), row, col, value);
    }

    fn incr_bias(&mut self, layer: usize, col: u32, value: f32) {
        add_at_ix(self.layer_biases_mut(layer), 0, col, value);
    }

    fn decr_bias(&mut self, layer: usize, col: u32, value: f32) {
        subtract_at_ix(self.layer_biases_mut(layer), 0, col, value);
    }

    fn incr_weight_precision(&mut self, layer: usize, row: u32, value: f32) {
        add_at_ix(self.layer_weight_precisions_mut(layer), row, 0, value);
    }

    fn decr_weight_precision(&mut self, layer: usize, row: u32, value: f32) {
        subtract_at_ix(self.layer_weight_precisions_mut(layer), row, 0, value);
    }

    fn incr_bias_precision(&mut self, layer: usize, value: f32) {
        add_at_ix(self.layer_bias_precision_mut(layer), 0, 0, value);
    }

    fn decr_bias_precision(&mut self, layer: usize, value: f32) {
        subtract_at_ix(self.layer_bias_precision_mut(layer), 0, 0, value);
    }

    fn incr_error_precision(&mut self, value: f32) {
        add_at_ix(self.error_precision_mut(), 0, 0, value);
    }

    fn decr_error_precision(&mut self, value: f32) {
        subtract_at_ix(self.error_precision_mut(), 0, 0, value);
    }

    // DO NOT run this in production code, this is extremely slow.
    //
    // This is a drop in replacement for the analytical `log_density_gradient` method.
    fn numerical_log_density_gradient(
        &mut self,
        x_train: &Array<f32>,
        y_train: &Array<f32>,
    ) -> BranchLogDensityGradient {
        BranchLogDensityGradient::from_param_vec(
            &self.numerical_ldg(x_train, y_train),
            self.layer_widths(),
            self.num_markers(),
        )
    }

    // DO NOT run this in production code, this is extremely slow.
    fn numerical_ldg(&mut self, x_train: &Array<f32>, y_train: &Array<f32>) -> Vec<f32> {
        let mut res = Vec::new();
        let mut next_pv = self.params().param_vec();
        let curr_pv = self.params().param_vec();
        let curr_ld =
            self.log_density(self.params(), self.precisions(), self.rss(x_train, y_train));
        let lw = self.layer_widths().clone();
        let nm = self.num_markers();

        for pix in 0..self.num_params() {
            // incr param
            next_pv[pix] += NUMERICAL_DELTA;
            // compute rss, ld
            self.params_mut().load_param_vec(&next_pv, &lw, nm);
            res.push(
                (self.log_density(self.params(), self.precisions(), self.rss(x_train, y_train))
                    - curr_ld)
                    / NUMERICAL_DELTA,
            );
            // decr param
            next_pv[pix] -= NUMERICAL_DELTA;
        }
        self.params_mut().load_param_vec(&curr_pv, &lw, nm);
        res
    }

    fn layer_weights(&self, index: usize) -> &Array<f32> {
        &self.params().weights[index]
    }

    fn layer_weights_mut(&mut self, index: usize) -> &mut Array<f32> {
        self.params_mut().layer_weights_mut(index)
    }

    fn layer_biases_mut(&mut self, index: usize) -> &mut Array<f32> {
        self.params_mut().layer_biases_mut(index)
    }

    fn layer_weight_precisions_mut(&mut self, index: usize) -> &mut Array<f32> {
        self.precisions_mut().layer_weight_precisions_mut(index)
    }

    fn layer_bias_precision_mut(&mut self, index: usize) -> &mut Array<f32> {
        self.precisions_mut().layer_bias_precision_mut(index)
    }

    fn error_precision_mut(&mut self) -> &mut Array<f32> {
        self.precisions_mut().error_precision_mut()
    }

    fn layer_biases(&self, index: usize) -> &Array<f32> {
        &self.params().biases[index]
    }

    fn weight_precisions(&self, index: usize) -> &Array<f32> {
        &self.precisions().weight_precisions[index]
    }

    fn bias_precision(&self, index: usize) -> &Array<f32> {
        &self.precisions().bias_precisions[index]
    }

    fn error_precision(&self) -> &Array<f32> {
        &self.precisions().error_precision
    }

    fn is_accepted(&mut self, acceptance_probability: f32) -> bool {
        self.rng().gen_range(0.0..1.0) < acceptance_probability
    }

    /// Quantify change of distance from starting point
    fn net_movement(&self, init_params: &BranchParams, momentum: &BranchMomentum) -> f32 {
        let mut dot_p = Array::new(&[0.0], dim4!(1, 1, 1, 1));
        for ix in 0..self.num_layers() {
            if self.layer_weights(ix).is_vector() {
                dot_p += dot(
                    &(self.layer_weights(ix) - init_params.layer_weights(ix)),
                    momentum.wrt_weights(ix),
                    MatProp::NONE,
                    MatProp::NONE,
                );
            } else if self.layer_weights(ix).is_scalar() {
                dot_p += (self.layer_weights(ix) - init_params.layer_weights(ix))
                    * momentum.wrt_weights(ix);
            } else {
                dot_p += sum(
                    &diag_extract(
                        &matmul(
                            &(self.layer_weights(ix) - init_params.layer_weights(ix)),
                            momentum.wrt_weights(ix),
                            MatProp::TRANS,
                            MatProp::NONE,
                        ),
                        0,
                    ),
                    0,
                );
            }
        }
        for ix in 0..(self.num_layers() - 1) {
            dot_p += matmul(
                &(self.layer_biases(ix) - init_params.layer_biases(ix)),
                momentum.wrt_biases(ix),
                MatProp::NONE,
                MatProp::TRANS,
            );
        }
        scalar_to_host(&dot_p)
    }

    fn is_u_turn(&self, init_params: &BranchParams, momentum: &BranchMomentum) -> bool {
        self.net_movement(init_params, momentum) < 0.0
    }

    fn sample_momentum(&self) -> BranchMomentum {
        let mut wrt_weights = Vec::with_capacity(self.num_layers());
        let mut wrt_biases = Vec::with_capacity(self.num_layers() - 1);
        for index in 0..self.num_layers() - 1 {
            wrt_weights.push(arrayfire::randn::<f32>(self.layer_weights(index).dims()));
            wrt_biases.push(arrayfire::randn::<f32>(self.layer_biases(index).dims()));
        }
        // output layer weight momentum
        wrt_weights.push(arrayfire::randn::<f32>(
            self.layer_weights(self.num_layers() - 1).dims(),
        ));
        BranchMomentum {
            wrt_weights,
            wrt_biases,
        }
    }

    fn sample_joint_momentum(&self) -> BranchMomentumJoint {
        let mut wrt_weights = Vec::with_capacity(self.num_layers());
        let mut wrt_biases = Vec::with_capacity(self.num_layers() - 1);
        let mut wrt_weight_precisions = Vec::with_capacity(self.num_layers());
        let mut wrt_bias_precisions = Vec::with_capacity(self.num_layers() - 1);
        for index in 0..self.num_layers() - 1 {
            wrt_weights.push(arrayfire::randn::<f32>(self.layer_weights(index).dims()));
            wrt_weight_precisions.push(arrayfire::randn::<f32>(
                self.layer_weight_precisions(index).dims(),
            ));
            wrt_biases.push(arrayfire::randn::<f32>(self.layer_biases(index).dims()));
            wrt_bias_precisions.push(arrayfire::randn::<f32>(
                self.layer_bias_precision(index).dims(),
            ));
        }
        // output layer weight momentum
        wrt_weights.push(arrayfire::randn::<f32>(
            self.layer_weights(self.num_layers() - 1).dims(),
        ));
        wrt_weight_precisions.push(arrayfire::randn::<f32>(
            self.layer_weight_precisions(self.num_layers() - 1).dims(),
        ));
        BranchMomentumJoint {
            wrt_weights,
            wrt_biases,
            wrt_weight_precisions,
            wrt_bias_precisions,
            wrt_error_precision: arrayfire::randn::<f32>(dim4!(1, 1, 1, 1)),
        }
    }

    fn num_precisions(&self) -> usize {
        self.precisions().num_precisions()
    }

    fn layer_weight_precisions(&self, layer_index: usize) -> &Array<f32> {
        self.precisions().layer_weight_precisions(layer_index)
    }

    fn layer_bias_precision(&self, layer_index: usize) -> &Array<f32> {
        self.precisions().layer_bias_precision(layer_index)
    }

    fn random_step_sizes(&mut self, mcmc_cfg: &MCMCCfg) -> StepSizes {
        let const_factor = mcmc_cfg.hmc_step_size_factor;

        let prop_factor = if mcmc_cfg.joint_hmc {
            (self.num_params() as f32 + self.num_precisions() as f32).powf(-0.25) * const_factor
        } else {
            (self.num_params() as f32).powf(-0.25) * const_factor
        };

        let mut wrt_weights = Vec::with_capacity(self.num_layers());
        for index in 0..self.num_layers() {
            wrt_weights.push(randu::<f32>(self.layer_weights(index).dims()) * prop_factor);
        }

        let mut wrt_biases = Vec::with_capacity(self.num_layers() - 1);
        for index in 0..(self.num_layers() - 1) {
            wrt_biases.push(randu::<f32>(self.layer_biases(index).dims()) * prop_factor);
        }

        if !mcmc_cfg.joint_hmc {
            return StepSizes {
                wrt_weights,
                wrt_biases,
                wrt_weight_precisions: None,
                wrt_bias_precisions: None,
                wrt_error_precision: None,
            };
        }

        let mut wrt_weight_precisions = Vec::new();
        for index in 0..self.num_layers() {
            wrt_weight_precisions
                .push(randu::<f32>(self.layer_weight_precisions(index).dims()) * prop_factor);
        }

        let mut wrt_bias_precisions = Vec::new();
        for index in 0..(self.num_layers() - 1) {
            wrt_bias_precisions
                .push(randu::<f32>(self.layer_bias_precision(index).dims()) * prop_factor);
        }

        let wrt_error_precision = randu::<f32>(dim4!(1)) * prop_factor;

        StepSizes {
            wrt_weights,
            wrt_biases,
            wrt_weight_precisions: Some(wrt_weight_precisions),
            wrt_bias_precisions: Some(wrt_bias_precisions),
            wrt_error_precision: Some(wrt_error_precision),
        }
    }

    fn uniform_step_sizes(&self, mcmc_cfg: &MCMCCfg) -> StepSizes {
        let val = mcmc_cfg.hmc_step_size_factor;
        let mut wrt_weights = Vec::with_capacity(self.num_layers());
        let mut wrt_biases = Vec::with_capacity(self.num_layers() - 1);
        for index in 0..self.num_layers() - 1 {
            wrt_weights.push(Array::new(
                &vec![val; self.layer_weights(index).elements()],
                self.layer_weights(index).dims(),
            ));
            wrt_biases.push(Array::new(
                &vec![val; self.layer_biases(index).elements()],
                self.layer_biases(index).dims(),
            ));
        }
        // output layer weights
        wrt_weights.push(Array::new(
            &vec![val; self.layer_weights(self.num_layers() - 1).elements()],
            self.layer_weights(self.num_layers() - 1).dims(),
        ));
        StepSizes {
            wrt_weights,
            wrt_biases,
            wrt_weight_precisions: None,
            wrt_bias_precisions: None,
            wrt_error_precision: None,
        }
    }

    /// Sets step sizes proportional to the prior standard deviation of each parameter.
    fn std_scaled_step_sizes(&self, mcmc_cfg: &MCMCCfg) -> StepSizes;

    fn izmailov_step_sizes(&mut self, mcmc_cfg: &MCMCCfg) -> StepSizes;

    /// Returns pre-activations and activations for all neurons in the network.
    ///
    /// Pre-activation of the output neuron is the same as its activation, these values
    /// are therefore not included in the pre-activations
    fn forward_feed(&self, x_train: &Array<f32>) -> (Vec<Array<f32>>, Vec<Array<f32>>) {
        let mut pre_activations: Vec<Array<f32>> = Vec::with_capacity(self.num_layers() - 2);
        let mut activations: Vec<Array<f32>> = Vec::with_capacity(self.num_layers() - 1);

        pre_activations.push(self.mid_layer_pre_activation(0, x_train));
        activations.push(self.h(pre_activations.last().unwrap()));

        for layer_index in 1..self.num_layers() - 1 {
            pre_activations
                .push(self.mid_layer_pre_activation(layer_index, activations.last().unwrap()));
            activations.push(self.h(pre_activations.last().unwrap()));
        }

        activations.push(self.output_neuron_activation(activations.last().unwrap()));
        (pre_activations, activations)
    }

    fn mid_layer_pre_activation(&self, layer_index: usize, input: &Array<f32>) -> Array<f32> {
        let xw = matmul(
            input,
            self.layer_weights(layer_index),
            MatProp::NONE,
            MatProp::NONE,
        );
        // TODO: tiling here everytime seems a bit inefficient, might be better to just store tiled versions of the biases?
        let bias_m = &arrayfire::tile(
            self.layer_biases(layer_index),
            dim4!(input.dims().get()[0], 1, 1, 1),
        );
        xw + bias_m
    }

    fn output_neuron_activation(&self, input: &Array<f32>) -> Array<f32> {
        matmul(
            input,
            self.layer_weights(self.num_layers() - 1),
            MatProp::NONE,
            MatProp::NONE,
        )
    }

    /// Computes the absolute values of the
    /// partial derivatives of the predicted value w.r.t. the input.
    /// The output is a n x m matrix.
    fn effect_sizes(&self, x_train: &Array<f32>, _y_train: &Array<f32>) -> Array<f32> {
        // forward propagate to get signals
        let (pre_activations, activations) = self.forward_feed(x_train);

        // back propagate
        let mut error = matmul(
            &activations.last().unwrap(),
            self.layer_weights(self.num_layers() - 1),
            MatProp::NONE,
            MatProp::TRANS,
        );

        for layer_index in (0..self.num_layers() - 1).rev() {
            // activation = &activations[layer_index];
            let delta: Array<f32> = self.dhdx(&pre_activations[layer_index]) * error;
            // let delta: Array<f32> = (1 - arrayfire::pow(activation, &2, false)) * error;
            error = matmul(
                &delta,
                self.layer_weights(layer_index),
                MatProp::NONE,
                MatProp::TRANS,
            );
        }
        error
    }

    fn backpropagate(&mut self, x_train: &Array<f32>, y_train: &Array<f32>) {
        // forward propagate to get signals
        let (pre_activations, activations) = self.forward_feed(x_train);

        let mut bias_gradient: Vec<Array<f32>> = Vec::with_capacity(self.num_layers() - 1);
        let mut weights_gradient: Vec<Array<f32>> = Vec::with_capacity(self.num_layers());

        // back propagate
        let mut error = activations.last().unwrap() - y_train;

        self.set_last_rss(&arrayfire::dot(
            &error,
            &error,
            MatProp::NONE,
            MatProp::NONE,
        ));

        weights_gradient.push(arrayfire::matmul(
            &activations[self.num_layers() - 2],
            &error,
            MatProp::TRANS,
            MatProp::NONE,
        ));

        error = matmul(
            &error,
            self.layer_weights(self.num_layers() - 1),
            MatProp::NONE,
            MatProp::TRANS,
        );

        for layer_index in (1..self.num_layers() - 1).rev() {
            let input = &activations[layer_index - 1];
            // activation = &activations[layer_index];
            let delta: Array<f32> = self.dhdx(&pre_activations[layer_index]) * error;
            bias_gradient.push(arrayfire::sum(&delta, 0));
            weights_gradient.push(arrayfire::transpose(
                &matmul(&delta, input, MatProp::TRANS, MatProp::NONE),
                false,
            ));
            error = matmul(
                &delta,
                self.layer_weights(layer_index),
                MatProp::NONE,
                MatProp::TRANS,
            );
        }

        let delta: Array<f32> = self.dhdx(&pre_activations[0]) * error;
        bias_gradient.push(arrayfire::sum(&delta, 0));
        weights_gradient.push(arrayfire::transpose(
            &matmul(&delta, x_train, MatProp::TRANS, MatProp::NONE),
            false,
        ));

        bias_gradient.reverse();
        weights_gradient.reverse();

        self.training_state_mut()
            .set_d_rss_wrt_weights(&weights_gradient);
        self.training_state_mut()
            .set_d_rss_wrt_biases(&bias_gradient);
    }

    // this is -H = (-U(q)) + (-K(p))
    fn neg_hamiltonian<T>(&self, momentum: &T, x: &Array<f32>, y: &Array<f32>) -> f32
    where
        T: Momentum,
    {
        self.log_density(self.params(), self.precisions(), self.rss(x, y)) - momentum.log_density()
    }

    // this is -H = (-U(q)) + (-K(p))
    fn neg_hamiltonian_joint<T>(
        &self,
        momentum: &T,
        x: &Array<f32>,
        y: &Array<f32>,
        hyperparams: &NetworkPrecisionHyperparameters,
    ) -> f32
    where
        T: Momentum,
    {
        self.log_density_joint(
            self.params(),
            self.precisions(),
            self.rss(x, y),
            hyperparams,
            y.elements(),
        ) - momentum.log_density()
    }

    fn rss(&self, x: &Array<f32>, y: &Array<f32>) -> f32 {
        let (_pre_activations, activations) = self.forward_feed(x);
        let r = activations.last().unwrap() - y;
        arrayfire::sum_all(&(&r * &r)).0
    }

    fn r2(&self, x: &Array<f32>, y: &Array<f32>) -> f32 {
        1. - self.rss(x, y) / arrayfire::sum_all(&(y * y)).0
    }

    fn predict(&self, x: &Array<f32>) -> Array<f32> {
        let (_pre_activations, activations) = self.forward_feed(x);
        activations.last().unwrap().copy()
    }

    fn prediction_and_density(&self, x: &Array<f32>, y: &Array<f32>) -> (Array<f32>, f32) {
        let y_pred = self.predict(x);
        let r = &y_pred - y;
        let rss = arrayfire::sum_all(&(&r * &r)).0;
        let log_density = self.log_density(self.params(), self.precisions(), rss);
        (y_pred, log_density)
    }

    fn accept_or_reject_hmc_state<T>(
        &mut self,
        momentum: &T,
        x_train: &Array<f32>,
        y_train: &Array<f32>,
        init_neg_hamiltonian: f32,
    ) -> HMCStepResult
    where
        T: Momentum,
    {
        // this is self.neg_hamiltonian unpacked. Doing this in order to save
        // one forward pass.
        let y_pred = self.predict(x_train);
        let r = &y_pred - y_train;
        let rss = arrayfire::sum_all(&(&r * &r)).0;
        let log_density = self.log_density(self.params(), self.precisions(), rss);
        // debug!("branch log density after step: {:.4}", log_density);
        let state_data = HMCStepResultData {
            y_pred,
            log_density,
        };

        let final_neg_hamiltonian = log_density - momentum.log_density();
        let log_acc_probability = final_neg_hamiltonian - init_neg_hamiltonian;
        let acc_probability = if log_acc_probability >= 0. {
            1.
        } else {
            log_acc_probability.exp()
        };
        if self.is_accepted(acc_probability) {
            HMCStepResult::Accepted(state_data)
        } else {
            HMCStepResult::Rejected
        }
    }

    fn gradient_descent(
        &mut self,
        x_train: &Array<f32>,
        y_train: &Array<f32>,
        mcmc_cfg: &MCMCCfg,
    ) -> HMCStepResult {
        let mut ldg = self.log_density_gradient(x_train, y_train);
        for _step in 0..(mcmc_cfg.hmc_integration_length) {
            self.params_mut()
                .descend_gradient(mcmc_cfg.hmc_step_size_factor, &ldg);
            ldg = self.log_density_gradient(x_train, y_train);
        }
        let y_pred = self.predict(x_train);
        let r = &y_pred - y_train;
        let rss = arrayfire::sum_all(&(&r * &r)).0;
        let log_density = self.log_density(self.params(), self.precisions(), rss);
        debug!("branch log density after step: {:.4}", log_density);

        HMCStepResult::Accepted(HMCStepResultData {
            y_pred,
            log_density,
        })
    }

    /// Performs gradient descent step for weights, biases and their precisions
    fn gradient_descent_joint(
        &mut self,
        x_train: &Array<f32>,
        y_train: &Array<f32>,
        mcmc_cfg: &MCMCCfg,
        hyperparams: &NetworkPrecisionHyperparameters,
    ) -> HMCStepResult {
        debug!(
            "error_precision before step: {:.4}",
            scalar_to_host(self.error_precision())
        );
        let init_params = self.params().clone();
        let init_precisions = self.precisions().clone();
        let mut ldg = self.log_density_gradient_joint(x_train, y_train, hyperparams);
        for _step in 0..(mcmc_cfg.hmc_integration_length) {
            self.params_mut()
                .descend_gradient(mcmc_cfg.hmc_step_size_factor, &ldg);
            self.precisions_mut()
                .descend_gradient(mcmc_cfg.hmc_step_size_factor, &ldg);
            ldg = self.log_density_gradient_joint(x_train, y_train, hyperparams);
        }
        let y_pred = self.predict(x_train);
        let r = &y_pred - y_train;
        let rss = arrayfire::sum_all(&(&r * &r)).0;
        let log_density = self.log_density_joint(
            self.params(),
            self.precisions(),
            rss,
            hyperparams,
            y_pred.elements(),
        );
        debug!("branch log density after step: {:.4}", log_density);
        debug!(
            "error_precision after step: {:.4}",
            scalar_to_host(self.error_precision())
        );

        if scalar_to_host(self.error_precision()) <= 0.0 {
            self.set_params(&init_params);
            self.set_precisions(&init_precisions);
            HMCStepResult::Rejected
        } else {
            HMCStepResult::Accepted(HMCStepResultData {
                y_pred,
                log_density,
            })
        }
    }

    /// Takes a single parameter and hyperparameter sample using HMC.
    /// Returns `false` if final state is rejected, `true` if accepted.
    fn hmc_step_joint(
        &mut self,
        x_train: &Array<f32>,
        y_train: &Array<f32>,
        mcmc_cfg: &MCMCCfg,
        hyperparams: &NetworkPrecisionHyperparameters,
    ) -> HMCStepResult {
        // adjust output weight reg_sum for own branch output weight sum
        let mut traj_file = None;
        let mut traj = Trajectory::new();
        if mcmc_cfg.trajectories {
            traj_file = Some(BufWriter::new(
                File::options()
                    .append(true)
                    .create(true)
                    .open(mcmc_cfg.trajectories_path())
                    .unwrap(),
            ));
        }
        if mcmc_cfg.num_grad {
            warn!("Joint sampling does not support numerical gradients yet. Using analytical gradients instead.");
        }
        // std scaled and izmailov might violate detailed balance for joint sampling,
        // I should check that
        match mcmc_cfg.hmc_step_size_mode {
            StepSizeMode::Izmailov => warn!("Join sampling does not support Izmailov step sizes yet. Using random step sizes instead."),
            StepSizeMode::StdScaled => warn!("Join sampling does not support StdScaled step sizes yet. Using random step sizes instead."),
            StepSizeMode::Uniform => warn!("Join sampling does not support Uniform step sizes yet. Using random step sizes instead."),
            _ => {}
        }

        let init_params = self.params().clone();
        let init_precisions = self.precisions().clone();
        let step_sizes = self.random_step_sizes(mcmc_cfg);
        let mut momentum = self.sample_joint_momentum();
        let init_neg_hamiltonian =
            self.neg_hamiltonian_joint(&momentum, x_train, y_train, hyperparams);
        let mut ldg = self.log_density_gradient_joint(x_train, y_train, hyperparams);

        if mcmc_cfg.trajectories {
            traj.add_hamiltonian(init_neg_hamiltonian);
        }

        // leapfrog
        for _step in 0..(mcmc_cfg.hmc_integration_length) {
            momentum.half_step(&step_sizes, &ldg);
            self.params_mut().full_step(&step_sizes, &momentum);
            self.precisions_mut().full_step(&step_sizes, &momentum);
            ldg = self.log_density_gradient_joint(x_train, y_train, hyperparams);
            momentum.half_step(&step_sizes, &ldg);

            // diagnostics and logging

            let curr_neg_hamiltonian =
                self.neg_hamiltonian_joint(&momentum, x_train, y_train, hyperparams);

            if mcmc_cfg.trajectories {
                traj.add_params(self.params().param_vec());
                traj.add_precisions(self.precisions().param_vec());
                traj.add_ldg(ldg.param_vec());
                traj.add_hamiltonian(curr_neg_hamiltonian);
                if mcmc_cfg.num_grad_traj {
                    // not implemented
                    // traj.add_num_ldg(self.numerical_ldg(x_train, y_train));
                }
            }

            if (curr_neg_hamiltonian - init_neg_hamiltonian).abs()
                > mcmc_cfg.hmc_max_hamiltonian_error
            {
                debug!(
                    "step: {}; hamiltonian error threshold crossed: terminating",
                    _step
                );

                if mcmc_cfg.trajectories {
                    to_writer(traj_file.as_mut().unwrap(), &traj).unwrap();
                    traj_file.as_mut().unwrap().write_all(b"\n").unwrap();
                }

                self.set_params(&init_params);
                self.set_precisions(&init_precisions);
                return HMCStepResult::RejectedEarly;
            }
        }

        if mcmc_cfg.trajectories {
            to_writer(traj_file.as_mut().unwrap(), &traj).unwrap();
            traj_file.as_mut().unwrap().write_all(b"\n").unwrap();
        }

        let res = match self.accept_or_reject_hmc_state(
            &momentum,
            x_train,
            y_train,
            init_neg_hamiltonian,
        ) {
            res @ HMCStepResult::Rejected => {
                self.set_params(&init_params);
                self.set_precisions(&init_precisions);
                res
            }
            res => res,
        };

        // add own branch output weight sum to summary reg sum

        res
    }

    fn subtract_output_weight_summary_stat_from_global(&mut self) {
        let by = self.summary_stat_fn(&self.params().output_layer_weights());
        self.output_weight_summary_stats_mut().decr_reg_sum(&by);
    }

    fn add_output_weight_summary_stat_to_global(&mut self) {
        let by = self.summary_stat_fn(&self.params().output_layer_weights());
        self.output_weight_summary_stats_mut().incr_reg_sum(&by);
    }

    /// Takes a single parameter sample using HMC.
    /// Returns `false` if final state is rejected, `true` if accepted.
    fn hmc_step(
        &mut self,
        x_train: &Array<f32>,
        y_train: &Array<f32>,
        mcmc_cfg: &MCMCCfg,
    ) -> HMCStepResult {
        let mut traj_file = None;
        let mut traj = Trajectory::new();
        if mcmc_cfg.trajectories {
            traj_file = Some(BufWriter::new(
                File::options()
                    .append(true)
                    .create(true)
                    .open(mcmc_cfg.trajectories_path())
                    .unwrap(),
            ));
        }

        let mut u_turned = false;
        let init_params = self.params().clone();
        let step_sizes = match mcmc_cfg.hmc_step_size_mode {
            StepSizeMode::StdScaled => self.std_scaled_step_sizes(mcmc_cfg),
            StepSizeMode::Random => self.random_step_sizes(mcmc_cfg),
            StepSizeMode::Uniform => self.uniform_step_sizes(mcmc_cfg),
            StepSizeMode::Izmailov => self.izmailov_step_sizes(mcmc_cfg),
        };

        let mut momentum = self.sample_momentum();
        // debug!(
        //     "branch log density before step: {:.4}",
        //     self.log_density(self.params(), self.precisions(), self.rss(x_train, y_train))
        // );
        let init_neg_hamiltonian = self.neg_hamiltonian(&momentum, x_train, y_train);

        if mcmc_cfg.trajectories {
            traj.add_hamiltonian(init_neg_hamiltonian);
        }

        // debug!("Starting hmc step");
        // debug!("initial hamiltonian: {:?}", init_neg_hamiltonian);
        let mut ldg = if mcmc_cfg.num_grad {
            self.numerical_log_density_gradient(x_train, y_train)
        } else {
            self.log_density_gradient(x_train, y_train)
        };

        // leapfrog
        for _step in 0..(mcmc_cfg.hmc_integration_length) {
            momentum.half_step(&step_sizes, &ldg);
            self.params_mut().full_step(&step_sizes, &momentum);

            ldg = if mcmc_cfg.num_grad {
                self.numerical_log_density_gradient(x_train, y_train)
            } else {
                self.log_density_gradient(x_train, y_train)
            };

            momentum.half_step(&step_sizes, &ldg);

            // diagnostics and logging

            let curr_neg_hamiltonian = self.neg_hamiltonian(&momentum, x_train, y_train);

            if mcmc_cfg.trajectories {
                traj.add_params(self.params().param_vec());
                traj.add_ldg(ldg.param_vec());
                traj.add_hamiltonian(curr_neg_hamiltonian);
                if mcmc_cfg.num_grad_traj {
                    traj.add_num_ldg(self.numerical_ldg(x_train, y_train));
                }
            }

            if (curr_neg_hamiltonian - init_neg_hamiltonian).abs()
                > mcmc_cfg.hmc_max_hamiltonian_error
            {
                debug!(
                    "step: {}; hamiltonian error threshold crossed: terminating",
                    _step
                );

                if mcmc_cfg.trajectories {
                    to_writer(traj_file.as_mut().unwrap(), &traj).unwrap();
                    traj_file.as_mut().unwrap().write_all(b"\n").unwrap();
                }

                self.set_params(&init_params);
                return HMCStepResult::RejectedEarly;
            }

            if !u_turned && self.is_u_turn(&init_params, &momentum) {
                warn!("U turn in HMC trajectory at step {}", _step);
                u_turned = true;
            }
        }

        if mcmc_cfg.trajectories {
            to_writer(traj_file.as_mut().unwrap(), &traj).unwrap();
            traj_file.as_mut().unwrap().write_all(b"\n").unwrap();
        }

        match self.accept_or_reject_hmc_state(&momentum, x_train, y_train, init_neg_hamiltonian) {
            res @ HMCStepResult::Rejected => {
                self.set_params(&init_params);
                res
            }
            res => res,
        }
    }
}

/// Performance statistics of branch on train data.
pub struct HMCStepResultData {
    /// Mean squared error
    pub y_pred: Array<f32>,
    /// Log density
    pub log_density: f32,
}

pub enum HMCStepResult {
    RejectedEarly,
    Rejected,
    Accepted(HMCStepResultData),
}
