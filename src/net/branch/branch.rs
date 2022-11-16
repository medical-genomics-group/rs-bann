use super::{
    super::{
        mcmc_cfg::{MCMCCfg, StepSizeMode},
        net::ModelType,
    },
    branch_cfg_builder::BranchCfgBuilder,
    momenta::BranchMomenta,
    params::BranchParams,
    params::BranchPrecisions,
    step_sizes::StepSizes,
    trajectory::Trajectory,
};
use crate::scalar_to_host;
use arrayfire::{diag_extract, dim4, dot, matmul, randu, sum, tanh, Array, MatProp};
use log::{debug, warn};
use rand::{prelude::ThreadRng, Rng};
use serde::{Deserialize, Serialize};
use serde_json::to_writer;
use std::{
    fs::File,
    io::{BufWriter, Write},
};

const NUMERICAL_DELTA: f32 = 0.001;
const GD_STEP_SIZE: f32 = 0.00001;

pub trait Branch {
    fn model_type() -> ModelType;

    fn build_cfg(cfg_bld: BranchCfgBuilder) -> BranchCfg;

    fn from_cfg(cfg: &BranchCfg) -> Self;

    fn to_cfg(&self) -> BranchCfg;

    fn set_params(&mut self, params: &BranchParams);

    fn params(&self) -> &BranchParams;

    fn params_mut(&mut self) -> &mut BranchParams;

    fn hyperparams(&self) -> &BranchPrecisions;

    fn num_params(&self) -> usize;

    fn num_layers(&self) -> usize;

    fn layer_width(&self, index: usize) -> usize;

    fn set_error_precision(&mut self, val: f32);

    fn rng(&mut self) -> &mut ThreadRng;

    fn sample_precisions(&mut self, prior_shape: f32, prior_scale: f32);

    fn num_markers(&self) -> usize;

    fn layer_widths(&self) -> &Vec<usize>;

    fn summary_layer_index(&self) -> usize {
        self.num_layers() - 2
    }

    fn output_layer_index(&self) -> usize {
        self.num_layers() - 1
    }

    fn log_density_gradient(
        &self,
        x_train: &Array<f32>,
        y_train: &Array<f32>,
    ) -> BranchLogDensityGradient;

    // This should be -U(q), e.g. log P(D | Theta)P(Theta)
    fn log_density(&self, params: &BranchParams, hyperparams: &BranchPrecisions, rss: f32) -> f32;

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
        let curr_ld = self.log_density(
            self.params(),
            self.hyperparams(),
            self.rss(x_train, y_train),
        );
        let lw = self.layer_widths().clone();
        let nm = self.num_markers();

        for pix in 0..self.num_params() {
            // incr param
            next_pv[pix] += NUMERICAL_DELTA;
            // compute rss, ld
            self.params_mut().load_param_vec(&next_pv, &lw, nm);
            res.push(
                (self.log_density(
                    self.params(),
                    self.hyperparams(),
                    self.rss(x_train, y_train),
                ) - curr_ld)
                    / NUMERICAL_DELTA,
            );
            // decr param
            next_pv[pix] -= NUMERICAL_DELTA;
        }
        self.params_mut().load_param_vec(&curr_pv, &lw, nm);
        res
    }

    fn weights(&self, index: usize) -> &Array<f32> {
        &self.params().weights[index]
    }

    fn biases(&self, index: usize) -> &Array<f32> {
        &self.params().biases[index]
    }

    fn weight_precisions(&self, index: usize) -> &Array<f32> {
        &self.hyperparams().weight_precisions[index]
    }

    fn bias_precision(&self, index: usize) -> f32 {
        self.hyperparams().bias_precisions[index]
    }

    fn error_precision(&self) -> f32 {
        self.hyperparams().error_precision
    }

    fn is_accepted(&mut self, acceptance_probability: f32) -> bool {
        self.rng().gen_range(0.0..1.0) < acceptance_probability
    }

    /// Quantify change of distance from starting point
    fn net_movement(&self, init_params: &BranchParams, momenta: &BranchMomenta) -> f32 {
        let mut dot_p = Array::new(&[0.0], dim4!(1, 1, 1, 1));
        for ix in 0..self.num_layers() {
            if self.weights(ix).is_vector() {
                dot_p += dot(
                    &(self.weights(ix) - init_params.weights(ix)),
                    momenta.wrt_weights(ix),
                    MatProp::NONE,
                    MatProp::NONE,
                );
            } else if self.weights(ix).is_scalar() {
                dot_p += (self.weights(ix) - init_params.weights(ix)) * momenta.wrt_weights(ix);
            } else {
                dot_p += sum(
                    &diag_extract(
                        &matmul(
                            &(self.weights(ix) - init_params.weights(ix)),
                            momenta.wrt_weights(ix),
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
                &(self.biases(ix) - init_params.biases(ix)),
                momenta.wrt_biases(ix),
                MatProp::NONE,
                MatProp::TRANS,
            );
        }
        scalar_to_host(&dot_p)
    }

    fn is_u_turn(&self, init_params: &BranchParams, momenta: &BranchMomenta) -> bool {
        self.net_movement(init_params, momenta) < 0.0
    }

    fn sample_momenta(&self) -> BranchMomenta {
        let mut wrt_weights = Vec::with_capacity(self.num_layers());
        let mut wrt_biases = Vec::with_capacity(self.num_layers() - 1);
        for index in 0..self.num_layers() - 1 {
            wrt_weights.push(arrayfire::randn::<f32>(self.weights(index).dims()));
            wrt_biases.push(arrayfire::randn::<f32>(self.biases(index).dims()));
        }
        // output layer weight momentum
        wrt_weights.push(arrayfire::randn::<f32>(
            self.weights(self.num_layers() - 1).dims(),
        ));
        BranchMomenta {
            wrt_weights,
            wrt_biases,
        }
    }

    fn random_step_sizes(&mut self, const_factor: f32) -> StepSizes {
        let mut wrt_weights = Vec::with_capacity(self.num_layers());
        let mut wrt_biases = Vec::with_capacity(self.num_layers() - 1);
        let prop_factor = (self.num_params() as f32).powf(-0.25) * const_factor;

        for index in 0..self.num_layers() {
            wrt_weights.push(randu::<f32>(self.weights(index).dims()) * prop_factor);
        }

        for index in 0..(self.num_layers() - 1) {
            wrt_biases.push(randu::<f32>(self.biases(index).dims()) * prop_factor);
        }
        StepSizes {
            wrt_weights,
            wrt_biases,
        }
    }

    fn uniform_step_sizes(&self, val: f32) -> StepSizes {
        let mut wrt_weights = Vec::with_capacity(self.num_layers());
        let mut wrt_biases = Vec::with_capacity(self.num_layers() - 1);
        for index in 0..self.num_layers() - 1 {
            wrt_weights.push(Array::new(
                &vec![val; self.weights(index).elements()],
                self.weights(index).dims(),
            ));
            wrt_biases.push(Array::new(
                &vec![val; self.biases(index).elements()],
                self.biases(index).dims(),
            ));
        }
        // output layer weights
        wrt_weights.push(Array::new(
            &vec![val; self.weights(self.num_layers() - 1).elements()],
            self.weights(self.num_layers() - 1).dims(),
        ));
        StepSizes {
            wrt_weights,
            wrt_biases,
        }
    }

    /// Sets step sizes proportional to the prior standard deviation of each parameter.
    fn std_scaled_step_sizes(&self, const_factor: f32) -> StepSizes;

    fn izmailov_step_sizes(&mut self, integration_length: usize) -> StepSizes;

    fn forward_feed(&self, x_train: &Array<f32>) -> Vec<Array<f32>> {
        let mut activations: Vec<Array<f32>> = Vec::with_capacity(self.num_layers() - 1);
        activations.push(self.mid_layer_activation(0, x_train));
        for layer_index in 1..self.num_layers() - 1 {
            activations.push(self.mid_layer_activation(layer_index, activations.last().unwrap()));
        }
        activations.push(self.output_neuron_activation(activations.last().unwrap()));
        activations
    }

    fn mid_layer_activation(&self, layer_index: usize, input: &Array<f32>) -> Array<f32> {
        let xw = matmul(
            input,
            self.weights(layer_index),
            MatProp::NONE,
            MatProp::NONE,
        );
        // TODO: tiling here everytime seems a bit inefficient, might be better to just store tiled versions of the biases?
        let bias_m = &arrayfire::tile(
            self.biases(layer_index),
            dim4!(input.dims().get()[0], 1, 1, 1),
        );
        tanh(&(xw + bias_m))
    }

    fn output_neuron_activation(&self, input: &Array<f32>) -> Array<f32> {
        matmul(
            input,
            self.weights(self.num_layers() - 1),
            MatProp::NONE,
            MatProp::NONE,
        )
    }

    fn backpropagate(
        &self,
        x_train: &Array<f32>,
        y_train: &Array<f32>,
    ) -> (Vec<Array<f32>>, Vec<Array<f32>>) {
        // forward propagate to get signals
        let activations = self.forward_feed(x_train);

        let mut bias_gradient: Vec<Array<f32>> = Vec::with_capacity(self.num_layers() - 1);
        let mut weights_gradient: Vec<Array<f32>> = Vec::with_capacity(self.num_layers());
        // back propagate
        let mut activation = activations.last().unwrap();

        // TODO: factor of 2 might be necessary here?
        let mut error = activation - y_train;
        weights_gradient.push(arrayfire::dot(
            &error,
            &activations[self.num_layers() - 2],
            MatProp::NONE,
            MatProp::NONE,
        ));
        error = matmul(
            &error,
            self.weights(self.num_layers() - 1),
            MatProp::NONE,
            MatProp::NONE,
        );

        for layer_index in (1..self.num_layers() - 1).rev() {
            let input = &activations[layer_index - 1];
            activation = &activations[layer_index];
            let delta: Array<f32> = (1 - arrayfire::pow(activation, &2, false)) * error;
            bias_gradient.push(arrayfire::sum(&delta, 0));
            weights_gradient.push(arrayfire::transpose(
                &matmul(&delta, input, MatProp::TRANS, MatProp::NONE),
                false,
            ));
            error = matmul(
                &delta,
                self.weights(layer_index),
                MatProp::NONE,
                MatProp::TRANS,
            );
        }

        let delta: Array<f32> = (1 - arrayfire::pow(&activations[0], &2, false)) * error;
        bias_gradient.push(arrayfire::sum(&delta, 0));
        weights_gradient.push(arrayfire::transpose(
            &matmul(&delta, x_train, MatProp::TRANS, MatProp::NONE),
            false,
        ));

        bias_gradient.reverse();
        weights_gradient.reverse();

        (weights_gradient, bias_gradient)
    }

    // this is -H = (-U(q)) + (-K(p))
    fn neg_hamiltonian(&self, momenta: &BranchMomenta, x: &Array<f32>, y: &Array<f32>) -> f32 {
        self.log_density(self.params(), self.hyperparams(), self.rss(x, y)) - momenta.log_density()
    }

    fn rss(&self, x: &Array<f32>, y: &Array<f32>) -> f32 {
        let r = self.forward_feed(x).last().unwrap() - y;
        arrayfire::sum_all(&(&r * &r)).0
    }

    fn r2(&self, x: &Array<f32>, y: &Array<f32>) -> f32 {
        1. - self.rss(x, y) / arrayfire::sum_all(&(y * y)).0
    }

    fn predict(&self, x: &Array<f32>) -> Array<f32> {
        self.forward_feed(x).last().unwrap().copy()
    }

    fn prediction_and_density(&self, x: &Array<f32>, y: &Array<f32>) -> (Array<f32>, f32) {
        let y_pred = self.predict(x);
        let r = &y_pred - y;
        let rss = arrayfire::sum_all(&(&r * &r)).0;
        let log_density = self.log_density(self.params(), self.hyperparams(), rss);
        (y_pred, log_density)
    }

    fn accept_or_reject_hmc_state(
        &mut self,
        momenta: &BranchMomenta,
        x_train: &Array<f32>,
        y_train: &Array<f32>,
        init_neg_hamiltonian: f32,
    ) -> HMCStepResult {
        // this is self.neg_hamiltonian unpacked. Doing this in order to save
        // one forward pass.
        let y_pred = self.predict(x_train);
        let r = &y_pred - y_train;
        let rss = arrayfire::sum_all(&(&r * &r)).0;
        let log_density = self.log_density(self.params(), self.hyperparams(), rss);
        debug!("branch log density after step: {:.4}", log_density);
        let state_data = HMCStepResultData {
            y_pred,
            log_density,
        };

        let final_neg_hamiltonian = log_density - momenta.log_density();
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
            self.params_mut().descent_gradient(GD_STEP_SIZE, &ldg);
            ldg = self.log_density_gradient(x_train, y_train);
        }
        let y_pred = self.predict(x_train);
        let r = &y_pred - y_train;
        let rss = arrayfire::sum_all(&(&r * &r)).0;
        let log_density = self.log_density(self.params(), self.hyperparams(), rss);
        debug!("branch log density after step: {:.4}", log_density);
        HMCStepResult::Accepted(HMCStepResultData {
            y_pred,
            log_density,
        })
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
            StepSizeMode::StdScaled => self.std_scaled_step_sizes(mcmc_cfg.hmc_step_size_factor),
            StepSizeMode::Random => self.random_step_sizes(mcmc_cfg.hmc_step_size_factor),
            StepSizeMode::Uniform => self.uniform_step_sizes(mcmc_cfg.hmc_step_size_factor),
            StepSizeMode::Izmailov => self.izmailov_step_sizes(mcmc_cfg.hmc_integration_length),
        };
        // debug!("Using step sizes: {:?}", step_sizes);
        let mut momenta = self.sample_momenta();
        debug!(
            "branch log density before step: {:.4}",
            self.log_density(
                self.params(),
                self.hyperparams(),
                self.rss(x_train, y_train)
            )
        );
        let init_neg_hamiltonian = self.neg_hamiltonian(&momenta, x_train, y_train);

        if mcmc_cfg.trajectories {
            traj.add_hamiltonian(init_neg_hamiltonian);
        }

        debug!("Starting hmc step");
        debug!("initial hamiltonian: {:?}", init_neg_hamiltonian);
        let mut ldg = if mcmc_cfg.num_grad {
            self.numerical_log_density_gradient(x_train, y_train)
        } else {
            self.log_density_gradient(x_train, y_train)
        };

        // leapfrog
        for _step in 0..(mcmc_cfg.hmc_integration_length) {
            momenta.half_step(&step_sizes, &ldg);
            self.params_mut().full_step(&step_sizes, &momenta);

            ldg = if mcmc_cfg.num_grad {
                self.numerical_log_density_gradient(x_train, y_train)
            } else {
                self.log_density_gradient(x_train, y_train)
            };

            momenta.half_step(&step_sizes, &ldg);

            // diagnostics and logging

            let curr_neg_hamiltonian = self.neg_hamiltonian(&momenta, x_train, y_train);

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

            if !u_turned && self.is_u_turn(&init_params, &momenta) {
                warn!("U turn in HMC trajectory at step {}", _step);
                u_turned = true;
            }
        }

        if mcmc_cfg.trajectories {
            to_writer(traj_file.as_mut().unwrap(), &traj).unwrap();
            traj_file.as_mut().unwrap().write_all(b"\n").unwrap();
        }

        // debug!("final gradients");
        // for lix in 0..ldg.wrt_weights.len() {
        //     debug!(
        //         "layer: {:}; weight grad: {:?}",
        //         lix,
        //         to_host(&ldg.wrt_weights[lix])
        //     );
        // }
        // for lix in 0..ldg.wrt_biases.len() {
        //     debug!(
        //         "layer: {:}; bias grad: {:?}",
        //         lix,
        //         to_host(&ldg.wrt_biases[lix])
        //     );
        // }

        match self.accept_or_reject_hmc_state(&momenta, &x_train, &y_train, init_neg_hamiltonian) {
            res @ HMCStepResult::Rejected => {
                self.set_params(&init_params);
                res
            }
            res @ _ => res,
        }
    }
}

/// Gradients of the log density w.r.t. the network parameters.
#[derive(Clone)]
pub struct BranchLogDensityGradient {
    pub wrt_weights: Vec<Array<f32>>,
    pub wrt_biases: Vec<Array<f32>>,
}

impl BranchLogDensityGradient {
    pub fn from_param_vec(
        param_vec: &[f32],
        layer_widths: &Vec<usize>,
        num_markers: usize,
    ) -> Self {
        let mut wrt_weights: Vec<Array<f32>> = vec![];
        let mut wrt_biases: Vec<Array<f32>> = vec![];
        let mut prev_width = num_markers;
        let mut read_ix: usize = 0;
        for width in layer_widths {
            let num_weights = prev_width * width;
            wrt_weights.push(Array::new(
                &param_vec[read_ix..read_ix + num_weights],
                dim4!(prev_width as u64, *width as u64, 1, 1),
            ));
            prev_width = *width;
            read_ix += num_weights;
        }
        for width in &layer_widths[..layer_widths.len() - 1] {
            let num_biases = width;
            wrt_biases.push(Array::new(
                &param_vec[read_ix..read_ix + num_biases],
                dim4!(1, *width as u64, 1, 1),
            ));
            read_ix += num_biases;
        }
        Self {
            wrt_weights,
            wrt_biases,
        }
    }

    fn num_params(&self) -> usize {
        let mut res: usize = 0;
        for i in 0..self.wrt_weights.len() {
            res += self.wrt_weights[i].elements();
        }
        for i in 0..self.wrt_biases.len() {
            res += self.wrt_biases[i].elements();
        }
        res
    }

    pub(crate) fn param_vec(&self) -> Vec<f32> {
        let mut host_vec = Vec::new();
        host_vec.resize(self.num_params(), 0.);
        let mut insert_ix: usize = 0;
        for i in 0..self.wrt_weights.len() {
            let len = self.wrt_weights[i].elements();
            self.wrt_weights[i].host(&mut host_vec[insert_ix..insert_ix + len]);
            insert_ix += len;
        }
        for i in 0..self.wrt_biases.len() {
            let len = self.wrt_biases[i].elements();
            self.wrt_biases[i].host(&mut host_vec[insert_ix..insert_ix + len]);
            insert_ix += len;
        }
        host_vec
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct BranchCfg {
    pub(crate) num_params: usize,
    pub(crate) num_markers: usize,
    pub(crate) layer_widths: Vec<usize>,
    pub(crate) params: Vec<f32>,
    pub(crate) precisions: BranchPrecisions,
}

impl BranchCfg {
    pub fn params(&self) -> &Vec<f32> {
        &self.params
    }

    pub fn output_layer_weight(&self) -> f32 {
        *self.params.last().expect("Branch params are empty!")
    }

    pub fn precisions(&self) -> &BranchPrecisions {
        &self.precisions
    }

    pub fn set_output_layer_precision(&mut self, precision: f32) {
        self.precisions.set_output_layer_precision(precision);
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

// TODO: this should be used within BranchCfg,
// for now I just need this for convenient output
#[derive(Clone, Serialize)]
pub struct BranchMeta {
    pub(crate) num_params: usize,
    pub(crate) num_markers: usize,
    pub(crate) layer_widths: Vec<usize>,
    pub(crate) prior_shape: f32,
    pub(crate) prior_scale: f32,
}

impl BranchMeta {
    pub fn from_cfg(cfg: &BranchCfg, prior_shape: f32, prior_scale: f32) -> Self {
        Self {
            num_params: cfg.num_params,
            num_markers: cfg.num_markers,
            layer_widths: cfg.layer_widths.clone(),
            prior_shape,
            prior_scale,
        }
    }
}
