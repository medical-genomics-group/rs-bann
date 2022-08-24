use super::{
    super::mcmc_cfg::{MCMCCfg, StepSizeMode},
    branch_cfg_builder::BranchCfgBuilder,
    momenta::BranchMomenta,
    params::BranchHyperparams,
    params::BranchParams,
    step_sizes::StepSizes,
    trajectory::Trajectory,
};
use crate::{scalar_to_host, to_host};
use arrayfire::{diag_extract, dim4, dot, matmul, randu, sqrt, sum, tanh, Array, MatProp};
use log::{debug, warn};
use rand::{prelude::ThreadRng, Rng};
use serde::Serialize;
use serde_json::to_writer;
use std::{
    fs::File,
    io::{BufWriter, Write},
};

pub trait Branch {
    fn build_cfg(cfg_bld: BranchCfgBuilder) -> BranchCfg;

    fn from_cfg(cfg: &BranchCfg) -> Self;

    fn to_cfg(&self) -> BranchCfg;

    fn set_params(&mut self, params: &BranchParams);

    fn params(&self) -> &BranchParams;

    fn params_mut(&mut self) -> &mut BranchParams;

    fn hyperparams(&self) -> &BranchHyperparams;

    fn num_params(&self) -> usize;

    fn num_layers(&self) -> usize;

    fn layer_width(&self, index: usize) -> usize;

    fn set_error_precision(&mut self, val: f64);

    fn rng(&mut self) -> &mut ThreadRng;

    fn sample_precisions(&mut self, prior_shape: f64, prior_scale: f64);

    fn num_markers(&self) -> usize;

    fn layer_widths(&self) -> &Vec<usize>;

    fn log_density_gradient(
        &self,
        x_train: &Array<f64>,
        y_train: &Array<f64>,
    ) -> BranchLogDensityGradient;

    fn log_density(&self, params: &BranchParams, hyperparams: &BranchHyperparams, rss: f64) -> f64;

    // The difference in log density when all but one variable are changed
    // i.e. a partial update is done.
    // DO NOT run this in production code, this will be extremely slow.
    fn step_effects_on_ld(
        &mut self,
        prev_params: &BranchParams,
        x: &Array<f64>,
        y: &Array<f64>,
    ) -> Vec<f64> {
        let mut res = Vec::new();
        let prev_pv = prev_params.param_vec();
        let curr_pv = self.params().param_vec();
        let mut pv = curr_pv.clone();
        let lw = self.layer_widths().clone();
        let nm = self.num_markers();
        for pix in 0..self.num_params() {
            // exchange value
            pv[pix] = prev_pv[pix];
            // compute rss, ld
            self.params_mut().load_param_vec(&pv, &lw, nm);
            let rss = self.rss(x, y);
            res.push(self.log_density(self.params(), self.hyperparams(), rss));
            // put param back
            pv[pix] = curr_pv[pix];
        }
        self.params_mut().load_param_vec(&curr_pv, &lw, nm);
        res
    }

    fn weights(&self, index: usize) -> &Array<f64> {
        &self.params().weights[index]
    }

    fn biases(&self, index: usize) -> &Array<f64> {
        &self.params().biases[index]
    }

    fn weight_precisions(&self, index: usize) -> &Array<f64> {
        &self.hyperparams().weight_precisions[index]
    }

    fn bias_precision(&self, index: usize) -> f64 {
        self.hyperparams().bias_precisions[index]
    }

    fn error_precision(&self) -> f64 {
        self.hyperparams().error_precision
    }

    fn is_accepted(&mut self, acceptance_probability: f64) -> bool {
        self.rng().gen_range(0.0..1.0) < acceptance_probability
    }

    /// Quantify change of distance from starting point
    fn net_movement(&self, init_params: &BranchParams, momenta: &BranchMomenta) -> f64 {
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
            wrt_weights.push(arrayfire::randn::<f64>(self.weights(index).dims()));
            wrt_biases.push(arrayfire::randn::<f64>(self.biases(index).dims()));
        }
        // output layer weight momentum
        wrt_weights.push(arrayfire::randn::<f64>(
            self.weights(self.num_layers() - 1).dims(),
        ));
        BranchMomenta {
            wrt_weights,
            wrt_biases,
        }
    }

    fn random_step_sizes(&mut self, const_factor: f64) -> StepSizes {
        let mut wrt_weights = Vec::with_capacity(self.num_layers());
        let mut wrt_biases = Vec::with_capacity(self.num_layers() - 1);
        let prop_factor = (self.num_params() as f64).powf(-0.25) * const_factor;

        for index in 0..self.num_layers() {
            wrt_weights.push(randu::<f64>(self.weights(index).dims()) * prop_factor);
        }

        for index in 0..(self.num_layers() - 1) {
            wrt_biases.push(randu::<f64>(self.biases(index).dims()) * prop_factor);
        }
        StepSizes {
            wrt_weights,
            wrt_biases,
        }
    }

    fn uniform_step_sizes(&self, val: f64) -> StepSizes {
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
    fn std_scaled_step_sizes(&self, const_factor: f64) -> StepSizes;

    fn izmailov_step_sizes(&mut self, integration_length: usize) -> StepSizes {
        let mut wrt_weights = Vec::with_capacity(self.num_layers());
        let mut wrt_biases = Vec::with_capacity(self.num_layers() - 1);

        for index in 0..self.num_layers() {
            wrt_weights.push(
                std::f64::consts::PI
                    / (2.
                        * sqrt(&self.hyperparams().weight_precisions[index])
                        * integration_length as f64),
            );
        }

        for index in 0..self.num_layers() - 1 {
            wrt_biases.push(Array::new(
                &vec![
                    std::f64::consts::PI
                        / (2.
                            * &self.hyperparams().bias_precisions[index].sqrt()
                            * integration_length as f64);
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

    fn forward_feed(&self, x_train: &Array<f64>) -> Vec<Array<f64>> {
        let mut activations: Vec<Array<f64>> = Vec::with_capacity(self.num_layers() - 1);
        activations.push(self.mid_layer_activation(0, x_train));
        for layer_index in 1..self.num_layers() - 1 {
            activations.push(self.mid_layer_activation(layer_index, activations.last().unwrap()));
        }
        activations.push(self.output_neuron_activation(activations.last().unwrap()));
        activations
    }

    fn mid_layer_activation(&self, layer_index: usize, input: &Array<f64>) -> Array<f64> {
        let xw = matmul(
            input,
            self.weights(layer_index),
            MatProp::NONE,
            MatProp::NONE,
        );
        let bias_m = &arrayfire::tile(
            self.biases(layer_index),
            dim4!(input.dims().get()[0], 1, 1, 1),
        );
        tanh(&(xw + bias_m))
    }

    fn output_neuron_activation(&self, input: &Array<f64>) -> Array<f64> {
        matmul(
            input,
            self.weights(self.num_layers() - 1),
            MatProp::NONE,
            MatProp::NONE,
        )
    }

    fn backpropagate(
        &self,
        x_train: &Array<f64>,
        y_train: &Array<f64>,
    ) -> (Vec<Array<f64>>, Vec<Array<f64>>) {
        // forward propagate to get signals
        let activations = self.forward_feed(x_train);

        let mut bias_gradient: Vec<Array<f64>> = Vec::with_capacity(self.num_layers() - 1);
        let mut weights_gradient: Vec<Array<f64>> = Vec::with_capacity(self.num_layers());
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
            let delta: Array<f64> = (1 - arrayfire::pow(activation, &2, false)) * error;
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

        let delta: Array<f64> = (1 - arrayfire::pow(&activations[0], &2, false)) * error;
        bias_gradient.push(arrayfire::sum(&delta, 0));
        weights_gradient.push(arrayfire::transpose(
            &matmul(&delta, x_train, MatProp::TRANS, MatProp::NONE),
            false,
        ));

        bias_gradient.reverse();
        weights_gradient.reverse();

        (weights_gradient, bias_gradient)
    }

    // this is -H = (-U) + (-K)
    fn neg_hamiltonian(&self, momenta: &BranchMomenta, x: &Array<f64>, y: &Array<f64>) -> f64 {
        self.log_density(self.params(), self.hyperparams(), self.rss(x, y)) + momenta.log_density()
    }

    fn rss(&self, x: &Array<f64>, y: &Array<f64>) -> f64 {
        let r = self.forward_feed(&x).last().unwrap() - y;
        arrayfire::sum_all(&(&r * &r)).0
    }

    fn predict(&self, x: &Array<f64>) -> Array<f64> {
        self.forward_feed(x).last().unwrap().copy()
    }

    /// Takes a single parameter sample using HMC.
    /// Returns `false` if final state is rejected, `true` if accepted.
    fn hmc_step(
        &mut self,
        x_train: &Array<f64>,
        y_train: &Array<f64>,
        mcmc_cfg: &MCMCCfg,
    ) -> HMCStepResult {
        let mut seld_file = None;
        let mut seld = Trajectory::new();
        let mut prev_params = None;
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
        if mcmc_cfg.seld {
            seld_file = Some(BufWriter::new(
                File::options()
                    .append(true)
                    .create(true)
                    .open(mcmc_cfg.seld_path())
                    .unwrap(),
            ));
            prev_params = Some(self.params().clone());
        }

        let mut u_turned = false;
        let init_params = self.params().clone();
        let step_sizes = match mcmc_cfg.hmc_step_size_mode {
            StepSizeMode::StdScaled => self.std_scaled_step_sizes(mcmc_cfg.hmc_step_size_factor),
            StepSizeMode::Random => self.random_step_sizes(mcmc_cfg.hmc_step_size_factor),
            StepSizeMode::Uniform => self.uniform_step_sizes(mcmc_cfg.hmc_step_size_factor),
            StepSizeMode::Izmailov => self.izmailov_step_sizes(mcmc_cfg.hmc_integration_length),
        };
        let mut momenta = self.sample_momenta();
        let init_neg_hamiltonian = self.neg_hamiltonian(&momenta, x_train, y_train);
        debug!("Starting hmc step");
        debug!("initial hamiltonian: {:?}", init_neg_hamiltonian);
        let mut ldg = self.log_density_gradient(x_train, y_train);

        // leapfrog
        for _step in 0..(mcmc_cfg.hmc_integration_length) {
            momenta.half_step(&step_sizes, &ldg);
            self.params_mut().full_step(&step_sizes, &momenta);
            ldg = self.log_density_gradient(x_train, y_train);

            momenta.half_step(&step_sizes, &ldg);

            // diagnostics and logging

            if mcmc_cfg.trajectories {
                traj.add(self.params().param_vec());
            }

            if mcmc_cfg.seld {
                seld.add(self.step_effects_on_ld(prev_params.as_ref().unwrap(), x_train, y_train));
            }

            let curr_neg_hamiltonian = self.neg_hamiltonian(&momenta, x_train, y_train);
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

                if mcmc_cfg.seld {
                    to_writer(seld_file.as_mut().unwrap(), &seld).unwrap();
                    seld_file.as_mut().unwrap().write_all(b"\n").unwrap();
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

        if mcmc_cfg.seld {
            to_writer(seld_file.as_mut().unwrap(), &seld).unwrap();
            seld_file.as_mut().unwrap().write_all(b"\n").unwrap();
        }

        debug!("final gradients");
        for lix in 0..ldg.wrt_weights.len() {
            debug!(
                "layer: {:}; weight grad: {:?}",
                lix,
                to_host(&ldg.wrt_weights[lix])
            );
        }
        for lix in 0..ldg.wrt_biases.len() {
            debug!(
                "layer: {:}; bias grad: {:?}",
                lix,
                to_host(&ldg.wrt_biases[lix])
            );
        }

        // accept or reject
        let final_neg_hamiltonian = self.neg_hamiltonian(&momenta, x_train, y_train);
        let log_acc_probability = final_neg_hamiltonian - init_neg_hamiltonian;
        let acc_probability = if log_acc_probability >= 0. {
            1.
        } else {
            log_acc_probability.exp()
        };
        if self.is_accepted(acc_probability) {
            debug!("accepted state with acc prob: {:?}", acc_probability);
            HMCStepResult::Accepted
        } else {
            debug!("rejected state with acc prob: {:?}", acc_probability);
            self.set_params(&init_params);
            HMCStepResult::Rejected
        }
    }
}

/// Gradients of the log density w.r.t. the network parameters.
#[derive(Clone)]
pub struct BranchLogDensityGradient {
    pub wrt_weights: Vec<Array<f64>>,
    pub wrt_biases: Vec<Array<f64>>,
}

impl BranchLogDensityGradient {
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

    pub(crate) fn param_vec(&self) -> Vec<f64> {
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

#[derive(Clone, Serialize)]
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

// TODO: this should be used within BranchCfg,
// for now I just need this for convenient output
#[derive(Clone, Serialize)]
pub struct BranchMeta {
    pub(crate) num_params: usize,
    pub(crate) num_markers: usize,
    pub(crate) layer_widths: Vec<usize>,
}

impl BranchMeta {
    pub fn from_cfg(cfg: &BranchCfg) -> Self {
        Self {
            num_params: cfg.num_params,
            num_markers: cfg.num_markers,
            layer_widths: cfg.layer_widths.clone(),
        }
    }
}
