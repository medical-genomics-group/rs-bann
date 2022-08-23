use super::{
    super::mcmc_cfg::{MCMCCfg, StepSizeMode},
    branch_cfg_builder::BranchCfgBuilder,
    momenta::BranchMomenta,
    params::BranchHyperparams,
    params::BranchParams,
    step_sizes::StepSizes,
};
use crate::{net::branch::micro_trace::MicroTrace, scalar_to_host, to_host};
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

    fn layer_widths(&self, index: usize) -> usize;

    fn set_error_precision(&mut self, val: f64);

    fn rng(&mut self) -> &mut ThreadRng;

    fn sample_precisions(&mut self, prior_shape: f64, prior_scale: f64);

    fn log_density_gradient(
        &self,
        x_train: &Array<f64>,
        y_train: &Array<f64>,
    ) -> BranchLogDensityGradient;

    fn log_density(&self, params: &BranchParams, hyperparams: &BranchHyperparams, rss: f64) -> f64;

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
                &momenta.wrt_biases(ix),
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
        let mut write_trace = false;
        let mut trace_file = None;
        let mut trace = MicroTrace::new();
        if let Some(file) = &mcmc_cfg.micro_trace_file {
            trace_file = Some(BufWriter::new(File::create(file).unwrap()));
            write_trace = true;
        }

        let mut u_turned = false;
        let init_params = self.params().clone();
        let step_sizes = match mcmc_cfg.hmc_step_size_mode {
            StepSizeMode::StdScaled => self.std_scaled_step_sizes(mcmc_cfg.hmc_step_size_factor),
            StepSizeMode::Random => self.random_step_sizes(mcmc_cfg.hmc_step_size_factor),
            StepSizeMode::Uniform => self.uniform_step_sizes(mcmc_cfg.hmc_step_size_factor),
            StepSizeMode::Izmailov => self.izmailov_step_sizes(mcmc_cfg.hmc_integration_length),
        };

        let init_momenta = self.sample_momenta();
        let init_neg_hamiltonian = self.neg_hamiltonian(&init_momenta, x_train, y_train);
        debug!("Starting hmc step");
        debug!("initial hamiltonian: {:?}", init_neg_hamiltonian);
        let mut momenta = init_momenta.clone();
        let mut ldg = self.log_density_gradient(x_train, y_train);

        // leapfrog
        for _step in 0..(mcmc_cfg.hmc_integration_length) {
            momenta.half_step(&step_sizes, &ldg);
            self.params_mut().full_step(&step_sizes, &momenta);
            ldg = self.log_density_gradient(x_train, y_train);

            momenta.half_step(&step_sizes, &ldg);

            // diagnostics and logging

            if (self.neg_hamiltonian(&momenta, x_train, y_train) - init_neg_hamiltonian).abs()
                > mcmc_cfg.hmc_max_hamiltonian_error
            {
                debug!("hamiltonian error threshold crossed: terminating");
                self.set_params(&init_params);
                return HMCStepResult::RejectedEarly;
            }

            if !u_turned && self.is_u_turn(&init_params, &momenta) {
                warn!("U turn in HMC trajectory at step {}", _step);
                u_turned = true;
            }

            if write_trace {
                trace.add(self.params().param_vec());
            }
        }

        if write_trace {
            to_writer(trace_file.as_mut().unwrap(), &trace).unwrap();
            trace_file.as_mut().unwrap().write(b"\n").unwrap();
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
