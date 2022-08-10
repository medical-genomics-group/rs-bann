use super::{
    super::{
        gibbs_steps::multi_param_precision_posterior,
        mcmc_cfg::{MCMCCfg, StepSizeMode},
    },
    momenta::BranchMomenta,
    params::{BranchHyperparams, BranchParams},
    step_sizes::StepSizes,
};
use crate::{scalar_to_host, to_host};
use arrayfire::{diag_extract, dim4, dot, matmul, sum, tanh, Array, MatProp};
use log::{debug, warn};
use rand::prelude::ThreadRng;
use rand::{thread_rng, Rng};
use rand_distr::Gamma;

pub enum HMCStepResult {
    RejectedEarly,
    Rejected,
    Accepted,
}

#[derive(Clone)]
pub struct BranchCfg {
    pub(crate) num_params: usize,
    pub(crate) num_markers: usize,
    pub(crate) layer_widths: Vec<usize>,
    pub(crate) params: Vec<f64>,
    pub(crate) hyperparams: BranchHyperparams,
}

/// Gradients of the log density w.r.t. the network parameters.
#[derive(Clone)]
pub(crate) struct BranchLogDensityGradient {
    pub wrt_weights: Vec<Array<f64>>,
    pub wrt_biases: Vec<Array<f64>>,
}

pub struct Branch {
    pub(crate) num_params: usize,
    pub(crate) num_markers: usize,
    pub(crate) params: BranchParams,
    pub(crate) hyperparams: BranchHyperparams,
    pub(crate) layer_widths: Vec<usize>,
    pub(crate) num_layers: usize,
    pub(crate) rng: ThreadRng,
}

impl Branch {
    /// Creates Branch on device with BranchCfg from host memory.
    pub fn from_cfg(cfg: &BranchCfg) -> Self {
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
    pub fn to_cfg(&self) -> BranchCfg {
        BranchCfg {
            num_params: self.num_params,
            num_markers: self.num_markers,
            layer_widths: self.layer_widths.clone(),
            params: self.params.param_vec(),
            hyperparams: self.hyperparams.clone(),
        }
    }

    pub fn num_params(&self) -> usize {
        self.num_params
    }

    pub fn num_markers(&self) -> usize {
        self.num_markers
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    pub fn weights(&self, index: usize) -> &Array<f64> {
        &self.params.weights[index]
    }

    pub fn biases(&self, index: usize) -> &Array<f64> {
        &self.params.biases[index]
    }

    pub fn weight_precision(&self, index: usize) -> f64 {
        self.hyperparams.weight_precisions[index]
    }

    pub fn bias_precision(&self, index: usize) -> f64 {
        self.hyperparams.bias_precisions[index]
    }

    pub fn error_precision(&self) -> f64 {
        self.hyperparams.error_precision
    }

    pub fn layer_widths(&self, index: usize) -> usize {
        self.layer_widths[index]
    }

    pub fn set_error_precision(&mut self, val: f64) {
        self.hyperparams.error_precision = val;
    }

    pub fn rss(&self, x: &Array<f64>, y: &Array<f64>) -> f64 {
        let r = self.forward_feed(&x).last().unwrap() - y;
        arrayfire::sum_all(&(&r * &r)).0
    }

    pub fn predict(&self, x: &Array<f64>) -> Array<f64> {
        self.forward_feed(x).last().unwrap().copy()
    }

    /// Samples precision values from their posterior distribution in a Gibbs step.
    pub fn sample_precisions(&mut self, prior_shape: f64, prior_scale: f64) {
        for i in 0..self.params.weights.len() {
            self.hyperparams.weight_precisions[i] = multi_param_precision_posterior(
                prior_shape,
                prior_scale,
                &self.params.weights[i],
                &mut self.rng,
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

    /// Takes a single parameter sample using HMC.
    /// Returns `false` if final state is rejected, `true` if accepted.
    pub fn hmc_step(
        &mut self,
        x_train: &Array<f64>,
        y_train: &Array<f64>,
        mcmc_cfg: &MCMCCfg,
    ) -> HMCStepResult {
        let init_params = self.params.clone();
        let step_sizes = match mcmc_cfg.hmc_step_size_mode {
            StepSizeMode::StdScaled => self.std_scaled_step_sizes(mcmc_cfg.hmc_step_size_factor),
            StepSizeMode::Random => self.random_step_sizes(mcmc_cfg.hmc_step_size_factor),
            StepSizeMode::Uniform => self.uniform_step_sizes(mcmc_cfg.hmc_step_size_factor),
        };

        // TODO: add u turn diagnostic for tuning
        let init_momenta = self.sample_momenta();
        let init_neg_hamiltonian = self.neg_hamiltonian(&init_momenta, x_train, y_train);
        debug!("Starting hmc step");
        debug!("initial hamiltonian: {:?}", init_neg_hamiltonian);
        let mut momenta = init_momenta.clone();
        let mut ldg = self.log_density_gradient(x_train, y_train);

        // leapfrog
        for _step in 0..(mcmc_cfg.hmc_integration_length) {
            momenta.half_step(&step_sizes, &ldg);
            self.params.full_step(&step_sizes, &momenta);
            ldg = self.log_density_gradient(x_train, y_train);

            momenta.half_step(&step_sizes, &ldg);

            if (self.neg_hamiltonian(&momenta, x_train, y_train) - init_neg_hamiltonian).abs()
                > mcmc_cfg.hmc_max_hamiltonian_error
            {
                debug!("hamiltonian error threshold crossed: terminating");
                self.params = init_params;
                return HMCStepResult::RejectedEarly;
            }

            if self.is_u_turn(&init_params, &momenta) {
                warn!("U turn in HMC trajectory at step {}", _step);
            }
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
            debug!("layer: {:}; {:?}", lix, to_host(&ldg.wrt_biases[lix]));
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
            self.params = init_params;
            HMCStepResult::Rejected
        }
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

    // this is -H = (-U) + (-K)
    fn neg_hamiltonian(&self, momenta: &BranchMomenta, x: &Array<f64>, y: &Array<f64>) -> f64 {
        self.params.log_density(&self.hyperparams, self.rss(x, y)) + momenta.log_density()
    }

    fn is_accepted(&mut self, acceptance_probability: f64) -> bool {
        self.rng.gen_range(0.0..1.0) < acceptance_probability
    }

    fn sample_momenta(&self) -> BranchMomenta {
        let mut wrt_weights = Vec::with_capacity(self.num_layers);
        let mut wrt_biases = Vec::with_capacity(self.num_layers - 1);
        for index in 0..self.num_layers - 1 {
            wrt_weights.push(arrayfire::randn::<f64>(self.weights(index).dims()));
            wrt_biases.push(arrayfire::randn::<f64>(self.biases(index).dims()));
        }
        // output layer weight momentum
        wrt_weights.push(arrayfire::randn::<f64>(
            self.weights(self.num_layers - 1).dims(),
        ));
        BranchMomenta {
            wrt_weights,
            wrt_biases,
        }
    }

    fn random_step_sizes(&mut self, const_factor: f64) -> StepSizes {
        let mut wrt_weights = Vec::with_capacity(self.num_layers);
        let mut wrt_biases = Vec::with_capacity(self.num_layers - 1);
        let gamma = Gamma::new(0.25, 0.5).unwrap();
        let prop_factor = (self.num_params as f64).powf(-0.25) * const_factor;
        for index in 0..self.num_layers - 1 {
            let n = self.weights(index).elements();
            wrt_weights.push(
                Array::new(
                    (&mut self.rng)
                        .sample_iter(gamma)
                        .take(n)
                        .collect::<Vec<f64>>()
                        .as_slice(),
                    self.weights(index).dims(),
                ) * prop_factor,
            );
            let n = self.biases(index).elements();
            wrt_biases.push(
                Array::new(
                    (&mut self.rng)
                        .sample_iter(gamma)
                        .take(n)
                        .collect::<Vec<f64>>()
                        .as_slice(),
                    self.biases(index).dims(),
                ) * prop_factor,
            );
        }
        // output layer weights
        let n = self.weights(self.num_layers - 1).elements();
        wrt_weights.push(
            Array::new(
                (&mut self.rng)
                    .sample_iter(gamma)
                    .take(n)
                    .collect::<Vec<f64>>()
                    .as_slice(),
                self.weights(self.num_layers - 1).dims(),
            ) * prop_factor,
        );
        StepSizes {
            wrt_weights,
            wrt_biases,
        }
    }

    fn uniform_step_sizes(&self, val: f64) -> StepSizes {
        let mut wrt_weights = Vec::with_capacity(self.num_layers);
        let mut wrt_biases = Vec::with_capacity(self.num_layers - 1);
        for index in 0..self.num_layers - 1 {
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
            &vec![val; self.weights(self.num_layers - 1).elements()],
            self.weights(self.num_layers - 1).dims(),
        ));
        StepSizes {
            wrt_weights,
            wrt_biases,
        }
    }

    /// Sets step sizes proportional to the prior standard deviation of each parameter.
    fn std_scaled_step_sizes(&self, const_factor: f64) -> StepSizes {
        let mut wrt_weights = Vec::with_capacity(self.num_layers);
        let mut wrt_biases = Vec::with_capacity(self.num_layers - 1);
        for index in 0..self.num_layers - 1 {
            wrt_weights.push(Array::new(
                &vec![
                    const_factor * (1. / self.weight_precision(index)).sqrt();
                    self.weights(index).elements()
                ],
                self.weights(index).dims(),
            ));
            wrt_biases.push(Array::new(
                &vec![
                    const_factor * (1. / self.bias_precision(index)).sqrt();
                    self.biases(index).elements()
                ],
                self.biases(index).dims(),
            ));
        }
        // output layer weights
        wrt_weights.push(Array::new(
            &vec![
                const_factor * (1. / self.weight_precision(self.num_layers - 1)).sqrt();
                self.weights(self.num_layers - 1).elements()
            ],
            self.weights(self.num_layers - 1).dims(),
        ));
        StepSizes {
            wrt_weights,
            wrt_biases,
        }
    }

    fn forward_feed(&self, x_train: &Array<f64>) -> Vec<Array<f64>> {
        let mut activations: Vec<Array<f64>> = Vec::with_capacity(self.num_layers - 1);
        activations.push(self.mid_layer_activation(0, x_train));
        for layer_index in 1..self.num_layers - 1 {
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
            self.weights(self.num_layers - 1),
            MatProp::NONE,
            MatProp::NONE,
        )
    }

    pub fn backpropagate(
        &self,
        x_train: &Array<f64>,
        y_train: &Array<f64>,
    ) -> (Vec<Array<f64>>, Vec<Array<f64>>) {
        // forward propagate to get signals
        let activations = self.forward_feed(x_train);

        let mut bias_gradient: Vec<Array<f64>> = Vec::with_capacity(self.num_layers - 1);
        let mut weights_gradient: Vec<Array<f64>> = Vec::with_capacity(self.num_layers);
        // back propagate
        let mut activation = activations.last().unwrap();

        // TODO: factor of 2 might be necessary here?
        let mut error = activation - y_train;
        weights_gradient.push(arrayfire::dot(
            &error,
            &activations[self.num_layers - 2],
            MatProp::NONE,
            MatProp::NONE,
        ));
        error = matmul(
            &error,
            self.weights(self.num_layers - 1),
            MatProp::NONE,
            MatProp::NONE,
        );

        for layer_index in (1..self.num_layers - 1).rev() {
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

    fn log_density_gradient(
        &self,
        x_train: &Array<f64>,
        y_train: &Array<f64>,
    ) -> BranchLogDensityGradient {
        let (d_rss_wrt_weights, d_rss_wrt_biases) = self.backpropagate(x_train, y_train);
        let mut ldg_wrt_weights: Vec<Array<f64>> = Vec::with_capacity(self.num_layers);
        let mut ldg_wrt_biases: Vec<Array<f64>> = Vec::with_capacity(self.num_layers - 1);
        for layer_index in 0..self.num_layers - 1 {
            ldg_wrt_weights.push(
                -self.weight_precision(layer_index) * self.weights(layer_index)
                    - self.error_precision() * &d_rss_wrt_weights[layer_index],
            );
            ldg_wrt_biases.push(
                -self.bias_precision(layer_index) * self.biases(layer_index)
                    - self.error_precision() * &d_rss_wrt_biases[layer_index],
            );
        }
        // output layer gradient
        ldg_wrt_weights.push(
            -self.weight_precision(self.num_layers - 1) * self.weights(self.num_layers - 1)
                - self.error_precision() * &d_rss_wrt_weights[self.num_layers - 1],
        );
        BranchLogDensityGradient {
            wrt_weights: ldg_wrt_weights,
            wrt_biases: ldg_wrt_biases,
        }
    }
}

#[cfg(test)]
mod tests {
    use arrayfire::{dim4, Array};
    // use arrayfire::{af_print, randu};

    use super::super::branch_builder::BranchBuilder;
    use super::Branch;

    use crate::net::branch::momenta::BranchMomenta;
    use crate::net::branch::params::BranchParams;
    use crate::to_host;

    // #[test]
    // fn test_af() {
    //     let num_rows: u64 = 5;
    //     let num_cols: u64 = 3;
    //     let dims = Dim4::new(&[num_rows, num_cols, 1, 1]);
    //     let a = randu::<f64>(dims);
    //     af_print!("Create a 5-by-3 matrix of random floats on the GPU", a);
    // }

    fn make_test_branch() -> Branch {
        let exp_weights = [
            Array::new(&[0., 1., 2., 3., 4., 5.], dim4![3, 2, 1, 1]),
            Array::new(&[1., 2.], dim4![2, 1, 1, 1]),
            Array::new(&[2.], dim4![1, 1, 1, 1]),
        ];
        let exp_biases = [
            Array::new(&[0., 1.], dim4![1, 2, 1, 1]),
            Array::new(&[2.], dim4![1, 1, 1, 1]),
        ];

        BranchBuilder::new()
            .with_num_markers(3)
            .add_hidden_layer(2)
            .add_layer_biases(&exp_biases[0])
            .add_layer_weights(&exp_weights[0])
            .add_summary_weights(&exp_weights[1])
            .add_summary_bias(&exp_biases[1])
            .add_output_weight(&exp_weights[2])
            .build()
    }

    fn make_test_uniform_params(c: f64) -> BranchParams {
        let weights = [
            Array::new(&[c; 6], dim4![3, 2, 1, 1]),
            Array::new(&[c; 2], dim4![2, 1, 1, 1]),
            Array::new(&[c], dim4![1, 1, 1, 1]),
        ]
        .to_vec();
        let biases = [
            Array::new(&[c; 2], dim4![1, 2, 1, 1]),
            Array::new(&[c], dim4![1, 1, 1, 1]),
        ]
        .to_vec();
        BranchParams { weights, biases }
    }

    fn make_test_uniform_momenta(c: f64) -> BranchMomenta {
        let wrt_weights = [
            Array::new(&[c; 6], dim4![3, 2, 1, 1]),
            Array::new(&[c; 2], dim4![2, 1, 1, 1]),
            Array::new(&[c], dim4![1, 1, 1, 1]),
        ]
        .to_vec();
        let wrt_biases = [
            Array::new(&[c; 2], dim4![1, 2, 1, 1]),
            Array::new(&[c], dim4![1, 1, 1, 1]),
        ]
        .to_vec();
        BranchMomenta {
            wrt_weights,
            wrt_biases,
        }
    }

    #[test]
    fn test_forward_feed() {
        let num_individuals = 4;
        let num_markers = 3;
        let branch = make_test_branch();
        let x_train: Array<f64> = Array::new(
            &[1., 0., 0., 2., 1., 1., 2., 0., 0., 2., 0., 1.],
            dim4![num_individuals, num_markers, 1, 1],
        );
        let activations = branch.forward_feed(&x_train);

        // correct number of activations
        assert_eq!(activations.len(), branch.num_layers);

        // correct dimensions of activations
        for i in 0..(branch.num_layers) {
            println!("{:?}", i);
            assert_eq!(
                activations[i].dims(),
                dim4![num_individuals, branch.layer_widths[i] as u64, 1, 1]
            );
        }

        let exp_activations: Vec<Array<f64>> = vec![
            Array::new(
                &[
                    0.7615941559557649,
                    0.9999092042625951,
                    0.9640275800758169,
                    0.9640275800758169,
                    0.9999997749296758,
                    0.9999999999998128,
                    0.999999969540041,
                    0.9999999999244973,
                ],
                dim4![4, 2, 1, 1],
            ),
            Array::new(
                &[
                    0.9998537383423458,
                    0.9999091877741149,
                    0.9999024315761632,
                    0.999902431588021,
                ],
                dim4![4, 1, 1, 1],
            ),
            Array::new(
                &[
                    1.9997074766846916,
                    1.9998183755482297,
                    1.9998048631523264,
                    1.999804863176042,
                ],
                dim4![4, 1, 1, 1],
            ),
        ];
        // correct values of activations
        for i in 0..(branch.num_layers) {
            println!("{:?}", i);
            assert_eq!(to_host(&activations[i]), to_host(&exp_activations[i]));
        }
    }

    #[test]
    fn test_backpropagation() {
        let num_individuals = 4;
        let num_markers = 3;
        let branch = make_test_branch();
        let x_train: Array<f64> = Array::new(
            &[1., 0., 0., 2., 1., 1., 2., 0., 0., 2., 0., 1.],
            dim4![num_individuals, num_markers, 1, 1],
        );
        let y_train: Array<f64> = Array::new(&[0.0, 2.0, 1.0, 1.5], dim4![4, 1, 1, 1]);
        let (weights_gradient, bias_gradient) = branch.backpropagate(&x_train, &y_train);

        // correct number of gradients
        assert_eq!(weights_gradient.len(), branch.num_layers);
        assert_eq!(bias_gradient.len(), branch.num_layers - 1);

        // correct dimensions of gradients
        for i in 0..(branch.num_layers) {
            println!("{:?}", i);
            assert_eq!(weights_gradient[i].dims(), branch.weights(i).dims());
        }
        for i in 0..(branch.num_layers - 1) {
            assert_eq!(bias_gradient[i].dims(), branch.biases(i).dims());
        }

        let exp_weight_grad = [
            Array::new(
                &[
                    0.0005188623902535914,
                    0.0005464341949822559,
                    1.3780500770415134e-5,
                    1.0532996754298074e-9,
                    1.148260428514749e-9,
                    5.890746731184353e-14,
                ],
                dim4![3, 2, 1, 1],
            ),
            Array::new(
                &[0.0014550522522557225, 0.0017549999714042658],
                dim4![2, 1, 1, 1],
            ),
            Array::new(&[3.4986967999732057], dim4![1, 1, 1, 1]),
        ];

        let exp_bias_grad = [
            Array::new(
                &[0.0005326482866282294, 1.1007800519475804e-9],
                dim4![2, 1, 1, 1],
            ),
            Array::new(&[0.0017550002465993087], dim4![1, 1, 1, 1]),
        ];

        // correct values of gradient
        for i in 0..(branch.num_layers) {
            assert_eq!(to_host(&weights_gradient[i]), to_host(&exp_weight_grad[i]));
        }
        for i in 0..(branch.num_layers - 1) {
            println!("{:?}", i);
            assert_eq!(to_host(&bias_gradient[i]), to_host(&exp_bias_grad[i]));
        }
    }

    #[test]
    fn test_log_density_gradient() {
        let num_individuals = 4;
        let num_markers = 3;
        let branch = make_test_branch();
        let x_train: Array<f64> = Array::new(
            &[1., 0., 0., 2., 1., 1., 2., 0., 0., 2., 0., 1.],
            dim4![num_individuals, num_markers, 1, 1],
        );
        let y_train: Array<f64> = Array::new(&[0.0, 2.0, 1.0, 1.5], dim4![4, 1, 1, 1]);
        let ldg = branch.log_density_gradient(&x_train, &y_train);

        // correct output length
        assert_eq!(ldg.wrt_weights.len(), branch.num_layers);
        assert_eq!(ldg.wrt_biases.len(), branch.num_layers - 1);

        // correct dimensions
        for i in 0..(branch.num_layers) {
            println!("{:?}", i);
            assert_eq!(ldg.wrt_weights[i].dims(), branch.weights(i).dims());
        }
        for i in 0..(branch.num_layers - 1) {
            assert_eq!(ldg.wrt_biases[i].dims(), branch.biases(i).dims());
        }

        let exp_ldg_wrt_w = [
            Array::new(
                &[
                    -0.0005188623902535914,
                    -1.0005464341949823,
                    -2.0000137805007703,
                    -3.0000000010532997,
                    -4.00000000114826,
                    -5.000000000000059,
                ],
                dim4![3, 2, 1, 1],
            ),
            Array::new(
                &[-1.0014550522522556, -2.0017549999714044],
                dim4![2, 1, 1, 1],
            ),
            Array::new(&[-5.498696799973206], dim4![1, 1, 1, 1]),
        ];

        let exp_ldg_wrt_b = [
            Array::new(
                &[-0.0005326482866282294, -1.0000000011007801],
                dim4![2, 1, 1, 1],
            ),
            Array::new(&[-2.0017550002465994], dim4![1, 1, 1, 1]),
        ];

        // correct values
        for i in 0..(branch.num_layers) {
            println!("{:?}", i);
            assert_eq!(to_host(&ldg.wrt_weights[i]), to_host(&exp_ldg_wrt_w[i]));
        }
        for i in 0..(branch.num_layers - 1) {
            assert_eq!(to_host(&ldg.wrt_biases[i]), to_host(&exp_ldg_wrt_b[i]));
        }
    }

    #[test]
    fn test_uniform_step_sizes() {
        let branch = make_test_branch();
        let val = 1.0;
        let step_sizes = branch.uniform_step_sizes(val);
        for i in 0..(branch.num_layers - 1) {
            let mut obs = to_host(&step_sizes.wrt_weights[i]);
            assert_eq!(obs, vec![val; obs.len()]);
            obs = to_host(&step_sizes.wrt_biases[i]);
            assert_eq!(obs, vec![val; obs.len()]);
        }
        let obs = to_host(&step_sizes.wrt_weights[branch.num_layers - 1]);
        assert_eq!(obs, vec![val; obs.len()]);
    }

    #[test]
    fn test_net_movement() {
        let branch = make_test_branch();
        let momenta = make_test_uniform_momenta(1.);
        let init_params = make_test_uniform_params(0.);
        assert!(branch.net_movement(&init_params, &momenta) > 0.0);
        let init_params = make_test_uniform_params(100.);
        assert!(branch.net_movement(&init_params, &momenta) < 0.0);
    }
}
