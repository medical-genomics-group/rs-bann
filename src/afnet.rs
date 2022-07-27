use arrayfire::{dim4, matmul, tanh, Array, MatProp};
use log::{debug, info};
use rand::prelude::ThreadRng;
use rand::{thread_rng, Rng};
use std::fmt;

/// Copy data from device to a host vector.
// TODO: this should not live in this module.
fn to_host(a: &Array<f64>) -> Vec<f64> {
    let mut buffer = Vec::<f64>::new();
    buffer.resize(a.elements(), 0.);
    a.host(&mut buffer);
    buffer
}

#[derive(Clone)]
struct StepSizes {
    wrt_weights: Vec<Array<f64>>,
    wrt_biases: Vec<Array<f64>>,
}

impl fmt::Debug for StepSizes {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.param_vec())
    }
}

impl StepSizes {
    fn param_vec(&self) -> Vec<f64> {
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

    // TODO: better to use some dual averaging scheme?
    // TODO: test this!
    // TODO: this will almost definitely violate reversibility and therefore detailed balance.
    // It might still work, but is not guaranteed to?
    fn update_with_second_derivative(
        &mut self,
        prev_momenta: &ArmMomenta,
        prev_gradients: &ArmLogDensityGradient,
        curr_gradients: &ArmLogDensityGradient,
    ) {
        for i in 0..self.wrt_weights.len() {
            // distance traveled
            let delta_t = &self.wrt_weights[i] * &prev_momenta.wrt_weights[i];
            // change in first derivative value
            let delta_f = &curr_gradients.wrt_weights[i] - &prev_gradients.wrt_weights[i];
            // TODO: the argument of the sqrt might need a negative sign
            self.wrt_weights[i] = 1 / (arrayfire::sqrt(&(delta_f / delta_t)));
        }
        for i in 0..self.wrt_biases.len() {
            // distance traveled
            let delta_t = &self.wrt_biases[i] * &prev_momenta.wrt_biases[i];
            // change in first derivative value
            let delta_f = &curr_gradients.wrt_biases[i] - &prev_gradients.wrt_biases[i];
            self.wrt_biases[i] = 1 / (arrayfire::sqrt(&(delta_f / delta_t)));
        }
    }
}

#[derive(Clone)]
struct ArmMomenta {
    wrt_weights: Vec<Array<f64>>,
    wrt_biases: Vec<Array<f64>>,
}

impl ArmMomenta {
    fn half_step(&mut self, step_sizes: &StepSizes, grad: &ArmLogDensityGradient) {
        for i in 0..self.wrt_weights.len() {
            self.wrt_weights[i] += &step_sizes.wrt_weights[i] * 0.5 * &grad.wrt_weights[i];
        }
        for i in 0..self.wrt_biases.len() {
            self.wrt_biases[i] += &step_sizes.wrt_biases[i] * 0.5 * &grad.wrt_biases[i];
        }
    }

    fn full_step(&mut self, step_sizes: &StepSizes, grad: &ArmLogDensityGradient) {
        for i in 0..self.wrt_weights.len() {
            self.wrt_weights[i] += &step_sizes.wrt_weights[i] * &grad.wrt_weights[i];
        }
        for i in 0..self.wrt_biases.len() {
            self.wrt_biases[i] += &step_sizes.wrt_biases[i] * &grad.wrt_biases[i];
        }
    }

    fn log_density(&self) -> f64 {
        let mut log_density: f64 = 0.;
        for i in 0..self.wrt_weights.len() {
            log_density += arrayfire::sum_all(&(&self.wrt_weights[i] * &self.wrt_weights[i])).0;
        }
        for i in 0..self.wrt_biases.len() {
            log_density += arrayfire::sum_all(&(&self.wrt_biases[i] * &self.wrt_biases[i])).0;
        }
        log_density
    }
}

/// Weights and biases
#[derive(Clone)]
struct ArmParams {
    weights: Vec<Array<f64>>,
    biases: Vec<Array<f64>>,
}

impl fmt::Debug for ArmParams {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.param_vec())
    }
}

impl ArmParams {
    fn param_vec(&self) -> Vec<f64> {
        let mut host_vec = Vec::new();
        host_vec.resize(self.num_params(), 0.);
        let mut insert_ix: usize = 0;
        for i in 0..self.weights.len() {
            let len = self.weights[i].elements();
            self.weights[i].host(&mut host_vec[insert_ix..insert_ix + len]);
            insert_ix += len;
        }
        for i in 0..self.biases.len() {
            let len = self.biases[i].elements();
            self.biases[i].host(&mut host_vec[insert_ix..insert_ix + len]);
            insert_ix += len;
        }
        host_vec
    }

    fn num_params(&self) -> usize {
        let mut res: usize = 0;
        for i in 0..self.weights.len() {
            res += self.weights[i].elements();
        }
        for i in 0..self.biases.len() {
            res += self.biases[i].elements();
        }
        res
    }

    fn full_step(&mut self, step_sizes: &StepSizes, mom: &ArmMomenta) {
        for i in 0..self.weights.len() {
            self.weights[i] += &step_sizes.wrt_weights[i] * &mom.wrt_weights[i];
        }
        for i in 0..self.biases.len() {
            self.biases[i] += &step_sizes.wrt_biases[i] * &mom.wrt_biases[i];
        }
    }

    fn log_density(&self, hyperparams: &ArmHyperparams, rss: f64) -> f64 {
        let mut log_density: f64 = -0.5 * hyperparams.error_precision * rss;
        for i in 0..self.weights.len() {
            log_density -= hyperparams.weight_precisions[i]
                * 0.5
                * arrayfire::sum_all(&(&self.weights[i] * &self.weights[i])).0;
        }
        for i in 0..self.biases.len() {
            log_density -= hyperparams.bias_precisions[i]
                * 0.5
                * arrayfire::sum_all(&(&self.biases[i] * &self.biases[i])).0;
        }
        log_density
    }
}

struct ArmHyperparams {
    weight_precisions: Vec<f64>,
    bias_precisions: Vec<f64>,
    error_precision: f64,
}

/// Gradients of the log density w.r.t. the network parameters.
#[derive(Clone)]
struct ArmLogDensityGradient {
    wrt_weights: Vec<Array<f64>>,
    wrt_biases: Vec<Array<f64>>,
}

pub struct Arm {
    num_params: usize,
    num_markers: usize,
    params: ArmParams,
    hyperparams: ArmHyperparams,
    layer_widths: Vec<usize>,
    num_layers: usize,
    rng: ThreadRng,
    verbose: bool,
}

impl Arm {
    pub fn rss(&self, x: &Array<f64>, y: &Array<f64>) -> f64 {
        let r = self.forward_feed(&x).last().unwrap() - y;
        arrayfire::sum_all(&(&r * &r)).0
    }

    pub fn predict(&self, x: &Array<f64>) -> Array<f64> {
        self.forward_feed(x).last().unwrap().copy()
    }

    /// Take a single parameter sample using HMC.
    /// Return `false` if final state is rejected, `true` if accepted.
    pub fn hmc_step(
        &mut self,
        x_train: &Array<f64>,
        y_train: &Array<f64>,
        integration_length: usize,
        init_step_size: f64,
    ) -> bool {
        let init_params = self.params.clone();
        // TODO: add heuristic step sizes
        // for that I will need to keep last rounds gradient, step sizes and momenta
        let step_sizes = self.uniform_step_sizes(init_step_size);
        // TODO: add u turn diagnostic for tuning
        let init_momenta = self.sample_momenta();
        let init_neg_hamiltonian = self.neg_hamiltonian(&init_momenta, x_train, y_train);
        if self.verbose {
            debug!("Starting hmc step");
            debug!("initial hamiltonian: {:?}", init_neg_hamiltonian);
        }
        let mut momenta = init_momenta.clone();

        // integrate
        // initial half step
        momenta.half_step(&step_sizes, &self.log_density_gradient(x_train, y_train));
        // leapfrog
        for step in 0..(integration_length - 1) {
            self.params.full_step(&step_sizes, &momenta);
            momenta.full_step(&step_sizes, &self.log_density_gradient(x_train, y_train));
            if self.verbose {
                debug!(
                    "step: {:?}, hamiltonian: {:?}",
                    step,
                    self.neg_hamiltonian(&momenta, x_train, y_train)
                )
            }
        }
        // final steps for alignment
        self.params.full_step(&step_sizes, &momenta);
        momenta.half_step(&step_sizes, &self.log_density_gradient(x_train, y_train));

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
            true
        } else {
            debug!("rejected state with acc prob: {:?}", acc_probability);
            self.params = init_params;
            false
        }
    }

    fn weights(&self, index: usize) -> &Array<f64> {
        &self.params.weights[index]
    }

    fn biases(&self, index: usize) -> &Array<f64> {
        &self.params.biases[index]
    }

    fn weight_precision(&self, index: usize) -> f64 {
        self.hyperparams.weight_precisions[index]
    }

    fn bias_precision(&self, index: usize) -> f64 {
        self.hyperparams.bias_precisions[index]
    }

    fn error_precision(&self) -> f64 {
        self.hyperparams.error_precision
    }

    // this is -H = (-U) + (-K)
    fn neg_hamiltonian(&self, momenta: &ArmMomenta, x: &Array<f64>, y: &Array<f64>) -> f64 {
        self.params.log_density(&self.hyperparams, self.rss(x, y)) + momenta.log_density()
    }

    fn is_accepted(&mut self, acceptance_probability: f64) -> bool {
        self.rng.gen_range(0.0..1.0) < acceptance_probability
    }

    fn sample_momenta(&self) -> ArmMomenta {
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
        ArmMomenta {
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
    ) -> ArmLogDensityGradient {
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
        ArmLogDensityGradient {
            wrt_weights: ldg_wrt_weights,
            wrt_biases: ldg_wrt_biases,
        }
    }
}

pub struct ArmBuilder {
    num_params: usize,
    num_markers: usize,
    layer_widths: Vec<usize>,
    num_layers: usize,
    initial_weight_value: Option<f64>,
    initial_bias_value: Option<f64>,
    initial_random_range: f64,
    biases: Vec<Option<Array<f64>>>,
    weights: Vec<Option<Array<f64>>>,
    verbose: bool,
}

impl ArmBuilder {
    pub fn new() -> Self {
        Self {
            num_params: 0,
            num_markers: 0,
            layer_widths: vec![],
            // we always have a summary and an output node, so at least 2 layers.
            num_layers: 2,
            initial_weight_value: None,
            initial_bias_value: None,
            initial_random_range: 0.05,
            biases: vec![],
            weights: vec![],
            verbose: false,
        }
    }

    pub fn verbose(&mut self) -> &mut Self {
        self.verbose = true;
        self
    }

    pub fn with_num_markers(&mut self, num_markers: usize) -> &mut Self {
        self.num_markers = num_markers;
        self
    }

    pub fn add_hidden_layer(&mut self, layer_width: usize) -> &mut Self {
        self.layer_widths.push(layer_width);
        self.num_layers += 1;
        self.biases.push(None);
        self.weights.push(None);
        self
    }

    pub fn add_layer_biases(&mut self, biases: &Array<f64>) -> &mut Self {
        assert!(
            biases.dims().get()[0] as usize == 1,
            "bias vector dim 0 != 1, expected row vector"
        );
        assert!(
            biases.dims().get()[1] as usize == *self.layer_widths.last().unwrap(),
            "bias dim 1 does not match width of last added layer"
        );
        *self.biases.last_mut().unwrap() = Some(biases.copy());
        self
    }

    pub fn add_layer_weights(&mut self, weights: &Array<f64>) -> &mut Self {
        let wdims = *weights.dims().get();
        let expected_ncols = if self.num_layers > 3 {
            self.layer_widths[self.num_layers - 4]
        } else {
            self.num_markers
        };
        assert!(
            wdims[0] as usize == expected_ncols,
            "incorrect weight dims in dim 0"
        );
        assert!(
            wdims[1] as usize == self.layer_widths[self.num_layers - 3],
            "incorrect weight dims in dim 1"
        );
        *self.weights.last_mut().unwrap() = Some(weights.copy());
        self
    }

    pub fn add_summary_bias(&mut self, bias: &Array<f64>) -> &mut Self {
        let wdims = *bias.dims().get();
        assert!(
            wdims[0] as usize == 1,
            "incorrect summary bias dims in dim 0: has to be 1."
        );
        assert!(
            wdims[1] as usize == 1,
            "incorrect summary bias dims in dim 1: has to be 1."
        );
        self.biases.push(Some(bias.copy()));
        self
    }

    pub fn add_summary_weights(&mut self, weights: &Array<f64>) -> &mut Self {
        let wdims = *weights.dims().get();
        assert!(
            wdims[0] as usize == self.layer_widths[self.num_layers - 3],
            "incorrect summary weight dims in dim 0"
        );
        assert!(
            wdims[1] as usize == 1,
            "incorrect summary weight dims in dim 1: has to be 1."
        );
        self.weights.push(Some(weights.copy()));
        self
    }

    pub fn add_output_weight(&mut self, weights: &Array<f64>) -> &mut Self {
        let wdims = *weights.dims().get();
        assert!(
            wdims[0] as usize == 1,
            "incorrect output weight dims in dim 0: has to be 1."
        );
        assert!(
            wdims[1] as usize == 1,
            "incorrect output weight dims in dim 1: has to be 1."
        );
        self.weights.push(Some(weights.copy()));
        self
    }

    pub fn with_initial_random_range(&mut self, range: f64) -> &mut Self {
        self.initial_random_range = range;
        self
    }

    pub fn with_initial_weights_value(&mut self, value: f64) -> &mut Self {
        self.initial_weight_value = Some(value);
        self
    }

    pub fn with_initial_bias_value(&mut self, value: f64) -> &mut Self {
        self.initial_bias_value = Some(value);
        self
    }

    pub fn build(&mut self) -> Arm {
        let mut widths: Vec<usize> = vec![self.num_markers];
        // summary and output node
        self.layer_widths.push(1);
        self.layer_widths.push(1);
        widths.append(&mut self.layer_widths.clone());

        // get total number of params in network
        for i in 1..=self.num_layers {
            self.num_params += widths[i - 1] * widths[i] + widths[i];
        }
        // remove count for output bias
        self.num_params -= 1;

        // add None if weights or biases not added for last layers
        for _ in 0..self.num_layers - self.weights.len() {
            self.weights.push(None);
        }

        for _ in 0..self.num_layers - 1 - self.biases.len() {
            self.biases.push(None);
        }

        let mut weights: Vec<Array<f64>> = vec![];
        let mut biases: Vec<Array<f64>> = vec![];

        for index in 0..self.num_layers {
            if let Some(w) = &self.weights[index] {
                weights.push(w.copy());
            } else if let Some(v) = self.initial_weight_value {
                weights.push(arrayfire::constant!(
                    v;
                    widths[index] as u64,
                    widths[index + 1] as u64
                ));
            } else {
                let dims = dim4![widths[index] as u64, widths[index + 1] as u64, 1, 1];
                weights.push(
                    // this does not includes the bias term.
                    self.initial_random_range * arrayfire::randu::<f64>(dims)
                        - self.initial_random_range / 2f64,
                );
            }

            // we don't include the output neurons bias here
            if index == self.num_layers - 1 {
                break;
            }
            if let Some(b) = &self.biases[index] {
                biases.push(b.copy());
            } else if let Some(v) = self.initial_bias_value {
                biases.push(arrayfire::constant!(v; 1, widths[index + 1] as u64));
            } else {
                biases.push(
                    self.initial_random_range
                        * arrayfire::randu::<f64>(dim4![1, widths[index + 1] as u64, 1, 1])
                        - self.initial_random_range / 2f64,
                );
            }
        }

        Arm {
            num_params: self.num_params,
            num_markers: self.num_markers,
            params: ArmParams { weights, biases },
            // TODO: impl build method for setting precisions
            hyperparams: ArmHyperparams {
                weight_precisions: vec![1.0; self.num_layers],
                bias_precisions: vec![1.0; self.num_layers - 1],
                error_precision: 1.0,
            },
            layer_widths: self.layer_widths.clone(),
            num_layers: self.num_layers,
            rng: thread_rng(),
            verbose: self.verbose,
        }
    }
}

mod tests {
    use arrayfire::{dim4, Array, Dim4};
    // use arrayfire::{af_print, randu};

    use super::{Arm, ArmBuilder, ArmParams};

    // #[test]
    // fn test_af() {
    //     let num_rows: u64 = 5;
    //     let num_cols: u64 = 3;
    //     let dims = Dim4::new(&[num_rows, num_cols, 1, 1]);
    //     let a = randu::<f64>(dims);
    //     af_print!("Create a 5-by-3 matrix of random floats on the GPU", a);
    // }

    fn to_host(a: &Array<f64>) -> Vec<f64> {
        let mut buffer = Vec::<f64>::new();
        buffer.resize(a.elements(), 0.);
        a.host(&mut buffer);
        buffer
    }

    fn test_arm() -> Arm {
        let exp_weights = [
            Array::new(&[0., 1., 2., 3., 4., 5.], dim4![3, 2, 1, 1]),
            Array::new(&[1., 2.], dim4![2, 1, 1, 1]),
            Array::new(&[2.], dim4![1, 1, 1, 1]),
        ];
        let exp_biases = [
            Array::new(&[0., 1.], dim4![1, 2, 1, 1]),
            Array::new(&[2.], dim4![1, 1, 1, 1]),
        ];

        ArmBuilder::new()
            .with_num_markers(3)
            .add_hidden_layer(2)
            .add_layer_biases(&exp_biases[0])
            .add_layer_weights(&exp_weights[0])
            .add_summary_weights(&exp_weights[1])
            .add_summary_bias(&exp_biases[1])
            .add_output_weight(&exp_weights[2])
            .build()
    }

    #[test]
    #[should_panic(expected = "bias dim 1 does not match width of last added layer")]
    fn test_build_arm_bias_dim_zero_failure() {
        let _arm = ArmBuilder::new()
            .with_num_markers(3)
            .add_hidden_layer(2)
            .add_layer_biases(&Array::new(&[0., 1., 2.], dim4![1, 3, 1, 1]))
            .add_layer_weights(&Array::new(&[0., 1., 2., 3., 4., 5.], dim4![3, 2, 1, 1]))
            .add_output_weight(&Array::new(&[1., 2.], dim4![2, 1, 1, 1]))
            .build();
    }

    #[test]
    #[should_panic(expected = "incorrect weight dims in dim 0")]
    fn test_build_arm_weight_dim_zero_failure() {
        let _arm = ArmBuilder::new()
            .with_num_markers(3)
            .add_hidden_layer(2)
            .add_layer_biases(&Array::new(&[0., 1.], dim4![1, 2, 1, 1]))
            .add_layer_weights(&Array::new(&[0., 1., 2., 3., 4., 5.], dim4![2, 3, 1, 1]))
            .add_output_weight(&Array::new(&[1., 2.], dim4![2, 1, 1, 1]))
            .build();
    }

    #[test]
    #[should_panic(expected = "incorrect weight dims in dim 1")]
    fn test_build_arm_weight_dim_one_failure() {
        let _arm = ArmBuilder::new()
            .with_num_markers(3)
            .add_hidden_layer(2)
            .add_layer_biases(&Array::new(&[0., 1.], dim4![1, 2, 1, 1]))
            .add_layer_weights(&Array::new(
                &[0., 1., 2., 3., 4., 5., 6., 7., 8.],
                dim4![3, 3, 1, 1],
            ))
            .add_output_weight(&Array::new(&[1., 2.], dim4![2, 1, 1, 1]))
            .build();
    }

    #[test]
    #[should_panic(expected = "incorrect summary weight dims in dim 0")]
    fn test_build_arm_summary_weight_dim_zero_failure() {
        let _arm = ArmBuilder::new()
            .with_num_markers(3)
            .add_hidden_layer(2)
            .add_layer_biases(&Array::new(&[0., 1.], dim4![1, 2, 1, 1]))
            .add_layer_weights(&Array::new(&[0., 1., 2., 3., 4., 5.], dim4![3, 2, 1, 1]))
            .add_summary_weights(&Array::new(&[1., 2.], dim4![1, 2, 1, 1]))
            .add_output_weight(&Array::new(&[1.], dim4![1, 1, 1, 1]))
            .build();
    }

    #[test]
    #[should_panic(expected = "incorrect summary weight dims in dim 1: has to be 1")]
    fn test_build_arm_summary_weight_dim_one_failure() {
        let _arm = ArmBuilder::new()
            .with_num_markers(3)
            .add_hidden_layer(2)
            .add_layer_biases(&Array::new(&[0., 1.], dim4![1, 2, 1, 1]))
            .add_layer_weights(&Array::new(&[0., 1., 2., 3., 4., 5.], dim4![3, 2, 1, 1]))
            .add_summary_weights(&Array::new(&[1., 2., 1., 2.], dim4![2, 2, 1, 1]))
            .add_output_weight(&Array::new(&[1.], dim4![1, 1, 1, 1]))
            .build();
    }

    #[test]
    #[should_panic(expected = "incorrect output weight dims in dim 0: has to be 1")]
    fn test_build_arm_output_weight_dim_zero_failure() {
        let _arm = ArmBuilder::new()
            .with_num_markers(3)
            .add_hidden_layer(2)
            .add_layer_biases(&Array::new(&[0., 1.], dim4![1, 2, 1, 1]))
            .add_layer_weights(&Array::new(&[0., 1., 2., 3., 4., 5.], dim4![3, 2, 1, 1]))
            .add_summary_weights(&Array::new(&[1., 2.], dim4![2, 1, 1, 1]))
            .add_output_weight(&Array::new(&[1., 2.], dim4![2, 1, 1, 1]))
            .build();
    }

    #[test]
    #[should_panic(expected = "incorrect output weight dims in dim 1: has to be 1")]
    fn test_build_arm_output_weight_dim_one_failure() {
        let _arm = ArmBuilder::new()
            .with_num_markers(3)
            .add_hidden_layer(2)
            .add_layer_biases(&Array::new(&[0., 1.], dim4![1, 2, 1, 1]))
            .add_layer_weights(&Array::new(&[0., 1., 2., 3., 4., 5.], dim4![3, 2, 1, 1]))
            .add_summary_weights(&Array::new(&[1., 2.], dim4![2, 1, 1, 1]))
            .add_output_weight(&Array::new(&[1., 2.], dim4![1, 2, 1, 1]))
            .build();
    }

    #[test]
    fn test_build_arm_success() {
        let exp_weights = [
            Array::new(&[0., 1., 2., 3., 4., 5.], dim4![3, 2, 1, 1]),
            Array::new(&[1., 2.], dim4![2, 1, 1, 1]),
            Array::new(&[2.], dim4![1, 1, 1, 1]),
        ];
        let exp_biases = [
            Array::new(&[0., 1.], dim4![1, 2, 1, 1]),
            Array::new(&[2.], dim4![1, 1, 1, 1]),
        ];
        let exp_layer_widths = [2, 1, 1];
        let exp_weight_dims: [Dim4; 3] = [dim4![3, 2, 1, 1], dim4![2, 1, 1, 1], dim4![1, 1, 1, 1]];
        let exp_bias_dims: [Dim4; 2] = [dim4![1, 2, 1, 1], dim4![1, 1, 1, 1]];
        let arm = ArmBuilder::new()
            .with_num_markers(3)
            .add_hidden_layer(2)
            .add_layer_biases(&exp_biases[0])
            .add_layer_weights(&exp_weights[0])
            .add_summary_weights(&exp_weights[1])
            .add_summary_bias(&exp_biases[1])
            .add_output_weight(&exp_weights[2])
            .build();

        // network size
        assert_eq!(arm.num_params, 6 + 2 + 2 + 1 + 1);
        assert_eq!(arm.num_layers, 3);
        assert_eq!(arm.num_markers, 3);
        for i in 0..arm.num_layers {
            println!("{:?}", i);
            assert_eq!(arm.layer_widths[i], exp_layer_widths[i]);
            assert_eq!(arm.weights(i).dims(), exp_weight_dims[i]);
            if i < arm.num_layers - 1 {
                assert_eq!(arm.biases(i).dims(), exp_bias_dims[i]);
            }
        }

        // param values
        // weights
        for i in 0..arm.num_layers {
            assert_eq!(to_host(&arm.weights(i)), to_host(&exp_weights[i]));
        }
        // biases
        for i in 0..arm.num_layers - 1 {
            assert_eq!(to_host(&arm.biases(i)), to_host(&exp_biases[i]));
        }
    }

    #[test]
    fn test_vec_of_arrays_deepcopy() {
        let dims = Dim4::new(&[2, 1, 1, 1]);
        let v1 = vec![arrayfire::randu::<f64>(dims), arrayfire::randu::<f64>(dims)];
        let mut v2 = v1.clone();
        let a3 = arrayfire::randu::<f64>(dims);
        let v1_1_host = to_host(&v1[0]);
        let mut v2_1_host = to_host(&v2[0]);
        assert_eq!(v1_1_host, v2_1_host);
        v2[0] += a3;
        v2_1_host = to_host(&v2[0]);
        assert_ne!(v1_1_host, v2_1_host);
    }

    #[test]
    fn test_forward_feed() {
        let num_individuals = 4;
        let num_markers = 3;
        let arm = test_arm();
        let x_train: Array<f64> = Array::new(
            &[1., 0., 0., 2., 1., 1., 2., 0., 0., 2., 0., 1.],
            dim4![num_individuals, num_markers, 1, 1],
        );
        let activations = arm.forward_feed(&x_train);

        // correct number of activations
        assert_eq!(activations.len(), arm.num_layers);

        // correct dimensions of activations
        for i in 0..(arm.num_layers) {
            println!("{:?}", i);
            assert_eq!(
                activations[i].dims(),
                dim4![num_individuals, arm.layer_widths[i] as u64, 1, 1]
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
        for i in 0..(arm.num_layers) {
            println!("{:?}", i);
            assert_eq!(to_host(&activations[i]), to_host(&exp_activations[i]));
        }
    }

    #[test]
    fn test_backpropagation() {
        let num_individuals = 4;
        let num_markers = 3;
        let arm = test_arm();
        let x_train: Array<f64> = Array::new(
            &[1., 0., 0., 2., 1., 1., 2., 0., 0., 2., 0., 1.],
            dim4![num_individuals, num_markers, 1, 1],
        );
        let y_train: Array<f64> = Array::new(&[0.0, 2.0, 1.0, 1.5], dim4![4, 1, 1, 1]);
        let (weights_gradient, bias_gradient) = arm.backpropagate(&x_train, &y_train);

        // correct number of gradients
        assert_eq!(weights_gradient.len(), arm.num_layers);
        assert_eq!(bias_gradient.len(), arm.num_layers - 1);

        // correct dimensions of gradients
        for i in 0..(arm.num_layers) {
            println!("{:?}", i);
            assert_eq!(weights_gradient[i].dims(), arm.weights(i).dims());
        }
        for i in 0..(arm.num_layers - 1) {
            assert_eq!(bias_gradient[i].dims(), arm.biases(i).dims());
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
        for i in 0..(arm.num_layers) {
            assert_eq!(to_host(&weights_gradient[i]), to_host(&exp_weight_grad[i]));
        }
        for i in 0..(arm.num_layers - 1) {
            println!("{:?}", i);
            assert_eq!(to_host(&bias_gradient[i]), to_host(&exp_bias_grad[i]));
        }
    }

    #[test]
    fn test_log_density_gradient() {
        let num_individuals = 4;
        let num_markers = 3;
        let arm = test_arm();
        let x_train: Array<f64> = Array::new(
            &[1., 0., 0., 2., 1., 1., 2., 0., 0., 2., 0., 1.],
            dim4![num_individuals, num_markers, 1, 1],
        );
        let y_train: Array<f64> = Array::new(&[0.0, 2.0, 1.0, 1.5], dim4![4, 1, 1, 1]);
        let ldg = arm.log_density_gradient(&x_train, &y_train);

        // correct output length
        assert_eq!(ldg.wrt_weights.len(), arm.num_layers);
        assert_eq!(ldg.wrt_biases.len(), arm.num_layers - 1);

        // correct dimensions
        for i in 0..(arm.num_layers) {
            println!("{:?}", i);
            assert_eq!(ldg.wrt_weights[i].dims(), arm.weights(i).dims());
        }
        for i in 0..(arm.num_layers - 1) {
            assert_eq!(ldg.wrt_biases[i].dims(), arm.biases(i).dims());
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
        for i in 0..(arm.num_layers) {
            println!("{:?}", i);
            assert_eq!(to_host(&ldg.wrt_weights[i]), to_host(&exp_ldg_wrt_w[i]));
        }
        for i in 0..(arm.num_layers - 1) {
            assert_eq!(to_host(&ldg.wrt_biases[i]), to_host(&exp_ldg_wrt_b[i]));
        }
    }

    #[test]
    fn test_param_vec() {
        let params = ArmParams {
            weights: vec![
                Array::new(&[0.1, 0.2], dim4![2, 1, 1, 1]),
                Array::new(&[0.3], dim4![1, 1, 1, 1]),
            ],
            biases: vec![Array::new(&[0.4], dim4![1, 1, 1, 1])],
        };
        let exp = vec![0.1, 0.2, 0.3, 0.4];
        assert_eq!(params.param_vec(), exp);
    }

    #[test]
    fn test_uniform_step_sizes() {
        let arm = test_arm();
        let val = 1.0;
        let step_sizes = arm.uniform_step_sizes(val);
        for i in 0..(arm.num_layers - 1) {
            let mut obs = to_host(&step_sizes.wrt_weights[i]);
            assert_eq!(obs, vec![val; obs.len()]);
            obs = to_host(&step_sizes.wrt_biases[i]);
            assert_eq!(obs, vec![val; obs.len()]);
        }
        let obs = to_host(&step_sizes.wrt_weights[arm.num_layers - 1]);
        assert_eq!(obs, vec![val; obs.len()]);
    }
}
