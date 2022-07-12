use arrayfire::{dim4, matmul, tanh, Array, MatProp};

fn to_host(a: &Array<f64>) -> Vec<f64> {
    let mut buffer = Vec::<f64>::new();
    buffer.resize(a.elements(), 0.);
    a.host(&mut buffer);
    buffer
}

#[derive(Clone)]
struct ArmMomenta {
    weights_momenta: Vec<Array<f64>>,
    bias_momenta: Vec<Array<f64>>,
}

/// Gradients of the log density w.r.t. the network parameters.
#[derive(Clone)]
struct ArmLogDensityGradient {
    wrt_weights: Vec<Array<f64>>,
    wrt_biases: Vec<Array<f64>>,
}

pub struct Arm {
    num_markers: usize,
    weights: Vec<Array<f64>>,
    biases: Vec<Array<f64>>,
    weight_precisions: Vec<f64>,
    bias_precisions: Vec<f64>,
    error_precision: f64,
    layer_widths: Vec<usize>,
    num_layers: usize,
}

impl Arm {
    pub fn hmc_step(&mut self, x_train: &Array<f64>, y_train: &Array<f64>) {
        let init_weights = self.weights.clone();
        // TODO: add heuristic step sizes
        // TODO: add u turn diagnostic for tuning
        let init_momenta = self.sample_momenta();
        let mut momenta = init_momenta.clone();
        // initial half step
    }

    // TODO: split into bias and weights
    fn sample_momenta(&self) -> ArmMomenta {
        let mut weights_momenta = Vec::with_capacity(self.num_layers);
        let mut bias_momenta = Vec::with_capacity(self.num_layers - 1);
        for index in 0..self.num_layers - 1 {
            weights_momenta.push(arrayfire::randn::<f64>(self.weights[index].dims()));
            bias_momenta.push(arrayfire::randn::<f64>(self.biases[index].dims()));
        }
        // output layer weight momentum
        weights_momenta.push(arrayfire::randn::<f64>(
            self.weights[self.num_layers - 1].dims(),
        ));
        ArmMomenta {
            weights_momenta,
            bias_momenta,
        }
    }

    fn update_momenta(
        &self,
        momenta: &mut ArmMomenta,
        step_sizes: &Vec<f64>,
        // the fraction of a step that will be taken
        step_size_fraction: f64,
        x_train: &Array<f64>,
        y_train: &Array<f64>,
    ) {
        let ld_gradient = self.log_density_gradient(x_train, y_train);
        // update each momentum component individually
        for index in 0..self.num_layers {
            momenta.weights_momenta[index] +=
                step_sizes[index] * step_size_fraction * &ld_gradient.wrt_weights[index];
            if index == self.num_layers - 1 {
                break;
            }
            momenta.bias_momenta[index] +=
                step_sizes[index] * step_size_fraction * &ld_gradient.wrt_biases[index];
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
            &self.weights[layer_index],
            MatProp::NONE,
            MatProp::NONE,
        );
        let bias_m = &arrayfire::tile(
            &self.biases[layer_index],
            dim4!(input.dims().get()[0], 1, 1, 1),
        );
        tanh(&(xw + bias_m))
    }

    fn output_neuron_activation(&self, input: &Array<f64>) -> Array<f64> {
        matmul(
            input,
            &self.weights[self.num_layers - 1],
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
            self.weights.last().unwrap(),
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
                &self.weights[layer_index],
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
                -self.weight_precisions[layer_index] * &self.weights[layer_index]
                    - self.error_precision * &d_rss_wrt_weights[layer_index],
            );
            ldg_wrt_biases.push(
                -self.bias_precisions[layer_index] * &self.biases[layer_index]
                    - self.error_precision * &d_rss_wrt_biases[layer_index],
            );
        }
        // output layer gradient
        ldg_wrt_weights.push(
            -self.weight_precisions[self.num_layers - 1] * &self.weights[self.num_layers - 1]
                - self.error_precision * &d_rss_wrt_weights[self.num_layers - 1],
        );
        ArmLogDensityGradient {
            wrt_weights: ldg_wrt_weights,
            wrt_biases: ldg_wrt_biases,
        }
    }
}

struct ArmBuilder {
    num_markers: usize,
    layer_widths: Vec<usize>,
    num_layers: usize,
    initial_weight_value: Option<f64>,
    initial_bias_value: Option<f64>,
    initial_random_range: f64,
    biases: Vec<Option<Array<f64>>>,
    weights: Vec<Option<Array<f64>>>,
}

impl ArmBuilder {
    fn new() -> Self {
        Self {
            num_markers: 0,
            layer_widths: vec![],
            // we always have a summary and an output node, so at least 2 layers.
            num_layers: 2,
            initial_weight_value: None,
            initial_bias_value: None,
            initial_random_range: 0.05,
            biases: vec![],
            weights: vec![],
        }
    }

    fn with_num_markers(&mut self, num_markers: usize) -> &mut Self {
        self.num_markers = num_markers;
        self
    }

    fn add_hidden_layer(&mut self, layer_width: usize) -> &mut Self {
        self.layer_widths.push(layer_width);
        self.num_layers += 1;
        self.biases.push(None);
        self.weights.push(None);
        self
    }

    fn add_layer_biases(&mut self, biases: Array<f64>) -> &mut Self {
        assert!(
            biases.dims().get()[0] as usize == 1,
            "bias vector dim 0 != 1, expected row vector"
        );
        assert!(
            biases.dims().get()[1] as usize == *self.layer_widths.last().unwrap(),
            "bias dim 1 does not match width of last added layer"
        );
        *self.biases.last_mut().unwrap() = Some(biases);
        self
    }

    fn add_layer_weights(&mut self, weights: Array<f64>) -> &mut Self {
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
        *self.weights.last_mut().unwrap() = Some(weights);
        self
    }

    fn add_summary_bias(&mut self, bias: Array<f64>) -> &mut Self {
        let wdims = *bias.dims().get();
        assert!(
            wdims[0] as usize == 1,
            "incorrect summary bias dims in dim 0: has to be 1."
        );
        assert!(
            wdims[1] as usize == 1,
            "incorrect summary bias dims in dim 1: has to be 1."
        );
        self.biases.push(Some(bias));
        self
    }

    fn add_summary_weights(&mut self, weights: Array<f64>) -> &mut Self {
        let wdims = *weights.dims().get();
        assert!(
            wdims[0] as usize == self.layer_widths[self.num_layers - 3],
            "incorrect summary weight dims in dim 0"
        );
        assert!(
            wdims[1] as usize == 1,
            "incorrect summary weight dims in dim 1: has to be 1."
        );
        self.weights.push(Some(weights));
        self
    }

    fn add_output_weight(&mut self, weights: Array<f64>) -> &mut Self {
        let wdims = *weights.dims().get();
        assert!(
            wdims[0] as usize == 1,
            "incorrect output weight dims in dim 0: has to be 1."
        );
        assert!(
            wdims[1] as usize == 1,
            "incorrect output weight dims in dim 1: has to be 1."
        );
        self.weights.push(Some(weights));
        self
    }

    fn with_initial_random_range(&mut self, range: f64) -> &mut Self {
        self.initial_random_range = range;
        self
    }

    fn with_initial_weights_value(&mut self, value: f64) -> &mut Self {
        self.initial_weight_value = Some(value);
        self
    }

    fn with_initial_bias_value(&mut self, value: f64) -> &mut Self {
        self.initial_bias_value = Some(value);
        self
    }

    fn build(&mut self) -> Arm {
        let mut widths: Vec<usize> = vec![self.num_markers];
        // summary and output node
        self.layer_widths.push(1);
        self.layer_widths.push(1);
        widths.append(&mut self.layer_widths.clone());

        // add None if weights or biases not added for last layers
        for _ in 0..self.num_layers - self.weights.len() {
            self.weights.push(None);
        }

        for _ in 0..self.num_layers - 1 - self.biases.len() {
            self.biases.push(None);
        }

        let mut weights = vec![];
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
            num_markers: self.num_markers,
            weights,
            biases,
            // TODO: impl build method for setting precisions
            weight_precisions: vec![1.0; self.num_layers],
            bias_precisions: vec![1.0; self.num_layers - 1],
            error_precision: 1.0,
            layer_widths: self.layer_widths.clone(),
            num_layers: self.num_layers,
        }
    }
}

mod tests {
    use arrayfire::{af_print, dim4, randu, Array, Dim4};

    use super::{Arm, ArmBuilder};

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
            .add_layer_biases(exp_biases[0].copy())
            .add_layer_weights(exp_weights[0].copy())
            .add_summary_weights(exp_weights[1].copy())
            .add_summary_bias(exp_biases[1].copy())
            .add_output_weight(exp_weights[2].copy())
            .build()
    }

    #[test]
    #[should_panic(expected = "bias dim 1 does not match width of last added layer")]
    fn test_build_arm_bias_dim_zero_failure() {
        let _arm = ArmBuilder::new()
            .with_num_markers(3)
            .add_hidden_layer(2)
            .add_layer_biases(Array::new(&[0., 1., 2.], dim4![1, 3, 1, 1]))
            .add_layer_weights(Array::new(&[0., 1., 2., 3., 4., 5.], dim4![3, 2, 1, 1]))
            .add_output_weight(Array::new(&[1., 2.], dim4![2, 1, 1, 1]))
            .build();
    }

    #[test]
    #[should_panic(expected = "incorrect weight dims in dim 0")]
    fn test_build_arm_weight_dim_zero_failure() {
        let _arm = ArmBuilder::new()
            .with_num_markers(3)
            .add_hidden_layer(2)
            .add_layer_biases(Array::new(&[0., 1.], dim4![1, 2, 1, 1]))
            .add_layer_weights(Array::new(&[0., 1., 2., 3., 4., 5.], dim4![2, 3, 1, 1]))
            .add_output_weight(Array::new(&[1., 2.], dim4![2, 1, 1, 1]))
            .build();
    }

    #[test]
    #[should_panic(expected = "incorrect weight dims in dim 1")]
    fn test_build_arm_weight_dim_one_failure() {
        let _arm = ArmBuilder::new()
            .with_num_markers(3)
            .add_hidden_layer(2)
            .add_layer_biases(Array::new(&[0., 1.], dim4![1, 2, 1, 1]))
            .add_layer_weights(Array::new(
                &[0., 1., 2., 3., 4., 5., 6., 7., 8.],
                dim4![3, 3, 1, 1],
            ))
            .add_output_weight(Array::new(&[1., 2.], dim4![2, 1, 1, 1]))
            .build();
    }

    #[test]
    #[should_panic(expected = "incorrect summary weight dims in dim 0")]
    fn test_build_arm_summary_weight_dim_zero_failure() {
        let _arm = ArmBuilder::new()
            .with_num_markers(3)
            .add_hidden_layer(2)
            .add_layer_biases(Array::new(&[0., 1.], dim4![1, 2, 1, 1]))
            .add_layer_weights(Array::new(&[0., 1., 2., 3., 4., 5.], dim4![3, 2, 1, 1]))
            .add_summary_weights(Array::new(&[1., 2.], dim4![1, 2, 1, 1]))
            .add_output_weight(Array::new(&[1.], dim4![1, 1, 1, 1]))
            .build();
    }

    #[test]
    #[should_panic(expected = "incorrect summary weight dims in dim 1: has to be 1")]
    fn test_build_arm_summary_weight_dim_one_failure() {
        let _arm = ArmBuilder::new()
            .with_num_markers(3)
            .add_hidden_layer(2)
            .add_layer_biases(Array::new(&[0., 1.], dim4![1, 2, 1, 1]))
            .add_layer_weights(Array::new(&[0., 1., 2., 3., 4., 5.], dim4![3, 2, 1, 1]))
            .add_summary_weights(Array::new(&[1., 2., 1., 2.], dim4![2, 2, 1, 1]))
            .add_output_weight(Array::new(&[1.], dim4![1, 1, 1, 1]))
            .build();
    }

    #[test]
    #[should_panic(expected = "incorrect output weight dims in dim 0: has to be 1")]
    fn test_build_arm_output_weight_dim_zero_failure() {
        let _arm = ArmBuilder::new()
            .with_num_markers(3)
            .add_hidden_layer(2)
            .add_layer_biases(Array::new(&[0., 1.], dim4![1, 2, 1, 1]))
            .add_layer_weights(Array::new(&[0., 1., 2., 3., 4., 5.], dim4![3, 2, 1, 1]))
            .add_summary_weights(Array::new(&[1., 2.], dim4![2, 1, 1, 1]))
            .add_output_weight(Array::new(&[1., 2.], dim4![2, 1, 1, 1]))
            .build();
    }

    #[test]
    #[should_panic(expected = "incorrect output weight dims in dim 1: has to be 1")]
    fn test_build_arm_output_weight_dim_one_failure() {
        let _arm = ArmBuilder::new()
            .with_num_markers(3)
            .add_hidden_layer(2)
            .add_layer_biases(Array::new(&[0., 1.], dim4![1, 2, 1, 1]))
            .add_layer_weights(Array::new(&[0., 1., 2., 3., 4., 5.], dim4![3, 2, 1, 1]))
            .add_summary_weights(Array::new(&[1., 2.], dim4![2, 1, 1, 1]))
            .add_output_weight(Array::new(&[1., 2.], dim4![1, 2, 1, 1]))
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
            .add_layer_biases(exp_biases[0].copy())
            .add_layer_weights(exp_weights[0].copy())
            .add_summary_weights(exp_weights[1].copy())
            .add_summary_bias(exp_biases[1].copy())
            .add_output_weight(exp_weights[2].copy())
            .build();

        // network size
        assert_eq!(arm.num_layers, 3);
        assert_eq!(arm.num_markers, 3);
        for i in 0..arm.num_layers {
            println!("{:?}", i);
            assert_eq!(arm.layer_widths[i], exp_layer_widths[i]);
            assert_eq!(arm.weights[i].dims(), exp_weight_dims[i]);
            if i < arm.num_layers - 1 {
                assert_eq!(arm.biases[i].dims(), exp_bias_dims[i]);
            }
        }

        // param values
        // weights
        for i in 0..arm.num_layers {
            assert_eq!(to_host(&arm.weights[i]), to_host(&exp_weights[i]));
        }
        // biases
        for i in 0..arm.num_layers - 1 {
            assert_eq!(to_host(&arm.biases[i]), to_host(&exp_biases[i]));
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
            assert_eq!(weights_gradient[i].dims(), arm.weights[i].dims());
        }
        for i in 0..(arm.num_layers - 1) {
            assert_eq!(bias_gradient[i].dims(), arm.biases[i].dims());
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

        // // correct values of gradient
        // assert_eq!(to_host(weights_gradient.last().unwrap()), vec![1.758_319_3]);
        // assert_eq!(to_host(&weights_gradient[0]), vec![0.010514336; 4]);
        // assert_eq!(to_host(bias_gradient.last().unwrap()), vec![0.14882116]);
        // assert_eq!(to_host(&bias_gradient[0]), vec![0.010514336; 2]);
    }

    // #[test]
    // fn test_log_density_gradient() {
    //     let arm = test_arm();
    //     let x_train: Array<f64> = arrayfire::constant!(1.0; 4, 2);
    //     let y_train: Array<f64> = Array::new(&[0.0, 0.0, 1.0, 1.0], dim4![4, 1, 1, 1]);
    //     //let ldg = arm.log_density_gradient(&x_train, &y_train);
    //     //assert_eq!(ldg.wrt_weights.len(), arm.num_layers);
    //     //assert_eq!(ldg.wrt_biases.len(), arm.num_layers - 1);
    // }
}
