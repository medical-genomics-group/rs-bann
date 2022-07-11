use arrayfire::{dim4, matmul, tanh, Array, MatProp};

#[derive(Clone)]
struct ArmMomenta {
    weights_momenta: Vec<Array<f32>>,
    bias_momenta: Vec<Array<f32>>,
}

/// Gradients of the log density w.r.t. the network parameters.
#[derive(Clone)]
struct ArmLogDensityGradient {
    wrt_weights: Vec<Array<f32>>,
    wrt_biases: Vec<Array<f32>>,
}

pub struct Arm {
    num_markers: usize,
    weights: Vec<Array<f32>>,
    biases: Vec<Array<f32>>,
    weight_precisions: Vec<f32>,
    bias_precisions: Vec<f32>,
    error_precision: f32,
    layer_widths: Vec<usize>,
    num_layers: usize,
}

impl Arm {
    pub fn hmc_step(&mut self, x_train: &Array<f32>, y_train: &Array<f32>) {
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
            weights_momenta.push(arrayfire::randn::<f32>(self.weights[index].dims()));
            bias_momenta.push(arrayfire::randn::<f32>(self.biases[index].dims()));
        }
        // output layer weight momentum
        weights_momenta.push(arrayfire::randn::<f32>(
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
        step_sizes: &Vec<f32>,
        // the fraction of a step that will be taken
        step_size_fraction: f32,
        x_train: &Array<f32>,
        y_train: &Array<f32>,
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

    fn forward_feed(&self, x_train: &Array<f32>) -> Vec<Array<f32>> {
        let mut activations: Vec<Array<f32>> = Vec::with_capacity(self.num_layers - 1);
        activations.push(self.mid_layer_activation(0, x_train));
        for layer_index in 1..self.num_layers - 1 {
            activations.push(self.mid_layer_activation(layer_index, activations.last().unwrap()));
        }
        activations.push(self.output_neuron_activation(activations.last().unwrap()));
        activations
    }

    fn mid_layer_activation(&self, layer_index: usize, input: &Array<f32>) -> Array<f32> {
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

    fn output_neuron_activation(&self, input: &Array<f32>) -> Array<f32> {
        matmul(
            input,
            &self.weights[self.num_layers - 1],
            MatProp::NONE,
            MatProp::NONE,
        )
    }

    pub fn backpropagate(
        &self,
        x_train: &Array<f32>,
        y_train: &Array<f32>,
    ) -> (Vec<Array<f32>>, Vec<Array<f32>>) {
        // forward propagate to get signals
        let activations = self.forward_feed(x_train);

        let mut bias_gradient: Vec<Array<f32>> = Vec::with_capacity(self.num_layers - 1);
        let mut weights_gradient: Vec<Array<f32>> = Vec::with_capacity(self.num_layers);
        // back propagate
        let mut activation = activations.last().unwrap();
        let mut error = activation - y_train;
        weights_gradient.push(arrayfire::dot(
            &error,
            activation,
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
            let delta: Array<f32> = (1 - arrayfire::pow(activation, &2, false)) * error;
            bias_gradient.push(arrayfire::sum(&delta, 0));
            weights_gradient.push(matmul(
                &arrayfire::transpose(&delta, false),
                input,
                MatProp::NONE,
                MatProp::NONE,
            ));
            error = matmul(
                &delta,
                &arrayfire::transpose(&self.weights[layer_index], false),
                MatProp::NONE,
                MatProp::NONE,
            );
        }

        let delta: Array<f32> = (1 - arrayfire::pow(&activations[0], &2, false)) * error;
        bias_gradient.push(arrayfire::sum(&delta, 0));
        weights_gradient.push(matmul(
            &arrayfire::transpose(&delta, false),
            x_train,
            MatProp::NONE,
            MatProp::NONE,
        ));

        bias_gradient.reverse();
        weights_gradient.reverse();

        (weights_gradient, bias_gradient)
    }

    fn log_density_gradient(
        &self,
        x_train: &Array<f32>,
        y_train: &Array<f32>,
    ) -> ArmLogDensityGradient {
        let (d_rss_wrt_weights, d_rss_wrt_biases) = self.backpropagate(x_train, y_train);
        let mut ldg_wrt_weights: Vec<Array<f32>> = Vec::with_capacity(self.num_layers);
        let mut ldg_wrt_biases: Vec<Array<f32>> = Vec::with_capacity(self.num_layers - 1);
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
    initial_weight_value: Option<f32>,
    initial_bias_value: Option<f32>,
    initial_random_range: f32,
    biases: Vec<Option<Array<f32>>>,
    weights: Vec<Option<Array<f32>>>,
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

    fn add_layer_biases(&mut self, biases: Array<f32>) -> &mut Self {
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

    fn add_layer_weights(&mut self, weights: Array<f32>) -> &mut Self {
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

    fn add_summary_bias(&mut self, bias: Array<f32>) -> &mut Self {
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

    fn add_summary_weights(&mut self, weights: Array<f32>) -> &mut Self {
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

    fn add_output_weight(&mut self, weights: Array<f32>) -> &mut Self {
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

    fn with_initial_random_range(&mut self, range: f32) -> &mut Self {
        self.initial_random_range = range;
        self
    }

    fn with_initial_weights_value(&mut self, value: f32) -> &mut Self {
        self.initial_weight_value = Some(value);
        self
    }

    fn with_initial_bias_value(&mut self, value: f32) -> &mut Self {
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
        let mut biases: Vec<Array<f32>> = vec![];

        for index in 0..self.num_layers {
            if let Some(w) = &self.weights[index] {
                println!(
                    "adding user defined weights with dims: {:?} at index: {:?}",
                    w.dims(),
                    index
                );
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
                    self.initial_random_range * arrayfire::randu::<f32>(dims)
                        - self.initial_random_range / 2f32,
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
                        * arrayfire::randu::<f32>(dim4![1, widths[index + 1] as u64, 1, 1])
                        - self.initial_random_range / 2f32,
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
    //     let a = randu::<f32>(dims);
    //     af_print!("Create a 5-by-3 matrix of random floats on the GPU", a);
    // }

    fn to_host(a: &Array<f32>) -> Vec<f32> {
        let mut buffer = Vec::<f32>::new();
        buffer.resize(a.elements(), 0.);
        a.host(&mut buffer);
        buffer
    }

    fn test_arm() -> Arm {
        ArmBuilder::new()
            .with_num_markers(3)
            .add_hidden_layer(2)
            .with_initial_bias_value(0.0)
            .with_initial_weights_value(1.0)
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
        let v1 = vec![arrayfire::randu::<f32>(dims), arrayfire::randu::<f32>(dims)];
        let mut v2 = v1.clone();
        let a3 = arrayfire::randu::<f32>(dims);
        let v1_1_host = to_host(&v1[0]);
        let mut v2_1_host = to_host(&v2[0]);
        assert_eq!(v1_1_host, v2_1_host);
        v2[0] += a3;
        v2_1_host = to_host(&v2[0]);
        assert_ne!(v1_1_host, v2_1_host);
    }

    // #[test]
    // fn test_backpropagation() {
    //     let arm = test_arm();
    //     let x_train: Array<f32> = arrayfire::constant!(1.0; 4, 2);
    //     let y_train: Array<f32> = Array::new(&[0.0, 0.0, 1.0, 1.0], dim4![4, 1, 1, 1]);
    //     let (weights_gradient, bias_gradient) = arm.backpropagate(&x_train, &y_train);

    //     // correct number of gradients
    //     assert_eq!(weights_gradient.len(), arm.num_layers);
    //     assert_eq!(bias_gradient.len(), arm.num_layers - 1);

    //     // correct dimensions of gradients
    //     for i in 0..(arm.num_layers) {
    //         println!("{:?}", i);
    //         assert_eq!(weights_gradient[i].dims(), arm.weights[i].dims());
    //     }
    //     for i in 0..(arm.num_layers - 1) {
    //         assert_eq!(bias_gradient[i].dims(), arm.biases[i].dims());
    //     }

    //     // correct values of gradients
    //     assert_eq!(to_host(weights_gradient.last().unwrap()), vec![1.758_319_3]);
    //     assert_eq!(to_host(&weights_gradient[0]), vec![0.010514336; 4]);
    //     assert_eq!(to_host(bias_gradient.last().unwrap()), vec![0.14882116]);
    //     assert_eq!(to_host(&bias_gradient[0]), vec![0.010514336; 2]);
    // }

    // #[test]
    // fn test_log_density_gradient() {
    //     let arm = test_arm();
    //     let x_train: Array<f32> = arrayfire::constant!(1.0; 4, 2);
    //     let y_train: Array<f32> = Array::new(&[0.0, 0.0, 1.0, 1.0], dim4![4, 1, 1, 1]);
    //     //let ldg = arm.log_density_gradient(&x_train, &y_train);
    //     //assert_eq!(ldg.wrt_weights.len(), arm.num_layers);
    //     //assert_eq!(ldg.wrt_biases.len(), arm.num_layers - 1);
    // }
}
