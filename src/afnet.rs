use arrayfire::{dim4, matmul, tanh, Array, MatProp};

#[derive(Clone)]
struct ArmMomenta {
    weights_momenta: Vec<Array<f32>>,
    bias_momenta: Vec<Array<f32>>,
}

#[derive(Clone)]
struct ArmGradient {
    weights_gradient: Vec<Array<f32>>,
    bias_gradient: Vec<Array<f32>>,
}

pub struct Arm {
    weights: Vec<Array<f32>>,
    biases: Vec<Array<f32>>,
    precisions: Vec<f32>,
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
        let mut weights_momenta = Vec::with_capacity(self.num_layers - 1);
        let mut bias_momenta = Vec::with_capacity(self.num_layers - 1);
        for index in 0..(self.num_layers) {
            weights_momenta.push(arrayfire::randn::<f32>(self.weights[index].dims()));
            bias_momenta.push(arrayfire::randn::<f32>(self.biases[index].dims()));
        }
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
        let gradient = self.log_density_gradient(x_train, y_train);
        // update each momentum component individually
        for index in 0..(self.num_layers) {
            momenta.weights_momenta[index] +=
                step_sizes[index] * step_size_fraction * &gradient.weights_gradient[index];
            momenta.bias_momenta[index] +=
                step_sizes[index] * step_size_fraction * &gradient.bias_gradient[index];
        }
    }

    fn forward_feed(&self, x_train: &Array<f32>) -> Vec<Array<f32>> {
        let mut signals: Vec<Array<f32>> = Vec::with_capacity(self.num_layers - 1);
        signals.push(self.layer_activation(0, x_train));

        for layer_index in 1..(self.num_layers - 1) {
            signals.push(self.layer_activation(layer_index - 1, &signals[layer_index - 1]));
        }

        signals
    }

    fn log_density_gradient(&self, x_train: &Array<f32>, y_train: &Array<f32>) -> ArmGradient {
        // forward propagate to get signals
        let signals = self.forward_feed(x_train);

        let mut bias_gradient: Vec<Array<f32>> = Vec::with_capacity(self.num_layers - 1);
        let mut weights_gradient: Vec<Array<f32>> = Vec::with_capacity(self.num_layers - 1);
        // back propagate
        let mut activation = signals.last().unwrap();
        let mut error = activation - y_train;
        bias_gradient.push(Array::new(&[1.0_f32], dim4![1, 1, 1, 1]));
        // TODO: do dimensions check out here?
        weights_gradient.push(matmul(&error, &activation, MatProp::NONE, MatProp::NONE));
        activation = &signals[self.num_layers - 1];
        // TODO: do dimensions check out here?
        error = matmul(
            &error,
            &self.weights.last().unwrap(),
            MatProp::NONE,
            MatProp::NONE,
        );

        for layer_index in (0..self.num_layers - 2).rev() {
            let signal = &signals[layer_index];
            let delta = (1 - arrayfire::pow(activation, &2, false)) * error;
            // TODO: sum here along some dimension?
            bias_gradient.push(delta);
            weights_gradient.push(matmul(&delta, &signal, MatProp::NONE, MatProp::NONE));

            activation = signal;
            error = matmul(
                &delta,
                &self.weights[layer_index],
                MatProp::NONE,
                MatProp::NONE,
            );
        }

        bias_gradient.reverse();
        weights_gradient.reverse();
        ArmGradient {
            weights_gradient,
            bias_gradient,
        }
    }

    fn layer_activation(&self, layer_index: usize, input: &Array<f32>) -> Array<f32> {
        tanh(
            &(&matmul(
                input,
                &self.weights[layer_index],
                MatProp::NONE,
                MatProp::NONE,
            ) + &self.biases[layer_index]),
        )
    }
}

struct ArmBuilder {
    num_markers: usize,
    layer_widths: Vec<usize>,
    num_layers: usize,
    initial_random_range: f32,
}

impl ArmBuilder {
    fn new() -> Self {
        Self {
            num_markers: 0,
            layer_widths: vec![],
            num_layers: 2,
            initial_random_range: 0.05,
        }
    }

    fn with_num_markers(&mut self, num_markers: usize) -> &mut Self {
        self.num_markers = num_markers;
        self
    }

    fn add_hidden_layer(&mut self, layer_width: usize) -> &mut Self {
        self.layer_widths.push(layer_width);
        self.num_layers += 1;
        self
    }

    fn with_initial_random_range(&mut self, range: f32) -> &mut Self {
        self.initial_random_range = range;
        self
    }

    fn build(&self) -> Arm {
        let mut widths: Vec<usize> = vec![self.num_markers];
        widths.append(&mut self.layer_widths.clone());
        // the summary node
        widths.push(1);
        // the output node
        widths.push(1);

        let num_weights = widths.len() - 1;
        let mut weights = vec![];
        let mut biases: Vec<Array<f32>> = vec![];
        for index in 0..num_weights {
            weights.push(
                // this does not includes the bias term.
                self.initial_random_range
                    * arrayfire::randu::<f32>(dim4![
                        widths[index] as u64,
                        widths[index + 1] as u64,
                        1,
                        1
                    ])
                    - self.initial_random_range / 2f32,
            );
            biases.push(
                self.initial_random_range
                    * arrayfire::randu::<f32>(dim4![widths[index] as u64, 1, 1, 1])
                    - self.initial_random_range / 2f32,
            );
        }
        Arm {
            weights,
            biases,
            precisions: vec![1.0; num_weights * 2],
            num_layers: self.num_layers,
        }
    }
}

mod tests {
    use arrayfire::{af_print, randu, Array, Dim4};

    #[test]
    fn test_af() {
        let num_rows: u64 = 5;
        let num_cols: u64 = 3;
        let dims = Dim4::new(&[num_rows, num_cols, 1, 1]);
        let a = randu::<f32>(dims);
        af_print!("Create a 5-by-3 matrix of random floats on the GPU", a);
    }

    fn to_host(a: &Array<f32>) -> Vec<f32> {
        let mut buffer = Vec::<f32>::new();
        buffer.resize(a.elements(), 0.);
        a.host(&mut buffer);
        buffer
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
}
