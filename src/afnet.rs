use arrayfire::{dim4, Array};

pub struct Arm {
    weights: Vec<Array<f32>>,
    // always biases, weights alternating
    precisions: Vec<f32>,
    num_layers: usize,
}

impl Arm {
    pub fn hmc_step(&mut self, x_train: &Array<f32>, y_train: &Array<f32>) {
        let init_weights = self.weights.clone();
        // TODO: add heuristic step sizes
        // TODO: add u turn diagnostic for tuning
        let momentum = self.sample_momentum();
    }

    fn sample_momentum(&self) -> Vec<Array<f32>> {
        let mut momentum = Vec::new();
        for index in 0..(self.num_layers) {
            momentum.push(arrayfire::randn::<f32>(self.weights[index].dims()))
        }
        momentum
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

        let mut weights = vec![];
        for index in 0..(widths.len() - 1) {
            weights.push(
                // this includes the bias term.
                self.initial_random_range
                    * arrayfire::randu::<f32>(dim4![
                        widths[index] as u64 + 1,
                        widths[index + 1] as u64,
                        1,
                        1
                    ])
                    - self.initial_random_range / 2f32,
            )
        }
        Arm {
            weights,
            precisions: vec![1.0; weights.len() * 2],
            num_layers: self.num_layers,
        }
    }
}

mod tests {
    use arrayfire::{af_print, randu, Dim4};

    #[test]
    fn test_af() {
        let num_rows: u64 = 5;
        let num_cols: u64 = 3;
        let dims = Dim4::new(&[num_rows, num_cols, 1, 1]);
        let a = randu::<f32>(dims);
        af_print!("Create a 5-by-3 matrix of random floats on the GPU", a);
    }
}
