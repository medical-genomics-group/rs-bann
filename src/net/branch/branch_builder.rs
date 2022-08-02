use super::branch::{Branch, BranchCfg};
use super::params::{BranchHyperparams, BranchParams};
use arrayfire::{dim4, Array};
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;

pub struct BranchBuilder {
    num_params: usize,
    num_markers: usize,
    layer_widths: Vec<usize>,
    num_layers: usize,
    initial_weight_value: Option<f64>,
    initial_bias_value: Option<f64>,
    initial_random_range: f64,
    biases: Vec<Option<Array<f64>>>,
    weights: Vec<Option<Array<f64>>>,
}

impl BranchBuilder {
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
        }
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

    pub fn build(&mut self) -> Branch {
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

        Branch {
            num_params: self.num_params,
            num_markers: self.num_markers,
            params: BranchParams { weights, biases },
            // TODO: impl build method for setting precisions
            hyperparams: BranchHyperparams {
                weight_precisions: vec![1.0; self.num_layers],
                bias_precisions: vec![1.0; self.num_layers - 1],
                error_precision: 1.0,
            },
            layer_widths: self.layer_widths.clone(),
            num_layers: self.num_layers,
            rng: thread_rng(),
        }
    }
}

pub struct BranchCfgBuilder {
    num_params: usize,
    num_markers: usize,
    layer_widths: Vec<usize>,
    num_layers: usize,
    initial_weight_value: Option<f64>,
    initial_bias_value: Option<f64>,
    initial_random_range: f64,
}

impl BranchCfgBuilder {
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
        }
    }

    pub fn with_num_markers(mut self, num_markers: usize) -> Self {
        self.num_markers = num_markers;
        self
    }

    pub fn add_hidden_layer(&mut self, layer_width: usize) {
        self.layer_widths.push(layer_width);
        self.num_layers += 1;
    }

    pub fn with_initial_random_range(mut self, range: f64) -> Self {
        self.initial_random_range = range;
        self
    }

    pub fn with_initial_weights_value(mut self, value: f64) -> Self {
        self.initial_weight_value = Some(value);
        self
    }

    pub fn with_initial_bias_value(mut self, value: f64) -> Self {
        self.initial_bias_value = Some(value);
        self
    }

    pub fn build(mut self) -> BranchCfg {
        let mut widths: Vec<usize> = vec![self.num_markers];
        // summary and output node
        self.layer_widths.push(1);
        self.layer_widths.push(1);
        widths.append(&mut self.layer_widths.clone());
        let mut num_weights = 0;
        let mut rng = thread_rng();

        // get total number of params in network
        for i in 1..=self.num_layers {
            self.num_params += widths[i - 1] * widths[i] + widths[i];
            num_weights += widths[i - 1] * widths[i];
        }
        // remove count for output bias
        self.num_params -= 1;

        let mut params: Vec<f64> = vec![0.0; self.num_params];

        if let Some(v) = self.initial_weight_value {
            params[0..num_weights].iter_mut().for_each(|x| *x = v);
        } else {
            let d = Uniform::from(-self.initial_random_range..self.initial_random_range);
            params[0..num_weights]
                .iter_mut()
                .for_each(|x| *x = d.sample(&mut rng));
        }

        if let Some(v) = self.initial_bias_value {
            params[num_weights..].iter_mut().for_each(|x| *x = v);
        } else {
            let d = Uniform::from(-self.initial_random_range..self.initial_random_range);
            params[num_weights..]
                .iter_mut()
                .for_each(|x| *x = d.sample(&mut rng));
        }

        BranchCfg {
            num_params: self.num_params,
            num_markers: self.num_markers,
            layer_widths: self.layer_widths.clone(),
            params,
            // TODO: impl build method for setting precisions
            hyperparams: BranchHyperparams {
                weight_precisions: vec![1.0; self.num_layers],
                bias_precisions: vec![1.0; self.num_layers - 1],
                error_precision: 1.0,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use arrayfire::{dim4, Array};
    // use arrayfire::{af_print, randu};

    use super::BranchBuilder;

    use crate::to_host;

    #[test]
    #[should_panic(expected = "bias dim 1 does not match width of last added layer")]
    fn test_build_branch_bias_dim_zero_failure() {
        let _branch = BranchBuilder::new()
            .with_num_markers(3)
            .add_hidden_layer(2)
            .add_layer_biases(&Array::new(&[0., 1., 2.], dim4![1, 3, 1, 1]))
            .add_layer_weights(&Array::new(&[0., 1., 2., 3., 4., 5.], dim4![3, 2, 1, 1]))
            .add_output_weight(&Array::new(&[1., 2.], dim4![2, 1, 1, 1]))
            .build();
    }

    #[test]
    #[should_panic(expected = "incorrect weight dims in dim 0")]
    fn test_build_branch_weight_dim_zero_failure() {
        let _branch = BranchBuilder::new()
            .with_num_markers(3)
            .add_hidden_layer(2)
            .add_layer_biases(&Array::new(&[0., 1.], dim4![1, 2, 1, 1]))
            .add_layer_weights(&Array::new(&[0., 1., 2., 3., 4., 5.], dim4![2, 3, 1, 1]))
            .add_output_weight(&Array::new(&[1., 2.], dim4![2, 1, 1, 1]))
            .build();
    }

    #[test]
    #[should_panic(expected = "incorrect weight dims in dim 1")]
    fn test_build_branch_weight_dim_one_failure() {
        let _branch = BranchBuilder::new()
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
    fn test_build_branch_summary_weight_dim_zero_failure() {
        let _branch = BranchBuilder::new()
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
    fn test_build_branch_summary_weight_dim_one_failure() {
        let _branch = BranchBuilder::new()
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
    fn test_build_branch_output_weight_dim_zero_failure() {
        let _branch = BranchBuilder::new()
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
    fn test_build_branch_output_weight_dim_one_failure() {
        let _branch = BranchBuilder::new()
            .with_num_markers(3)
            .add_hidden_layer(2)
            .add_layer_biases(&Array::new(&[0., 1.], dim4![1, 2, 1, 1]))
            .add_layer_weights(&Array::new(&[0., 1., 2., 3., 4., 5.], dim4![3, 2, 1, 1]))
            .add_summary_weights(&Array::new(&[1., 2.], dim4![2, 1, 1, 1]))
            .add_output_weight(&Array::new(&[1., 2.], dim4![1, 2, 1, 1]))
            .build();
    }

    #[test]
    fn test_build_branch_success() {
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
        let exp_weight_dims = [dim4![3, 2, 1, 1], dim4![2, 1, 1, 1], dim4![1, 1, 1, 1]];
        let exp_bias_dims = [dim4![1, 2, 1, 1], dim4![1, 1, 1, 1]];
        let branch = BranchBuilder::new()
            .with_num_markers(3)
            .add_hidden_layer(2)
            .add_layer_biases(&exp_biases[0])
            .add_layer_weights(&exp_weights[0])
            .add_summary_weights(&exp_weights[1])
            .add_summary_bias(&exp_biases[1])
            .add_output_weight(&exp_weights[2])
            .build();

        // network size
        assert_eq!(branch.num_params(), 6 + 2 + 2 + 1 + 1);
        assert_eq!(branch.num_layers(), 3);
        assert_eq!(branch.num_markers(), 3);
        for i in 0..branch.num_layers() {
            println!("{:?}", i);
            assert_eq!(branch.layer_widths(i), exp_layer_widths[i]);
            assert_eq!(branch.weights(i).dims(), exp_weight_dims[i]);
            if i < branch.num_layers() - 1 {
                assert_eq!(branch.biases(i).dims(), exp_bias_dims[i]);
            }
        }

        // param values
        // weights
        for i in 0..branch.num_layers() {
            assert_eq!(to_host(&branch.weights(i)), to_host(&exp_weights[i]));
        }
        // biases
        for i in 0..branch.num_layers() - 1 {
            assert_eq!(to_host(&branch.biases(i)), to_host(&exp_biases[i]));
        }
    }
}
