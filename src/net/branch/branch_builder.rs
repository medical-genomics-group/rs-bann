use super::super::params::{BranchParams, BranchPrecisions};
use super::lasso_ard::LassoArdBranch;
use super::lasso_base::LassoBaseBranch;
use super::ridge_ard::RidgeArdBranch;
use super::ridge_base::RidgeBaseBranch;
use super::training_state::TrainingState;
use crate::af_helpers::af_scalar;
use arrayfire::{constant, dim4, Array};
use rand::thread_rng;

pub struct BranchBuilder {
    num_params: usize,
    num_markers: usize,
    layer_widths: Vec<usize>,
    num_layers: usize,
    initial_weight_value: Option<f32>,
    initial_bias_value: Option<f32>,
    initial_random_range: f32,
    biases: Vec<Option<Array<f32>>>,
    weights: Vec<Option<Array<f32>>>,
    initial_precision_value: Option<f32>,
}

impl Default for BranchBuilder {
    fn default() -> Self {
        Self::new()
    }
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
            initial_precision_value: None,
        }
    }

    fn num_weights(&self) -> usize {
        let mut res = 0;
        for e in &self.weights {
            res += e.as_ref().unwrap().elements();
        }
        res
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

    pub fn add_layer_biases(&mut self, biases: &Array<f32>) -> &mut Self {
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

    pub fn add_layer_weights(&mut self, weights: &Array<f32>) -> &mut Self {
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

    pub fn add_summary_bias(&mut self, bias: &Array<f32>) -> &mut Self {
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

    pub fn add_summary_weights(&mut self, weights: &Array<f32>) -> &mut Self {
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

    pub fn add_output_weight(&mut self, weights: &Array<f32>) -> &mut Self {
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

    pub fn with_initial_random_range(&mut self, range: f32) -> &mut Self {
        self.initial_random_range = range;
        self
    }

    pub fn with_initial_weights_value(&mut self, value: f32) -> &mut Self {
        self.initial_weight_value = Some(value);
        self
    }

    pub fn with_initial_bias_value(&mut self, value: f32) -> &mut Self {
        self.initial_bias_value = Some(value);
        self
    }

    pub fn with_initial_precision_value(&mut self, value: f32) -> &mut Self {
        self.initial_precision_value = Some(value);
        self
    }

    pub fn build_ridge_base(&mut self) -> RidgeBaseBranch {
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

        let mut weights: Vec<Array<f32>> = vec![];
        let mut biases: Vec<Array<f32>> = vec![];

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

        let prec = self.initial_precision_value.unwrap_or(1.0);

        RidgeBaseBranch {
            num_params: self.num_params,
            num_weights: self.num_weights(),
            num_markers: self.num_markers,
            params: BranchParams { weights, biases },
            // TODO: impl build method for setting precisions
            precisions: BranchPrecisions {
                weight_precisions: vec![af_scalar(prec); self.num_layers],
                bias_precisions: vec![af_scalar(prec); self.num_layers - 1],
                error_precision: af_scalar(prec),
            },
            layer_widths: self.layer_widths.clone(),
            num_layers: self.num_layers,
            rng: thread_rng(),
            training_state: TrainingState::default(),
        }
    }

    // TODO: this is not lasso yet
    pub fn build_lasso_base(&mut self) -> LassoBaseBranch {
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

        let mut weights: Vec<Array<f32>> = vec![];
        let mut biases: Vec<Array<f32>> = vec![];

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

        let prec = self.initial_precision_value.unwrap_or(1.0);

        LassoBaseBranch {
            num_params: self.num_params,
            num_weights: self.num_weights(),
            num_markers: self.num_markers,
            params: BranchParams { weights, biases },
            // TODO: impl build method for setting precisions
            precisions: BranchPrecisions {
                weight_precisions: vec![af_scalar(prec); self.num_layers],
                bias_precisions: vec![af_scalar(prec); self.num_layers - 1],
                error_precision: af_scalar(prec),
            },
            layer_widths: self.layer_widths.clone(),
            num_layers: self.num_layers,
            rng: thread_rng(),
            training_state: TrainingState::default(),
        }
    }

    // TODO: this is not lasso yet
    pub fn build_lasso_ard(&mut self) -> LassoArdBranch {
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

        let mut weights: Vec<Array<f32>> = vec![];
        let mut biases: Vec<Array<f32>> = vec![];

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

        let prec = self.initial_precision_value.unwrap_or(1.0);

        LassoArdBranch {
            num_params: self.num_params,
            num_weights: self.num_weights(),
            num_markers: self.num_markers,
            params: BranchParams { weights, biases },
            // TODO: impl build method for setting precisions
            precisions: BranchPrecisions {
                weight_precisions: widths.iter().map(|w| constant!(prec; *w as u64)).collect(),
                bias_precisions: vec![af_scalar(prec); self.num_layers - 1],
                error_precision: af_scalar(prec),
            },
            layer_widths: self.layer_widths.clone(),
            num_layers: self.num_layers,
            rng: thread_rng(),
            training_state: TrainingState::default(),
        }
    }

    pub fn build_ridge_ard(&mut self) -> RidgeArdBranch {
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

        let mut weights: Vec<Array<f32>> = vec![];
        let mut biases: Vec<Array<f32>> = vec![];

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

        let prec = self.initial_precision_value.unwrap_or(1.0);

        RidgeArdBranch {
            num_params: self.num_params,
            num_weights: self.num_weights(),
            num_markers: self.num_markers,
            params: BranchParams { weights, biases },
            // TODO: impl build method for setting precisions
            precisions: BranchPrecisions {
                weight_precisions: widths.iter().map(|w| constant!(prec; *w as u64)).collect(),
                bias_precisions: vec![af_scalar(prec); self.num_layers - 1],
                error_precision: af_scalar(prec),
            },
            layer_widths: self.layer_widths.clone(),
            num_layers: self.num_layers,
            rng: thread_rng(),
            training_state: TrainingState::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use arrayfire::{dim4, Array};
    // use arrayfire::{af_print, randu};

    use super::super::branch::Branch;
    use super::BranchBuilder;

    use crate::af_helpers::to_host;

    #[test]
    #[should_panic(expected = "bias dim 1 does not match width of last added layer")]
    fn test_build_branch_bias_dim_zero_failure() {
        let _branch = BranchBuilder::new()
            .with_num_markers(3)
            .add_hidden_layer(2)
            .add_layer_biases(&Array::new(&[0., 1., 2.], dim4![1, 3, 1, 1]))
            .add_layer_weights(&Array::new(&[0., 1., 2., 3., 4., 5.], dim4![3, 2, 1, 1]))
            .add_output_weight(&Array::new(&[1., 2.], dim4![2, 1, 1, 1]))
            .build_ridge_base();
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
            .build_ridge_base();
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
            .build_ridge_base();
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
            .build_ridge_base();
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
            .build_ridge_base();
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
            .build_ridge_base();
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
            .build_ridge_base();
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
            .build_ridge_base();

        // network size
        assert_eq!(branch.num_params(), 6 + 2 + 2 + 1 + 1);
        assert_eq!(branch.num_layers(), 3);
        assert_eq!(branch.num_markers, 3);
        for i in 0..branch.num_layers() {
            assert_eq!(branch.layer_width(i), exp_layer_widths[i]);
            assert_eq!(branch.layer_weights(i).dims(), exp_weight_dims[i]);
            if i < branch.num_layers() - 1 {
                assert_eq!(branch.layer_biases(i).dims(), exp_bias_dims[i]);
            }
        }

        // param values
        // weights
        for i in 0..branch.num_layers() {
            assert_eq!(to_host(branch.layer_weights(i)), to_host(&exp_weights[i]));
        }
        // biases
        for i in 0..branch.num_layers() - 1 {
            assert_eq!(to_host(branch.layer_biases(i)), to_host(&exp_biases[i]));
        }
    }
}
