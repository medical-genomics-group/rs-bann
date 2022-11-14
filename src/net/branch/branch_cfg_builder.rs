use super::branch::BranchCfg;
use super::params::BranchHyperparams;
use arrayfire::{constant, dim4, Array};
use rand::distributions::Distribution;
use rand::thread_rng;
use rand_distr::{Gamma, Normal};

struct GammaParams {
    shape: f32,
    scale: f32,
}

pub struct BranchCfgBuilder {
    num_params: usize,
    num_markers: usize,
    layer_widths: Vec<usize>,
    num_layers: usize,
    initial_weight_value: Option<f32>,
    initial_bias_value: Option<f32>,
    init_param_variance: Option<f32>,
    init_gamma_params: Option<GammaParams>,
}

impl BranchCfgBuilder {
    pub fn new() -> Self {
        Self {
            num_params: 0,
            num_markers: 0,
            // this does not contain the input layer width
            layer_widths: vec![],
            // we always have a summary and an output node, so at least 2 layers.
            num_layers: 2,
            initial_weight_value: None,
            initial_bias_value: None,
            init_param_variance: None,
            init_gamma_params: None,
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

    pub fn with_init_param_variance(mut self, range: f32) -> Self {
        self.init_param_variance = Some(range);
        self
    }

    pub fn with_initial_weights_value(mut self, value: f32) -> Self {
        self.initial_weight_value = Some(value);
        self
    }

    pub fn with_initial_bias_value(mut self, value: f32) -> Self {
        self.initial_bias_value = Some(value);
        self
    }

    pub fn with_init_gamma_params(mut self, shape: f32, scale: f32) -> Self {
        self.init_gamma_params = Some(GammaParams { shape, scale });
        self
    }

    pub fn build_base(mut self) -> BranchCfg {
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

        let mut params: Vec<f32> = vec![0.0; self.num_params];
        let mut weight_precisions = vec![Array::new(&[1.0], dim4!(1, 1, 1, 1)); self.num_layers];

        // initialize weights and biases
        if let Some(v) = self.initial_weight_value {
            params[0..num_weights].iter_mut().for_each(|x| *x = v);
        } else if let Some(v) = self.init_param_variance {
            let d = Normal::new(0.0, v.sqrt()).unwrap();
            params[0..num_weights]
                .iter_mut()
                .for_each(|x| *x = d.sample(&mut rng));
            weight_precisions = vec![Array::new(&[1.0 / v], dim4!(1, 1, 1, 1)); self.num_layers];
        } else if let Some(gamma_params) = &self.init_gamma_params {
            // iterate over layers
            let precision_prior = Gamma::new(gamma_params.shape, gamma_params.scale).unwrap();
            let mut prev_width = self.num_markers;
            let mut insert_ix: usize = 0;
            for (lix, width) in self.layer_widths.iter().enumerate() {
                let layer_precision = precision_prior.sample(&mut rng);
                weight_precisions[lix] = Array::new(&[layer_precision], dim4!(1, 1, 1, 1));
                let layer_std = (1.0 / layer_precision).sqrt();
                let layer_weight_prior = Normal::new(0.0, layer_std).unwrap();
                let num_weights = prev_width * width;
                params[insert_ix..insert_ix + num_weights]
                    .iter_mut()
                    .for_each(|x| *x = layer_weight_prior.sample(&mut rng));
                insert_ix += num_weights;
                prev_width = *width;
            }
        }

        let mut bias_precisions = vec![1.0; self.num_layers - 1];

        // initialize biases
        if let Some(v) = self.initial_bias_value {
            params[num_weights..].iter_mut().for_each(|x| *x = v);
        } else if let Some(v) = self.init_param_variance {
            let d = Normal::new(0.0, v.sqrt()).unwrap();
            params[num_weights..]
                .iter_mut()
                .for_each(|x| *x = d.sample(&mut rng));
            bias_precisions = vec![1.0 / v; self.num_layers - 1];
        } else if let Some(gamma_params) = &self.init_gamma_params {
            // iterate over layers
            let precision_prior = Gamma::new(gamma_params.shape, gamma_params.scale).unwrap();
            let mut insert_ix = num_weights;
            for (lix, width) in self.layer_widths[..self.num_layers - 1].iter().enumerate() {
                let num_biases = width;
                let layer_bias_precision = precision_prior.sample(&mut rng);
                let layer_bias_std = (1.0 / layer_bias_precision).sqrt();
                bias_precisions[lix] = layer_bias_precision;
                let layer_bias_prior = Normal::new(0.0, layer_bias_std).unwrap();
                params[insert_ix..insert_ix + num_biases]
                    .iter_mut()
                    .for_each(|x| *x = layer_bias_prior.sample(&mut rng));
                insert_ix += num_biases;
            }
        }

        BranchCfg {
            num_params: self.num_params,
            num_markers: self.num_markers,
            layer_widths: self.layer_widths.clone(),
            params,
            // TODO: impl build method for setting precisions
            hyperparams: BranchHyperparams {
                weight_precisions,
                bias_precisions,
                error_precision: 1.0,
            },
        }
    }

    pub fn build_ard(mut self) -> BranchCfg {
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

        let mut params: Vec<f32> = vec![0.0; self.num_params];
        let mut weight_precisions: Vec<Array<f32>> = widths
            .iter()
            .take(widths.len() - 1)
            .map(|w| constant!(1.0; *w as u64))
            .collect();

        if let Some(v) = self.initial_weight_value {
            params[0..num_weights].iter_mut().for_each(|x| *x = v);
        } else if let Some(v) = self.init_param_variance {
            let d = Normal::new(0.0, v.sqrt()).unwrap();
            params[0..num_weights]
                .iter_mut()
                .for_each(|x| *x = d.sample(&mut rng));
            weight_precisions = widths
                .iter()
                .take(widths.len() - 1)
                .map(|w| constant!(1.0 / v; *w as u64))
                .collect();
        } else if let Some(gamma_params) = &self.init_gamma_params {
            let precision_prior = Gamma::new(gamma_params.shape, gamma_params.scale).unwrap();
            let mut prev_width = self.num_markers;
            let mut insert_ix: usize = 0;
            for (lix, width) in self.layer_widths.iter().enumerate() {
                let num_weights = prev_width * width;
                let mut layer_precisions = vec![0.0; prev_width];
                for ard_group_ix in 0..prev_width {
                    let ard_group_precision = precision_prior.sample(&mut rng);
                    layer_precisions[ard_group_ix] = ard_group_precision;
                    let ard_group_std = (1.0 / ard_group_precision).sqrt();
                    let ard_group_prior = Normal::new(0.0, ard_group_std).unwrap();
                    (ard_group_ix..num_weights)
                        .step_by(prev_width)
                        .for_each(|ix| params[insert_ix + ix] = ard_group_prior.sample(&mut rng));
                }
                weight_precisions[lix] =
                    Array::new(&layer_precisions, dim4!(prev_width as u64, 1, 1, 1));
                insert_ix += num_weights;
                prev_width = *width;
            }
        }

        let mut bias_precisions = vec![1.0; self.num_layers - 1];

        if let Some(v) = self.initial_bias_value {
            params[num_weights..].iter_mut().for_each(|x| *x = v);
        } else if let Some(v) = self.init_param_variance {
            let d = Normal::new(0.0, v.sqrt()).unwrap();
            params[num_weights..]
                .iter_mut()
                .for_each(|x| *x = d.sample(&mut rng));
            bias_precisions = vec![1.0 / v; self.num_layers - 1];
        } else if let Some(gamma_params) = &self.init_gamma_params {
            // here we still have one precision per layer
            let precision_prior = Gamma::new(gamma_params.shape, gamma_params.scale).unwrap();
            let mut insert_ix: usize = num_weights;
            for (lix, width) in self.layer_widths[..self.num_layers - 1].iter().enumerate() {
                let num_biases = width;
                let layer_bias_precision = precision_prior.sample(&mut rng);
                let layer_bias_std = (1.0 / layer_bias_precision).sqrt();
                bias_precisions[lix] = layer_bias_precision;
                let layer_bias_prior = Normal::new(0.0, layer_bias_std).unwrap();
                params[insert_ix..insert_ix + num_biases]
                    .iter_mut()
                    .for_each(|x| *x = layer_bias_prior.sample(&mut rng));
                insert_ix += num_biases;
            }
        }

        BranchCfg {
            num_params: self.num_params,
            num_markers: self.num_markers,
            layer_widths: self.layer_widths.clone(),
            params,
            // TODO: impl build method for setting precisions
            hyperparams: BranchHyperparams {
                weight_precisions,
                bias_precisions,
                error_precision: 1.0,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    // use arrayfire::constant;

    use super::BranchCfgBuilder;

    #[test]
    fn test_build_branch_cfg() {
        let mut bld = BranchCfgBuilder::new().with_num_markers(3);
        bld.add_hidden_layer(3);
        let cfg = bld.build_base();
        assert_eq!(cfg.num_markers, 3);
        assert_eq!(cfg.num_params, 17);
    }

    // fn test_af_constant_dims() {
    //     assert_eq!(constant!(1.0; 5).dims()[0], 5);
    // }
}
