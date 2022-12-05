use super::super::params::BranchPrecisions;
use super::branch::BranchCfg;
use arrayfire::{constant, dim4, Array};
use rand::distributions::Distribution;
use rand::{rngs::ThreadRng, thread_rng};
use rand_distr::{Gamma, Normal};

// we always have a summary and an output node, so at least 2 layers.
const MIN_NUM_LAYERS: usize = 2;

struct GammaParams {
    shape: f32,
    scale: f32,
}

pub struct BranchCfgBuilder {
    num_params: usize,
    num_weights: usize,
    num_markers: usize,
    layer_widths: Vec<usize>,
    num_layers: usize,
    output_param_variance: f32,
    initial_weight_value: Option<f32>,
    initial_bias_value: Option<f32>,
    init_param_variance: Option<f32>,
    init_gamma_params: Option<GammaParams>,
    sample_precisions: bool,
}

impl Default for BranchCfgBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BranchCfgBuilder {
    pub fn new() -> Self {
        Self {
            num_params: 0,
            num_weights: 0,
            num_markers: 0,
            // this does not contain the input layer width
            layer_widths: vec![],
            num_layers: MIN_NUM_LAYERS,
            output_param_variance: 1.0,
            initial_weight_value: None,
            initial_bias_value: None,
            init_param_variance: None,
            init_gamma_params: None,
            sample_precisions: false,
        }
    }

    // TODO: allow this through cli
    /// Sample precisions from prior defined by hyperparams instead
    /// of setting them to the prior means.
    pub fn with_sampled_precisions(mut self) -> Self {
        self.sample_precisions = true;
        self
    }

    pub fn with_output_param_variance(mut self, variance: f32) -> Self {
        self.output_param_variance = variance;
        self
    }

    pub fn with_num_markers(mut self, num_markers: usize) -> Self {
        self.num_markers = num_markers;
        self
    }

    fn num_added_layers(&self) -> usize {
        self.num_layers - MIN_NUM_LAYERS
    }

    fn last_layer_before_summary_width(&self) -> usize {
        if self.num_added_layers() == 0 {
            self.num_markers
        } else {
            self.layer_widths[self.num_added_layers() - 1]
        }
    }

    pub fn add_hidden_layer(&mut self, layer_width: usize) {
        self.num_weights += self.last_layer_before_summary_width() * layer_width;
        self.num_params += (self.last_layer_before_summary_width() + 1) * layer_width;
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

    fn init_weights_with_initial_weight_value(&self, params: &mut [f32], num_weights: usize) {
        let v = self.initial_weight_value.unwrap();
        params[0..num_weights].iter_mut().for_each(|x| *x = v);
    }

    fn init_biases_with_initial_weight_value(&self, params: &mut [f32], num_weights: usize) {
        let v = self.initial_weight_value.unwrap();
        params[num_weights..].iter_mut().for_each(|x| *x = v);
    }

    fn init_weights_with_initial_param_variance(
        &self,
        params: &mut [f32],
        num_weights: usize,
        rng: &mut ThreadRng,
    ) {
        let v = self.init_param_variance.unwrap();
        let d = Normal::new(0.0, v.sqrt()).unwrap();
        params[0..num_weights]
            .iter_mut()
            .for_each(|x| *x = d.sample(rng));
    }

    fn init_biases_with_initial_param_variance(
        &self,
        params: &mut [f32],
        num_weights: usize,
        rng: &mut ThreadRng,
    ) {
        let v = self.init_param_variance.unwrap();
        let d = Normal::new(0.0, v.sqrt()).unwrap();
        params[num_weights..]
            .iter_mut()
            .for_each(|x| *x = d.sample(rng));
    }

    fn init_weights_with_init_gamma(&self, params: &mut [f32], rng: &mut ThreadRng) {
        // iterate over layers
        let gamma_params = self.init_gamma_params.as_ref().unwrap();
        let precision_prior = Gamma::new(gamma_params.shape, gamma_params.scale).unwrap();
        let mut prev_width = self.num_markers;
        let mut insert_ix: usize = 0;
        for (_lix, width) in self.layer_widths[..self.num_layers].iter().enumerate() {
            let layer_precision = if self.sample_precisions {
                precision_prior.sample(rng)
            } else {
                // set to mean of gamma
                gamma_params.shape * gamma_params.scale
            };
            let layer_std = (1.0 / layer_precision).sqrt();
            let layer_weight_prior = Normal::new(0.0, layer_std).unwrap();
            let num_weights = prev_width * width;
            params[insert_ix..insert_ix + num_weights]
                .iter_mut()
                .for_each(|x| *x = layer_weight_prior.sample(rng));
            insert_ix += num_weights;
            prev_width = *width;
        }
    }

    fn init_biases_with_init_gamma(
        &self,
        params: &mut [f32],
        num_weights: usize,
        rng: &mut ThreadRng,
    ) {
        // iterate over layers
        let gamma_params = self.init_gamma_params.as_ref().unwrap();
        let precision_prior = Gamma::new(gamma_params.shape, gamma_params.scale).unwrap();
        let mut insert_ix = num_weights;
        for (_lix, width) in self.layer_widths[..self.num_layers - 1].iter().enumerate() {
            let num_biases = width;
            let layer_bias_precision = if self.sample_precisions {
                precision_prior.sample(rng)
            } else {
                gamma_params.shape * gamma_params.scale
            };
            let layer_bias_std = (1.0 / layer_bias_precision).sqrt();
            let layer_bias_prior = Normal::new(0.0, layer_bias_std).unwrap();
            params[insert_ix..insert_ix + num_biases]
                .iter_mut()
                .for_each(|x| *x = layer_bias_prior.sample(rng));
            insert_ix += num_biases;
        }
    }

    // iterate over weight groups and compute the precision that maximizes the likelihood
    // of the sample (i.e. the prior density)
    fn base_weight_precisions(&self, params: &[f32]) -> Vec<Array<f32>> {
        let mut weight_precisions = vec![Array::new(&[1.0], dim4!(1, 1, 1, 1)); self.num_layers];

        let mut prev_width = self.num_markers;
        let mut insert_ix: usize = 0;
        for (lix, width) in self.layer_widths[..self.num_layers].iter().enumerate() {
            let num_weights = prev_width * width;
            // this is the precision at the extreme point of the prior as a function of the precision
            let layer_precision: f32 = 2.0f32
                / params[insert_ix..insert_ix + num_weights]
                    .iter()
                    .map(|e| e * e)
                    .sum::<f32>();
            weight_precisions[lix] = Array::new(&[layer_precision], dim4!(1, 1, 1, 1));
            insert_ix += num_weights;
            prev_width = *width;
        }

        weight_precisions
    }

    // iterate over bias groups and compute the precision that maximizes the likelihood
    // of the sample (i.e. the prior density)
    fn bias_precisions(&self, num_weights: usize, params: &[f32]) -> Vec<f32> {
        let mut bias_precisions = vec![0.0; self.num_layers - 1];

        let mut insert_ix = num_weights;
        for (lix, width) in self.layer_widths[..self.num_layers - 1].iter().enumerate() {
            let num_biases = width;
            let layer_precision: f32 = 2.0f32
                / params[insert_ix..insert_ix + num_biases]
                    .iter()
                    .map(|e| e * e)
                    .sum::<f32>();
            bias_precisions[lix] = layer_precision;
            insert_ix += num_biases;
        }

        bias_precisions
    }

    fn finalize_num_params(&mut self) {
        // summary layer weights + bias
        self.num_params += self.last_layer_before_summary_width() + 1;
        self.num_weights += self.last_layer_before_summary_width();
        // output layer weight
        self.num_weights += 1;
        self.num_params += 1;

        // summary and output node
        self.layer_widths.push(1);
        self.layer_widths.push(1);
    }

    // // including input layer
    // fn widths(&self) -> Vec<usize> {
    //     let mut widths: Vec<usize> = vec![self.num_markers];
    //     widths.append(&mut self.layer_widths.clone());
    //     widths
    // }

    // iterate over weight groups and compute the precision that maximizes the likelihood
    // of the sample (i.e. the prior density)
    fn ard_weight_precisions(&self, params: &[f32]) -> Vec<Array<f32>> {
        let mut weight_precisions = vec![Array::new(&[1.0], dim4!(1, 1, 1, 1)); self.num_layers];

        let mut prev_width = self.num_markers;
        let mut insert_ix = 0;
        for (lix, width) in self.layer_widths[..self.num_ard_layers()]
            .iter()
            .enumerate()
        {
            let num_weights = prev_width * width;
            let mut layer_precisions = vec![0.0; prev_width];
            for ard_group_ix in 0..prev_width {
                layer_precisions[ard_group_ix] = 2.0f32
                    / (ard_group_ix..num_weights)
                        .step_by(prev_width)
                        .map(|ix| params[insert_ix + ix])
                        .map(|e| e * e)
                        .sum::<f32>();
            }
            weight_precisions[lix] =
                Array::new(&layer_precisions, dim4!(prev_width as u64, 1, 1, 1));
            insert_ix += num_weights;
            prev_width = *width;
        }
        // summary layer
        let summary_layer_precision = 2.0f32
            / params[insert_ix..insert_ix + prev_width]
                .iter()
                .map(|e| e * e)
                .sum::<f32>();
        weight_precisions[self.num_ard_layers()] = constant!(summary_layer_precision; 1);

        // output layer has to be done jointly for all branches

        weight_precisions
    }

    fn num_ard_layers(&self) -> usize {
        self.num_layers - 2
    }

    pub fn build_base(mut self) -> BranchCfg {
        let mut res = self.build();
        res.precisions.weight_precisions = self.base_weight_precisions(&res.params);
        res
    }

    pub fn build_ard(mut self) -> BranchCfg {
        let mut res = self.build();
        res.precisions.weight_precisions = self.ard_weight_precisions(&res.params);
        res
    }

    fn build(&mut self) -> BranchCfg {
        self.finalize_num_params();
        let mut rng = thread_rng();

        let mut params: Vec<f32> = vec![0.0; self.num_params];

        if self.initial_weight_value.is_some() {
            self.init_weights_with_initial_weight_value(&mut params, self.num_weights);
        } else if self.init_param_variance.is_some() {
            self.init_weights_with_initial_param_variance(&mut params, self.num_weights, &mut rng);
        } else if self.init_gamma_params.is_some() {
            self.init_weights_with_init_gamma(&mut params, &mut rng);
        }

        // initialize biases
        if self.initial_bias_value.is_some() {
            self.init_biases_with_initial_weight_value(&mut params, self.num_weights);
        } else if self.init_param_variance.is_some() {
            self.init_biases_with_initial_param_variance(&mut params, self.num_weights, &mut rng);
        } else if self.init_gamma_params.is_some() {
            self.init_biases_with_init_gamma(&mut params, self.num_weights, &mut rng);
        }

        let bias_precisions = self.bias_precisions(self.num_weights, &params);

        BranchCfg {
            num_params: self.num_params,
            num_weights: self.num_weights,
            num_markers: self.num_markers,
            layer_widths: self.layer_widths.clone(),
            params,
            precisions: BranchPrecisions {
                weight_precisions: Vec::new(),
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
