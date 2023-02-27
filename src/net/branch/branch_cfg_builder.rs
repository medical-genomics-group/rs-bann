use super::branch_cfg::BranchCfg;
use crate::net::{
    activation_functions::ActivationFunction,
    params::{BranchParamsHost, BranchPrecisionsHost},
};
use rand::{distributions::Distribution, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rand_distr::{Bernoulli, Gamma, Normal};

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
    summary_layer_width: usize,
    layer_widths: Vec<usize>,
    num_layers: usize,
    output_param_variance: f32,
    initial_weight_value: Option<f32>,
    initial_bias_value: Option<f32>,
    init_param_variance: Option<f32>,
    init_gamma_params: Option<GammaParams>,
    sample_precisions: bool,
    proportion_effective_markers: f32,
    fixed_param_precision: Option<f32>,
    rng: ChaCha20Rng,
    activation_function: ActivationFunction,
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
            summary_layer_width: 1,
            // this does not contain the input layer width
            layer_widths: vec![],
            num_layers: MIN_NUM_LAYERS,
            output_param_variance: 1.0,
            initial_weight_value: None,
            initial_bias_value: None,
            init_param_variance: None,
            init_gamma_params: None,
            sample_precisions: false,
            proportion_effective_markers: 1.0,
            fixed_param_precision: None,
            rng: ChaCha20Rng::from_entropy(),
            activation_function: ActivationFunction::Tanh,
        }
    }

    pub fn with_activation_function(mut self, af: ActivationFunction) -> Self {
        self.activation_function = af;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = ChaCha20Rng::seed_from_u64(seed);
        self
    }

    pub fn with_fixed_param_precision(mut self, precision: Option<f32>) -> Self {
        self.fixed_param_precision = precision;
        self
    }

    // TODO: allow this through cli
    /// Sample precisions from prior defined by hyperparams instead
    /// of setting them to the prior means.
    pub fn with_sampled_precisions(mut self) -> Self {
        self.sample_precisions = true;
        self
    }

    pub fn with_proportion_effective_markers(mut self, proportion: f32) -> Self {
        self.proportion_effective_markers = proportion;
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

    pub fn with_summary_layer_width(mut self, layer_width: usize) -> Self {
        self.summary_layer_width = layer_width;
        self
    }

    fn remove_markers_from_model(&mut self, params: &mut BranchParamsHost) {
        // remove markers from model
        if self.proportion_effective_markers < 1.0 {
            let inclusion_dist = Bernoulli::new(self.proportion_effective_markers as f64).unwrap();
            (0..self.num_markers)
                .filter(|_| !inclusion_dist.sample(&mut self.rng))
                .for_each(|marker_ix| params.set_marker_effects_to_zero(marker_ix));
        }
    }

    fn init_weights_with_initial_weight_value(&mut self, params: &mut BranchParamsHost) {
        params.set_all_weights_to_constant(self.initial_weight_value.unwrap());
        self.remove_markers_from_model(params);
    }

    fn init_biases_with_initial_weight_value(&self, params: &mut BranchParamsHost) {
        params.set_all_biases_to_constant(self.initial_weight_value.unwrap());
    }

    /// Initialize weights with var = 1 / num_markers and set biases to 0.
    fn default_param_init(&mut self, params: &mut BranchParamsHost) {
        let v = 1.0 / self.num_markers as f32;
        let d = Normal::new(0.0, v.sqrt()).unwrap();
        params.set_all_weights_from_distribution(&d, &mut self.rng);
        params.set_all_biases_to_constant(0.0);
        self.remove_markers_from_model(params);
    }

    fn init_weights_with_initial_param_variance(&mut self, params: &mut BranchParamsHost) {
        let v = self.init_param_variance.unwrap();
        let d = Normal::new(0.0, v.sqrt()).unwrap();
        params.set_all_weights_from_distribution(&d, &mut self.rng);
        self.remove_markers_from_model(params);
    }

    fn init_biases_with_initial_param_variance(&mut self, params: &mut BranchParamsHost) {
        let v = self.init_param_variance.unwrap();
        let d = Normal::new(0.0, v.sqrt()).unwrap();
        params.set_all_biases_from_distribution(&d, &mut self.rng);
    }

    fn init_weights_with_init_gamma(&mut self, params: &mut BranchParamsHost) {
        // iterate over layers
        let gamma_params = self.init_gamma_params.as_ref().unwrap();
        let precision_prior = Gamma::new(gamma_params.shape, gamma_params.scale).unwrap();
        for lix in 0..self.num_layers {
            let layer_precision = if self.sample_precisions {
                precision_prior.sample(&mut self.rng)
            } else {
                // set to mean of gamma
                gamma_params.shape * gamma_params.scale
            };
            let layer_std = (1.0 / layer_precision).sqrt();
            let layer_weight_prior = Normal::new(0.0, layer_std).unwrap();
            params.set_layer_weights_from_distribution(lix, &layer_weight_prior, &mut self.rng);
        }
        self.remove_markers_from_model(params);
    }

    fn init_biases_with_init_gamma(&mut self, params: &mut BranchParamsHost) {
        // iterate over layers
        let gamma_params = self.init_gamma_params.as_ref().unwrap();
        let precision_prior = Gamma::new(gamma_params.shape, gamma_params.scale).unwrap();
        for lix in 0..self.num_layers - 1 {
            let layer_bias_precision = if self.sample_precisions {
                precision_prior.sample(&mut self.rng)
            } else {
                gamma_params.shape * gamma_params.scale
            };
            let layer_bias_std = (1.0 / layer_bias_precision).sqrt();
            let layer_bias_prior = Normal::new(0.0, layer_bias_std).unwrap();
            params.set_layer_biases_from_distribution(lix, &layer_bias_prior, &mut self.rng);
        }
    }

    // iterate over weight groups and compute the precision that maximizes the likelihood
    // of the sample (i.e. the prior density)
    fn base_weight_precisions_maximum_likelihood(
        &self,
        params: &BranchParamsHost,
    ) -> Vec<Vec<f32>> {
        let mut weight_precisions = vec![vec![0.0]; self.num_layers];

        for lix in 0..self.num_layers {
            // this is the precision at the extreme point of the prior as a function of the precision
            let layer_precision: f32 = params.num_weights_in_layer(lix) as f32
                / crate::arr_helpers::sum_of_squares(&params.weights[lix]);
            weight_precisions[lix] = vec![layer_precision];
        }

        weight_precisions
    }

    fn base_weight_precisions_fixed(&self) -> Vec<Vec<f32>> {
        vec![
            vec![self
                .fixed_param_precision
                .expect("Fixed bias precisions initialized with empty Option!")];
            self.num_layers
        ]
    }

    // iterate over bias groups and compute the precision that maximizes the likelihood
    // of the sample (i.e. the prior density)
    fn bias_precisions_maximum_likelihood(&self, params: &BranchParamsHost) -> Vec<Vec<f32>> {
        let mut bias_precisions = vec![vec![0.0]; self.num_layers - 1];

        for lix in 0..self.num_layers - 1 {
            let layer_precision: f32 = params.num_biases_in_layer(lix) as f32
                / crate::arr_helpers::sum_of_squares(&params.biases[lix]);
            bias_precisions[lix] = vec![layer_precision];
        }

        bias_precisions
    }

    fn bias_precisions_fixed(&self) -> Vec<Vec<f32>> {
        vec![
            vec![self
                .fixed_param_precision
                .expect("Fixed bias precisions initialized with empty Option!")];
            self.num_layers - 1
        ]
    }

    fn finalize_num_params(&mut self) {
        // summary layer weights + biases
        self.num_params += (self.last_layer_before_summary_width() + 1) * self.summary_layer_width;
        self.num_weights += self.last_layer_before_summary_width() * self.summary_layer_width;
        // output layer weights
        self.num_weights += self.summary_layer_width;
        self.num_params += self.summary_layer_width;

        // summary layer width
        self.layer_widths.push(self.summary_layer_width);
        // output layer width
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
    fn ard_weight_precisions_maximum_likelihood(&self, params: &BranchParamsHost) -> Vec<Vec<f32>> {
        let mut weight_precisions = vec![vec![1.0]; self.num_layers];

        let mut prev_width = self.num_markers;
        for (lix, width) in self.layer_widths[..self.num_ard_layers()]
            .iter()
            .enumerate()
        {
            let mut layer_precisions = vec![0.0; prev_width];
            for ard_group_ix in 0..prev_width {
                layer_precisions[ard_group_ix] = *width as f32
                    / crate::arr_helpers::sum_of_squares(&params.ard_group(lix, ard_group_ix));
            }
            weight_precisions[lix] = layer_precisions;
            prev_width = *width;
        }

        // output layer has to be done jointly for all branches

        weight_precisions
    }

    fn ard_weight_precisions_fixed(&self) -> Vec<Vec<f32>> {
        unimplemented!("ARD type models with fixed param precisions are not implemented. Use a Base type model with fixed precisions instead.")
    }

    fn num_ard_layers(&self) -> usize {
        self.num_layers - 1
    }

    pub fn build_base(mut self) -> BranchCfg {
        let mut res = self.build();
        res.precisions.weight_precisions = if self.fixed_param_precision.is_some() {
            self.base_weight_precisions_fixed()
        } else {
            self.base_weight_precisions_maximum_likelihood(&res.params)
        };
        res
    }

    pub fn build_ard(mut self) -> BranchCfg {
        let mut res = self.build();
        res.precisions.weight_precisions = if self.fixed_param_precision.is_some() {
            self.ard_weight_precisions_fixed()
        } else {
            self.ard_weight_precisions_maximum_likelihood(&res.params)
        };
        res
    }

    fn build(&mut self) -> BranchCfg {
        self.finalize_num_params();

        let mut params = BranchParamsHost::new(self.layer_widths.clone(), self.num_markers);

        if self.initial_weight_value.is_some() {
            self.init_weights_with_initial_weight_value(&mut params);
        } else if self.init_param_variance.is_some() {
            self.init_weights_with_initial_param_variance(&mut params);
            self.init_biases_with_initial_param_variance(&mut params);
        } else if self.init_gamma_params.is_some() {
            self.init_weights_with_init_gamma(&mut params);
            self.init_biases_with_init_gamma(&mut params);
        } else {
            self.default_param_init(&mut params);
        }

        if self.initial_bias_value.is_some() {
            self.init_biases_with_initial_weight_value(&mut params);
        }

        let bias_precisions = if self.fixed_param_precision.is_some() {
            self.bias_precisions_fixed()
        } else {
            self.bias_precisions_maximum_likelihood(&params)
        };

        BranchCfg {
            num_params: self.num_params,
            num_weights: self.num_weights,
            num_markers: self.num_markers,
            layer_widths: self.layer_widths.clone(),
            params,
            precisions: BranchPrecisionsHost {
                weight_precisions: Vec::new(),
                bias_precisions,
                error_precision: vec![2.0],
            },
            activation_function: self.activation_function,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BranchCfgBuilder;
    use crate::net::params::{BranchParamsHost, OutputWeightSummaryStatsHost};

    #[test]
    fn build_branch_cfg() {
        let mut bld = BranchCfgBuilder::new()
            .with_num_markers(3)
            .with_initial_weights_value(0.1);
        bld.add_hidden_layer(3);
        let cfg = bld.build_base();
        assert_eq!(cfg.num_markers, 3);
        assert_eq!(cfg.num_params, 17);
        assert_eq!(cfg.params.layer_widths, vec![3, 1, 1]);
        assert_eq!(cfg.params.num_markers, 3);
        assert_eq!(cfg.params.weights[0].len(), 9);
    }

    // fn af_constant_dims() {
    //     assert_eq!(constant!(1.0; 5).dims()[0], 5);
    // }

    #[test]
    fn remove_markers_from_model() {
        let mut bld = BranchCfgBuilder::new()
            .with_num_markers(4)
            .with_initial_weights_value(0.1)
            .with_seed(12344321)
            .with_proportion_effective_markers(0.25);
        bld.add_hidden_layer(2);
        let cfg = bld.build_base();
        let exp = BranchParamsHost {
            weights: vec![
                vec![0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1],
                vec![0.1, 0.1],
                vec![0.1],
            ],
            biases: vec![vec![0.0, 0.0], vec![0.0]],
            layer_widths: vec![2, 1, 1],
            num_markers: 4,
            output_weight_summary_stats: OutputWeightSummaryStatsHost::default(),
        };
        assert_eq!(cfg.params, exp);
    }
}
