use super::branch::BranchCfg;
use super::params::BranchHyperparams;
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;

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
    use super::BranchCfgBuilder;

    #[test]
    fn test_build_branch_cfg() {
        let mut bld = BranchCfgBuilder::new().with_num_markers(3);
        bld.add_hidden_layer(3);
        let cfg = bld.build();
        assert_eq!(cfg.num_markers, 3);
        assert_eq!(cfg.num_params, 17);
    }
}
