use super::{
    branch::branch::Branch,
    branch::branch::BranchCfg,
    branch::branch_cfg_builder::BranchCfgBuilder,
    net::{Net, OutputBias},
    params::{NetworkPrecisionHyperparameters, PrecisionHyperparameters},
};
use std::marker::PhantomData;

/// Number of markers per branch: dynamic
/// Depth of branches: same for all branches
/// Width of branch layers: same within branches, dynamic between branches
pub struct BlockNetCfg<B: Branch> {
    num_markers: Vec<usize>,
    num_hidden_layers: usize,
    hidden_layer_widths: Vec<usize>,
    summary_layer_widths: Vec<usize>,
    dense_precision_prior_hyperparams: PrecisionHyperparameters,
    summary_precision_prior_hyperparams: PrecisionHyperparameters,
    output_precision_prior_hyperparams: PrecisionHyperparameters,
    init_param_variance: f32,
    init_gamma_shape: Option<f32>,
    init_gamma_scale: Option<f32>,
    proportion_effective_markers: f32,
    branch_type: PhantomData<B>,
}

impl<B: Branch> Default for BlockNetCfg<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Branch> BlockNetCfg<B> {
    pub fn new() -> Self {
        BlockNetCfg {
            num_markers: vec![],
            // summary layer is not counted as hidden layer
            num_hidden_layers: 0,
            hidden_layer_widths: vec![],
            summary_layer_widths: vec![],
            dense_precision_prior_hyperparams: PrecisionHyperparameters::default(),
            summary_precision_prior_hyperparams: PrecisionHyperparameters::default(),
            output_precision_prior_hyperparams: PrecisionHyperparameters::default(),
            init_param_variance: 0.05,
            init_gamma_shape: None,
            init_gamma_scale: None,
            proportion_effective_markers: 1.0,
            branch_type: PhantomData,
        }
    }

    pub fn add_branch(
        &mut self,
        num_markers: usize,
        hidden_layer_width: usize,
        summary_layer_width: usize,
    ) {
        self.num_markers.push(num_markers);
        self.hidden_layer_widths.push(hidden_layer_width);
        assert!(
            summary_layer_width != 0,
            "Branch cannot be initiated with summary layer width = 0."
        );
        self.summary_layer_widths.push(summary_layer_width);
    }

    pub fn with_num_hidden_layers(mut self, depth: usize) -> Self {
        self.num_hidden_layers = depth;
        self
    }

    pub fn with_proportion_effective_markers(mut self, proportion: f32) -> Self {
        self.proportion_effective_markers = proportion;
        self
    }

    pub fn with_dense_precision_prior(mut self, shape: f32, scale: f32) -> Self {
        self.dense_precision_prior_hyperparams = PrecisionHyperparameters::new(shape, scale);
        self
    }

    pub fn with_summary_precision_prior(mut self, shape: f32, scale: f32) -> Self {
        self.summary_precision_prior_hyperparams = PrecisionHyperparameters::new(shape, scale);
        self
    }

    pub fn with_output_precision_prior(mut self, shape: f32, scale: f32) -> Self {
        self.output_precision_prior_hyperparams = PrecisionHyperparameters::new(shape, scale);
        self
    }

    pub fn with_init_param_variance(mut self, val: f32) -> Self {
        self.init_param_variance = val;
        self
    }

    pub fn with_init_gamma_params(mut self, shape: f32, scale: f32) -> Self {
        self.init_gamma_shape = Some(shape);
        self.init_gamma_scale = Some(scale);
        self
    }

    fn update_branch_cfgs_output_weight_precision(&self, cfgs: &mut Vec<BranchCfg>) {
        let output_weight_precision = cfgs.len() as f32
            / cfgs
                .iter()
                .map(|c| c.params[c.num_weights - 1])
                .map(|e| e * e)
                .sum::<f32>();
        cfgs.iter_mut().for_each(|c| {
            *c.precisions.weight_precisions.last_mut().unwrap() =
                arrayfire::constant!(output_weight_precision; 1)
        });
    }

    pub fn build_net(&self) -> Net<B> {
        let mut branch_cfgs: Vec<BranchCfg> = Vec::new();
        let num_branches = self.hidden_layer_widths.len();
        for branch_ix in 0..num_branches {
            let mut cfg_bld =
                if let (Some(k), Some(s)) = (self.init_gamma_shape, self.init_gamma_scale) {
                    BranchCfgBuilder::new()
                        .with_num_markers(self.num_markers[branch_ix])
                        .with_proportion_effective_markers(self.proportion_effective_markers)
                        .with_init_gamma_params(k, s)
                        .with_summary_layer_width(self.summary_layer_widths[branch_ix])
                } else {
                    BranchCfgBuilder::new()
                        .with_num_markers(self.num_markers[branch_ix])
                        .with_proportion_effective_markers(self.proportion_effective_markers)
                        .with_init_param_variance(self.init_param_variance)
                        .with_summary_layer_width(self.summary_layer_widths[branch_ix])
                };
            for _ in 0..self.num_hidden_layers {
                cfg_bld.add_hidden_layer(self.hidden_layer_widths[branch_ix]);
            }
            branch_cfgs.push(B::build_cfg(cfg_bld));
        }
        self.update_branch_cfgs_output_weight_precision(&mut branch_cfgs);
        Net::new(
            NetworkPrecisionHyperparameters {
                dense: self.dense_precision_prior_hyperparams.clone(),
                summary: self.summary_precision_prior_hyperparams.clone(),
                output: self.output_precision_prior_hyperparams.clone(),
            },
            num_branches,
            branch_cfgs,
            OutputBias {
                error_precision: 1.0,
                precision: 1.0,
                bias: 0.0,
            },
            1.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::branch::ridge_base::RidgeBaseBranch;
    use super::BlockNetCfg;

    #[test]
    fn test_block_net_architecture_num_params_in_branch() {
        let mut cfg = BlockNetCfg::<RidgeBaseBranch>::new().with_num_hidden_layers(1);
        cfg.add_branch(3, 3, 1);
        cfg.add_branch(3, 3, 2);
        let net = cfg.build_net();
        assert_eq!(net.branch_cfg(0).num_params, 17);
        assert_eq!(net.branch_cfg(1).num_params, 22);
    }
}
