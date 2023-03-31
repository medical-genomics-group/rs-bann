use crate::arr_helpers::sum_of_squares;

use super::{
    activation_functions::ActivationFunction,
    branch::branch_cfg::BranchCfg,
    branch::branch_cfg_builder::BranchCfgBuilder,
    branch::branch_sampler::BranchSampler,
    net::{Net, OutputBias},
    params::{
        GlobalParams, NetworkPrecisionHyperparameters, OutputWeightSummaryStatsHost,
        PrecisionHyperparameters,
    },
};
use std::marker::PhantomData;

const DEFAULT_INIT_OUTPUT_LAYER_PRECISION: f32 = 0.05;
pub enum HiddenLayerWidthRule {
    Fixed(usize),
    FractionOfInput(f32),
}

pub enum SummaryLayerWidthRule {
    Fixed(usize),
    LikeHiddenLayerWidth,
    FractionOfHiddenLayerWidth(f32),
}

/// Number of markers per branch: dynamic
/// Depth of branches: same for all branches
/// Width of branch layers: same within branches, dynamic between branches
pub struct BlockNetCfg<B: BranchSampler> {
    hidden_layer_width_rule: HiddenLayerWidthRule,
    summary_layer_width_rule: SummaryLayerWidthRule,
    num_markers: Vec<usize>,
    num_hidden_layers: usize,
    hidden_layer_widths: Vec<usize>,
    summary_layer_widths: Vec<usize>,
    dense_precision_prior_hyperparams: PrecisionHyperparameters,
    summary_precision_prior_hyperparams: PrecisionHyperparameters,
    output_precision_prior_hyperparams: PrecisionHyperparameters,
    init_param_variance: Option<f32>,
    init_gamma_shape: Option<f32>,
    init_gamma_scale: Option<f32>,
    num_effective_markers: Option<usize>,
    proportion_effective_markers: Option<f32>,
    output_weight_summary_stats: OutputWeightSummaryStatsHost,
    fixed_param_precision: Option<f32>,
    branch_type: PhantomData<B>,
    activation_function: ActivationFunction,
}

impl<B: BranchSampler> Default for BlockNetCfg<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: BranchSampler> BlockNetCfg<B> {
    pub fn new() -> Self {
        BlockNetCfg {
            hidden_layer_width_rule: HiddenLayerWidthRule::FractionOfInput(0.5),
            summary_layer_width_rule: SummaryLayerWidthRule::LikeHiddenLayerWidth,
            num_markers: vec![],
            // summary layer is not counted as hidden layer
            num_hidden_layers: 0,
            hidden_layer_widths: vec![],
            summary_layer_widths: vec![],
            dense_precision_prior_hyperparams: PrecisionHyperparameters::default(),
            summary_precision_prior_hyperparams: PrecisionHyperparameters::default(),
            output_precision_prior_hyperparams: PrecisionHyperparameters::default(),
            init_param_variance: None,
            init_gamma_shape: None,
            init_gamma_scale: None,
            num_effective_markers: None,
            proportion_effective_markers: None,
            output_weight_summary_stats: OutputWeightSummaryStatsHost::default(),
            fixed_param_precision: None,
            branch_type: PhantomData,
            activation_function: ActivationFunction::Tanh,
        }
    }

    pub fn with_activation_function(mut self, af: ActivationFunction) -> Self {
        self.activation_function = af;
        self
    }

    pub fn with_fixed_param_precision(mut self, precision: Option<f32>) -> Self {
        self.fixed_param_precision = precision;
        self
    }

    pub fn add_branch(&mut self, num_markers: usize) {
        self.num_markers.push(num_markers);
        match self.hidden_layer_width_rule {
            HiddenLayerWidthRule::Fixed(width) => self.hidden_layer_widths.push(width),
            HiddenLayerWidthRule::FractionOfInput(fraction) => self
                .hidden_layer_widths
                // make sure that width doesn't go below 1
                .push(((num_markers as f32 * fraction) as usize).max(1)),
        }

        let summary_layer_width = match self.summary_layer_width_rule {
            SummaryLayerWidthRule::Fixed(width) => {
                assert!(
                    width != 0,
                    "Branch cannot be initiated with summary layer width = 0."
                );
                width
            }
            SummaryLayerWidthRule::FractionOfHiddenLayerWidth(fraction) => {
                ((*self.hidden_layer_widths.last().unwrap() as f32 * fraction) as usize).max(1)
            }
            SummaryLayerWidthRule::LikeHiddenLayerWidth => {
                *self.hidden_layer_widths.last().unwrap()
            }
        };

        self.summary_layer_widths.push(summary_layer_width);
        self.output_weight_summary_stats
            .incr_num_params(summary_layer_width);
    }

    pub fn with_hidden_layer_width_rule(mut self, hlwr: HiddenLayerWidthRule) -> Self {
        self.hidden_layer_width_rule = hlwr;
        self
    }

    pub fn with_summary_layer_width_rule(mut self, slwr: SummaryLayerWidthRule) -> Self {
        self.summary_layer_width_rule = slwr;
        self
    }

    pub fn with_num_hidden_layers(mut self, depth: usize) -> Self {
        self.num_hidden_layers = depth;
        self
    }

    pub fn with_proportion_effective_markers(mut self, proportion: Option<f32>) -> Self {
        self.proportion_effective_markers = proportion;
        self
    }

    pub fn with_num_effective_markers(mut self, num: Option<usize>) -> Self {
        self.num_effective_markers = num;
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
        self.init_param_variance = Some(val);
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
                .map(|c| c.output_layer_weights())
                .map(sum_of_squares)
                .sum::<f32>();
        cfgs.iter_mut().for_each(|c| {
            *c.precisions.weight_precisions.last_mut().unwrap() = vec![output_weight_precision]
        });
    }

    pub fn build_net(&mut self) -> Net<B> {
        let mut branch_cfgs: Vec<BranchCfg> = Vec::new();
        let num_branches = self.hidden_layer_widths.len();
        for branch_ix in 0..num_branches {
            let mut cfg_bld = BranchCfgBuilder::new()
                .with_num_markers(self.num_markers[branch_ix])
                .with_num_effective_markers(self.num_effective_markers)
                .with_proportion_effective_markers(self.proportion_effective_markers)
                .with_summary_layer_width(self.summary_layer_widths[branch_ix])
                .with_fixed_param_precision(self.fixed_param_precision)
                .with_activation_function(self.activation_function);
            cfg_bld = if let (Some(k), Some(s)) = (self.init_gamma_shape, self.init_gamma_scale) {
                cfg_bld.with_init_gamma_params(k, s)
            } else if let Some(v) = self.init_param_variance {
                cfg_bld.with_init_param_variance(v)
            } else {
                cfg_bld
            };
            for _ in 0..self.num_hidden_layers {
                cfg_bld.add_hidden_layer(self.hidden_layer_widths[branch_ix]);
            }
            branch_cfgs.push(B::build_cfg(cfg_bld));
            self.output_weight_summary_stats
                .incr_reg_sum(B::summary_stat_fn_host(
                    branch_cfgs.last().unwrap().output_layer_weights(),
                ));
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
                error_precision: 2.0,
                precision: 1.0,
                bias: 0.0,
            },
            GlobalParams {
                error_precision: 2.0,
                output_layer_precision: self
                    .fixed_param_precision
                    .unwrap_or(DEFAULT_INIT_OUTPUT_LAYER_PRECISION),
                output_weight_summary_stats: self.output_weight_summary_stats,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::branch::ridge_base::RidgeBaseBranch;
    use super::{BlockNetCfg, HiddenLayerWidthRule, SummaryLayerWidthRule};

    #[test]
    fn block_net_architecture_num_params_in_branch() {
        let mut cfg = BlockNetCfg::<RidgeBaseBranch>::new()
            .with_num_hidden_layers(1)
            .with_hidden_layer_width_rule(HiddenLayerWidthRule::Fixed(3))
            .with_summary_layer_width_rule(SummaryLayerWidthRule::Fixed(2));
        cfg.add_branch(3);
        cfg.add_branch(3);
        let net = cfg.build_net();
        assert_eq!(net.branch_cfg(0).num_params, 22);
        assert_eq!(net.branch_cfg(1).num_params, 22);
    }
}
