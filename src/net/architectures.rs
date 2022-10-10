use super::{
    branch::branch::Branch,
    branch::branch::BranchCfg,
    branch::branch_cfg_builder::BranchCfgBuilder,
    net::{Net, OutputBias},
    train_stats::TrainingStats,
};
use std::marker::PhantomData;

/// Number of markers per branch: dynamic
/// Depth of branches: same for all branches
/// Width of branch layers: same within branches, dynamic between branches
pub struct BlockNetCfg<B: Branch> {
    num_markers: Vec<usize>,
    depth: usize,
    widths: Vec<usize>,
    precision_prior_shape: f32,
    precision_prior_scale: f32,
    initial_random_range: f32,
    branch_type: PhantomData<B>,
}

impl<B: Branch> BlockNetCfg<B> {
    pub fn new() -> Self {
        BlockNetCfg {
            num_markers: vec![],
            // TODO: rename! this is not network depth, but the number
            // of hidden layers in a single branch, i.e. depth -2
            depth: 0,
            widths: vec![],
            precision_prior_shape: 1.,
            precision_prior_scale: 1.,
            initial_random_range: 0.05,
            branch_type: PhantomData,
        }
    }

    pub fn add_branch(&mut self, num_markers: usize, width: usize) {
        self.num_markers.push(num_markers);
        self.widths.push(width);
    }

    pub fn with_depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }

    pub fn with_precision_prior(mut self, shape: f32, scale: f32) -> Self {
        self.precision_prior_shape = shape;
        self.precision_prior_scale = scale;
        self
    }

    pub fn with_initial_random_range(mut self, val: f32) -> Self {
        self.initial_random_range = val;
        self
    }

    pub fn build_net(&self) -> Net<B> {
        let mut branch_cfgs: Vec<BranchCfg> = Vec::new();
        let num_branches = self.widths.len();
        for branch_ix in 0..num_branches {
            let mut cfg_bld = BranchCfgBuilder::new()
                .with_num_markers(self.num_markers[branch_ix])
                .with_initial_random_range(self.initial_random_range);
            for _ in 0..self.depth {
                cfg_bld.add_hidden_layer(self.widths[branch_ix]);
            }
            branch_cfgs.push(B::build_cfg(cfg_bld));
        }
        Net {
            precision_prior_shape: self.precision_prior_shape,
            precision_prior_scale: self.precision_prior_scale,
            num_branches,
            branch_cfgs,
            output_bias: OutputBias {
                precision: 1.0,
                bias: 0.0,
            },
            error_precision: 1.0,
            training_stats: TrainingStats::new(),
            branch_type: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::branch::base_branch::BaseBranch;
    use super::BlockNetCfg;

    #[test]
    fn test_block_net_architecture_num_params_in_branch() {
        let mut cfg = BlockNetCfg::<BaseBranch>::new().with_depth(1);
        cfg.add_branch(3, 3);
        cfg.add_branch(3, 3);
        let net = cfg.build_net();
        assert_eq!(net.branch_cfgs[0].num_params, 17);
        assert_eq!(net.branch_cfgs[1].num_params, 17);
    }
}
