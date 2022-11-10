use super::{
    branch::branch::Branch,
    branch::branch::BranchCfg,
    branch::branch_cfg_builder::BranchCfgBuilder,
    net::{Net, OutputBias},
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
    init_param_variance: f32,
    init_gamma_shape: Option<f32>,
    init_gamma_scale: Option<f32>,
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
            init_param_variance: 0.05,
            init_gamma_shape: None,
            init_gamma_scale: None,
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

    pub fn with_init_param_variance(mut self, val: f32) -> Self {
        self.init_param_variance = val;
        self
    }

    pub fn with_init_gamma_params(mut self, shape: f32, scale: f32) -> Self {
        self.init_gamma_shape = Some(shape);
        self.init_gamma_scale = Some(scale);
        self
    }

    pub fn build_net(&self) -> Net<B> {
        let mut branch_cfgs: Vec<BranchCfg> = Vec::new();
        let num_branches = self.widths.len();
        for branch_ix in 0..num_branches {
            let mut cfg_bld =
                if let (Some(k), Some(s)) = (self.init_gamma_shape, self.init_gamma_scale) {
                    BranchCfgBuilder::new()
                        .with_num_markers(self.num_markers[branch_ix])
                        .with_init_gamma_params(k, s)
                } else {
                    BranchCfgBuilder::new()
                        .with_num_markers(self.num_markers[branch_ix])
                        .with_init_param_variance(self.init_param_variance)
                };
            for _ in 0..self.depth {
                cfg_bld.add_hidden_layer(self.widths[branch_ix]);
            }
            branch_cfgs.push(B::build_cfg(cfg_bld));
        }
        Net::new(
            self.precision_prior_shape,
            self.precision_prior_scale,
            num_branches,
            branch_cfgs,
            OutputBias {
                precision: 1.0,
                bias: 0.0,
            },
            1.0,
        )
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
        assert_eq!(net.branch_cfg(0).num_params, 17);
        assert_eq!(net.branch_cfg(1).num_params, 17);
    }
}
