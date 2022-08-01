use super::{
    branch::branch::BranchCfg,
    branch::branch_builder::BranchCfgBuilder,
    net::{Net, OutputBias},
};

/// Number of markers per branch: dynamic
/// Depth of branches: same for all branches
/// Width of branch layers: same within branches, dynamic between branches
pub struct BlockNetCfg {
    num_markers: Vec<usize>,
    depth: usize,
    widths: Vec<usize>,
    precision_prior_shape: f64,
    precision_prior_scale: f64,
}

impl BlockNetCfg {
    pub fn new() -> Self {
        BlockNetCfg {
            num_markers: vec![],
            depth: 0,
            widths: vec![],
            precision_prior_shape: 1.,
            precision_prior_scale: 1.,
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

    pub fn with_precision_prior(mut self, shape: f64, scale: f64) -> Self {
        self.precision_prior_shape = shape;
        self.precision_prior_scale = scale;
        self
    }

    pub fn build_net(&self) -> Net {
        let mut branch_cfgs: Vec<BranchCfg> = Vec::new();
        let num_branches = self.widths.len();
        for branch_ix in 0..num_branches {
            let mut cfg_bld = BranchCfgBuilder::new().with_num_markers(self.num_markers[branch_ix]);
            for _ in 0..self.depth {
                cfg_bld.add_hidden_layer(self.widths[branch_ix]);
            }
            branch_cfgs.push(cfg_bld.build());
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
        }
    }
}
