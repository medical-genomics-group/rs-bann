use super::branch::branch::Branch;
use super::branch::branch::BranchCfg;
use arrayfire::{dim4, Array};
use rand::prelude::SliceRandom;
use rand::thread_rng;

/// The full network model
struct Net {
    num_branches: usize,
    branch_cfgs: Vec<BranchCfg>,
}

impl Net {
    // X has to be column major!
    // TODO: X will likely have to be in compressed format on host memory, so Ill have to unpack
    // it before loading it into device memory
    pub fn train(&mut self, x_train: &Vec<Vec<f64>>, y_train: &Vec<f64>, chain_length: usize) {
        let mut acceptance_counts: Vec<usize> = vec![0; self.num_branches];
        let mut rng = thread_rng();
        let num_individuals = y_train.len();
        let mut residual = Array::new(y_train, dim4![num_individuals as u64, 1, 1, 1]);
        let mut branch_ixs = (0..self.num_branches).collect::<Vec<usize>>();
        for ix in 0..chain_length {
            // TODO: update output bias term (Katya says before everything else), including output residual variance?
            // determine random order in which branches are trained
            branch_ixs.shuffle(&mut rng);
            for &branch_ix in &branch_ixs {
                let cfg = &self.branch_cfgs[branch_ix];
                // load marker data onto device
                let x = Array::new(
                    &x_train[branch_ix],
                    dim4!(num_individuals as u64, cfg.num_markers as u64),
                );
                // load branch cfg
                let mut branch = Branch::from_cfg(&self.branch_cfgs[branch_ix]);
                // remove prev contribution from residual
                residual -= branch.predict(&x);
                // TODO: use some input cfg for hmc params
                if branch.hmc_step(&x, &residual, 70, None, 1000.) {
                    acceptance_counts[branch_ix] += 1;
                }
                // update residual
                residual += branch.predict(&x);
                // TODO: update hyperparams!
                // dump branch cfg
                self.branch_cfgs[ix] = branch.to_cfg();
            }
        }
    }
}
