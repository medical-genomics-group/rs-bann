use super::branch::branch::Branch;
use super::branch::branch::BranchCfg;
use super::gibbs_steps::multi_param_precision_posterior;
use super::gibbs_steps::single_param_precision_posterior;
use arrayfire::sum_all;
use arrayfire::Scalar;
use arrayfire::{dim4, Array};
use rand::prelude::SliceRandom;
use rand::rngs::ThreadRng;
use rand::thread_rng;
use rand_distr::{Distribution, Gamma, Normal};

struct OutputBias {
    precision: f64,
    bias: f64,
}

impl OutputBias {
    fn sample_bias(
        &mut self,
        error_precision: f64,
        residual: &Array<f64>,
        n: usize,
        rng: &mut ThreadRng,
    ) {
        let (sum_r, _) = sum_all(residual);
        let nu = error_precision / (n as f64 * error_precision + self.precision);
        let mean = nu * sum_r;
        let std = (1. / (n as f64 * error_precision + self.precision)).sqrt();
        self.bias = Normal::new(mean, std).unwrap().sample(rng);
    }

    fn sample_precision(&mut self, prior_shape: f64, prior_scale: f64, rng: &mut ThreadRng) {
        self.precision = single_param_precision_posterior(prior_shape, prior_scale, self.bias, rng);
    }

    fn af_bias(&self) -> Array<f64> {
        Array::new(&[self.bias], dim4!(1, 1, 1, 1))
    }
}

/// The full network model
struct Net {
    precision_prior_shape: f64,
    precision_prior_scale: f64,
    num_branches: usize,
    branch_cfgs: Vec<BranchCfg>,
    output_bias: OutputBias,
    error_precision: f64,
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
            // sample ouput bias term
            residual += self.output_bias.af_bias();
            self.output_bias.sample_bias(
                self.error_precision,
                &residual,
                num_individuals,
                &mut rng,
            );
            self.output_bias.sample_precision(
                self.precision_prior_shape,
                self.precision_prior_scale,
                &mut rng,
            );
            residual -= self.output_bias.af_bias();
            // shuffle order in which branches are trained
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
                // tell branch about global error precision
                branch.set_error_precision(self.error_precision);
                // remove prev contribution from residual
                residual += branch.predict(&x);
                // TODO: use some input cfg for hmc params
                if branch.hmc_step(&x, &residual, 70, None, 1000.) {
                    acceptance_counts[branch_ix] += 1;
                }
                branch.sample_precisions(self.precision_prior_shape, self.precision_prior_scale);
                // update residual
                residual -= branch.predict(&x);
                // dump branch cfg
                self.branch_cfgs[ix] = branch.to_cfg();
            }
            // update error precision
            self.error_precision = multi_param_precision_posterior(
                self.precision_prior_shape,
                self.precision_prior_scale,
                &residual,
                &mut rng,
            );
        }
    }
}
