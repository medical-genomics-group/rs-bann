use super::branch::branch::Branch;
use super::branch::branch::BranchCfg;
use super::gibbs_steps::multi_param_precision_posterior;
use super::gibbs_steps::single_param_precision_posterior;
use crate::to_host;
use arrayfire::sum_all;
use arrayfire::{dim4, Array};
use rand::prelude::SliceRandom;
use rand::rngs::ThreadRng;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

pub struct OutputBias {
    pub(crate) precision: f64,
    pub(crate) bias: f64,
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
pub struct Net {
    pub(crate) precision_prior_shape: f64,
    pub(crate) precision_prior_scale: f64,
    pub(crate) num_branches: usize,
    pub(crate) branch_cfgs: Vec<BranchCfg>,
    pub(crate) output_bias: OutputBias,
    pub(crate) error_precision: f64,
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
                let mut branch = Branch::from_cfg(&cfg);
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

    // TODO: predict using posterior predictive distribution instead of point estimate
    pub fn predict(&self, x_test: &Vec<Vec<f64>>, num_individuals: usize) -> Vec<f64> {
        // I expect X to be column major
        let mut y_hat = Array::new(
            &vec![0.0; num_individuals],
            dim4![num_individuals as u64, 1, 1, 1],
        );
        // add bias
        y_hat += self.output_bias.af_bias();
        // add all branch predictions
        for branch_ix in 0..self.num_branches {
            let cfg = &self.branch_cfgs[branch_ix];
            let x = Array::new(
                &x_test[branch_ix],
                dim4!(num_individuals as u64, cfg.num_markers as u64),
            );
            y_hat += Branch::from_cfg(&cfg).predict(&x);
        }
        to_host(&y_hat)
    }
}
