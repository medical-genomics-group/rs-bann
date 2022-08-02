use super::{
    branch::branch::{Branch, BranchCfg, HMCStepResult},
    gibbs_steps::{multi_param_precision_posterior, single_param_precision_posterior},
    mcmc_cfg::MCMCCfg,
};
use crate::to_host;
use arrayfire::sum_all;
use arrayfire::{dim4, Array};
use log::info;
use rand::prelude::SliceRandom;
use rand::rngs::ThreadRng;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

pub(crate) struct TrainingStats {
    num_samples: usize,
    num_accepted: usize,
    num_early_rejected: usize,
}

impl TrainingStats {
    pub(crate) fn new() -> Self {
        Self {
            num_samples: 0,
            num_accepted: 0,
            num_early_rejected: 0,
        }
    }

    fn add_hmc_step_result(&mut self, res: HMCStepResult) {
        self.num_samples += 1;
        match res {
            HMCStepResult::Accepted => self.num_accepted += 1,
            HMCStepResult::RejectedEarly => self.num_early_rejected += 1,
            HMCStepResult::Rejected => {}
        }
    }

    fn print_stats(&self) {
        info!(
            "acc_rate: {:?} \t | early_reject_rate: {:?} \t | end_reject_rate: {:?}",
            self.acceptance_rate(),
            self.early_rejection_rate(),
            self.end_rejection_rate()
        );
    }

    fn acceptance_rate(&self) -> f64 {
        self.num_accepted as f64 / self.num_samples as f64
    }

    fn early_rejection_rate(&self) -> f64 {
        self.num_early_rejected as f64 / self.num_samples as f64
    }

    fn end_rejection_rate(&self) -> f64 {
        (self.num_samples - self.num_early_rejected - self.num_accepted) as f64
            / self.num_samples as f64
    }
}

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
    pub(crate) training_stats: TrainingStats,
}

impl Net {
    // X has to be column major!
    // TODO: X will likely have to be in compressed format on host memory, so Ill have to unpack
    // it before loading it into device memory
    pub fn train(&mut self, x_train: &Vec<Vec<f64>>, y_train: &Vec<f64>, mcmc_cfg: &MCMCCfg) {
        let mut rng = thread_rng();
        let num_individuals = y_train.len();
        let mut residual = self.residual(
            x_train,
            &Array::new(&y_train, dim4!(num_individuals as u64, 1, 1, 1)),
        );
        let mut branch_ixs = (0..self.num_branches).collect::<Vec<usize>>();
        for _ in 0..mcmc_cfg.chain_length {
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
                // TODO: save last prediction contribution for each branch to reduce compute
                residual += branch.predict(&x);
                self.note_hmc_step_result(branch.hmc_step(&x, &residual, &mcmc_cfg));
                branch.sample_precisions(self.precision_prior_shape, self.precision_prior_scale);
                // update residual
                residual -= branch.predict(&x);
                // dump branch cfg
                self.branch_cfgs[branch_ix] = branch.to_cfg();
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

    fn residual(&self, x: &Vec<Vec<f64>>, y: &Array<f64>) -> Array<f64> {
        y - self.predict_arr(x, y.elements())
    }

    // TODO: predict using posterior predictive distribution instead of point estimate
    fn predict_arr(&self, x_test: &Vec<Vec<f64>>, num_individuals: usize) -> Array<f64> {
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
        y_hat
    }

    pub fn rss(&self, x_test: &Vec<Vec<f64>>, y_test: &Vec<f64>) -> f64 {
        let y_test_arr = Array::new(y_test, dim4!(y_test.len() as u64, 1, 1, 1));
        let y_hat = self.predict_arr(&x_test, y_test.len());
        let residual = y_test_arr - y_hat;
        super::gibbs_steps::sum_of_squares(&residual)
    }

    pub fn print_training_stats(&self) {
        self.training_stats.print_stats();
    }

    fn note_hmc_step_result(&mut self, res: HMCStepResult) {
        self.training_stats.add_hmc_step_result(res);
    }

    pub fn reset_training_stats(&mut self) {
        self.training_stats = TrainingStats::new();
    }
}

#[cfg(test)]
mod tests {
    use crate::to_host;
    use arrayfire::{dim4, Array, MatProp};

    #[test]
    fn test_arrayfire_dot_expected_result() {
        // this should compute the sum of squares of two arrays, right?
        let a = Array::new(&[1.0, 2.0, 3.0], dim4!(3, 1, 1, 1));
        let b = Array::new(&[3.0, 2.0, 1.0], dim4!(3, 1, 1, 1));
        let exp = 10.0;
        let dotp = arrayfire::dot(&a, &b, MatProp::NONE, MatProp::NONE);
        assert_eq!(to_host(&dotp)[0], exp);
    }
}
