use super::{
    branch::branch::{Branch, BranchCfg, HMCStepResult},
    gibbs_steps::{multi_param_precision_posterior, single_param_precision_posterior},
    mcmc_cfg::MCMCCfg,
};
use crate::to_host;
use arrayfire::sum_all;
use arrayfire::{dim4, Array};
use log::{debug, info};
use rand::prelude::SliceRandom;
use rand::rngs::ThreadRng;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

pub struct ReportCfg<'data> {
    interval: usize,
    test_data: Option<&'data Data<'data>>,
}

impl<'data> ReportCfg<'data> {
    pub fn new(interval: usize, test_data: Option<&'data Data<'data>>) -> Self {
        Self {
            interval,
            test_data,
        }
    }
}

pub struct Data<'data> {
    x: &'data Vec<Vec<f64>>,
    y: &'data Vec<f64>,
}

impl<'data> Data<'data> {
    pub fn new(x: &'data Vec<Vec<f64>>, y: &'data Vec<f64>) -> Self {
        Data { x, y }
    }
}
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
    pub fn num_params(&self) -> usize {
        let mut res = 0;
        for cfg in &self.branch_cfgs {
            res += cfg.num_params
        }
        res
    }

    pub fn num_branch_params(&self, branch_ix: usize) -> usize {
        self.branch_cfgs[branch_ix].num_params
    }

    // X has to be column major!
    // TODO: X will likely have to be in compressed format on host memory, so Ill have to unpack
    // it before loading it into device memory
    pub fn train(
        &mut self,
        train_data: &Data,
        mcmc_cfg: &MCMCCfg,
        verbose: bool,
        report_cfg: Option<ReportCfg>,
    ) {
        let mut rng = thread_rng();
        let num_individuals = train_data.y.len();
        let mut residual = self.residual(
            train_data.x,
            &Array::new(&train_data.y, dim4!(num_individuals as u64, 1, 1, 1)),
        );
        let mut branch_ixs = (0..self.num_branches).collect::<Vec<usize>>();
        let mut report_interval = 1;

        info!(
            "Training net with {:} branches, {:} params",
            self.num_branches,
            self.num_params()
        );

        // report
        if verbose {
            self.report_training_state(0, train_data, report_cfg.as_ref().unwrap().test_data);
            report_interval = report_cfg.as_ref().unwrap().interval;
        }

        for chain_ix in 1..=mcmc_cfg.chain_length {
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
                debug!("Updating branch {:}", branch_ix);
                let cfg = &self.branch_cfgs[branch_ix];
                // load marker data onto device
                let x = Array::new(
                    &train_data.x[branch_ix],
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

            // report
            if verbose && chain_ix % report_interval == 0 {
                self.report_training_state(
                    chain_ix,
                    train_data,
                    report_cfg.as_ref().unwrap().test_data,
                );
            }
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

    pub fn rss(&self, data: &Data) -> f64 {
        let y_test_arr = Array::new(data.y, dim4!(data.y.len() as u64, 1, 1, 1));
        let y_hat = self.predict_arr(&data.x, data.y.len());
        let residual = y_test_arr - y_hat;
        super::gibbs_steps::sum_of_squares(&residual)
    }

    fn report_training_state(&self, iteration: usize, train_data: &Data, test_data: Option<&Data>) {
        if let Some(tst) = test_data {
            info!(
                "iteration: {:} \t | acc: {:.2} \t | early_rej: {:.2} \t | end_rej: {:.2} \t | loss(trn): {:.4} \t | loss(tst): {:.4}",
                iteration,
                self.training_stats.acceptance_rate(),
                self.training_stats.early_rejection_rate(),
                self.training_stats.end_rejection_rate(),
                self.rss(train_data),
                self.rss(tst));
        } else {
            info!(
                "iteration: {:} \t | acc: {:.2} \t | early_rej: {:.2} \t | end_rej: {:.2} \t | loss(trn): {:.4}",
                iteration,
                self.training_stats.acceptance_rate(),
                self.training_stats.early_rejection_rate(),
                self.training_stats.end_rejection_rate(),
                self.rss(train_data));
        }
    }

    fn note_hmc_step_result(&mut self, res: HMCStepResult) {
        self.training_stats.add_hmc_step_result(res);
    }

    fn reset_training_stats(&mut self) {
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
