use super::{
    branch::branch::{Branch, BranchCfg, BranchMeta, HMCStepResult},
    data::{Data, Genotypes},
    gibbs_steps::{multi_param_precision_posterior, single_param_precision_posterior},
    mcmc_cfg::MCMCCfg,
    train_stats::{ReportCfg, TrainingStats},
};
use crate::to_host;
use arrayfire::{dim4, sum_all, Array};
use bincode::{deserialize_from, serialize_into};
use log::{debug, info};
use rand::{prelude::SliceRandom, rngs::ThreadRng, thread_rng};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use serde_json::to_writer;
use std::path::Path;
use std::{
    fmt::{Display, Formatter},
    fs::File,
    io::{BufReader, BufWriter, Write},
    marker::PhantomData,
};

#[derive(clap::ValueEnum, Clone, Debug, Serialize, Deserialize)]
pub enum ModelType {
    ARD,
    Base,
    StdNormal,
}

impl Display for ModelType {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
        // or, alternatively:
        // fmt::Debug::fmt(self, f)
    }
}

#[derive(Serialize, Deserialize)]
pub struct OutputBias {
    pub(crate) precision: f32,
    pub(crate) bias: f32,
}

impl OutputBias {
    fn sample_bias(
        &mut self,
        error_precision: f32,
        residual: &Array<f32>,
        n: usize,
        rng: &mut ThreadRng,
    ) {
        let (sum_r, _) = sum_all(residual);
        let nu = error_precision / (n as f32 * error_precision + self.precision);
        let mean = nu * sum_r;
        let std = (1. / (n as f32 * error_precision + self.precision)).sqrt();
        self.bias = Normal::new(mean, std).unwrap().sample(rng);
    }

    fn sample_precision(&mut self, prior_shape: f32, prior_scale: f32, rng: &mut ThreadRng) {
        self.precision = single_param_precision_posterior(prior_shape, prior_scale, self.bias, rng);
    }

    fn af_bias(&self) -> Array<f32> {
        Array::new(&[self.bias], dim4!(1, 1, 1, 1))
    }
}

#[derive(Serialize, Deserialize)]
/// The full network model
pub struct Net<B: Branch> {
    pub(crate) precision_prior_shape: f32,
    pub(crate) precision_prior_scale: f32,
    pub(crate) num_branches: usize,
    pub(crate) branch_cfgs: Vec<BranchCfg>,
    pub(crate) output_bias: OutputBias,
    pub(crate) error_precision: f32,
    pub(crate) training_stats: TrainingStats,
    pub(crate) branch_type: PhantomData<B>,
}

impl<B: Branch> Net<B> {
    pub fn from_file(path: &Path) -> Self {
        let mut r = BufReader::new(File::open(path).unwrap());
        deserialize_from(&mut r).unwrap()
    }

    pub fn to_file(&self, path: &Path) {
        let mut f = BufWriter::new(File::create(path).unwrap());
        serialize_into(&mut f, self).unwrap();
    }

    pub fn model_type(&self) -> ModelType {
        B::model_type()
    }

    pub fn branch_cfg(&self, branch_ix: usize) -> &BranchCfg {
        &self.branch_cfgs[branch_ix]
    }

    pub fn branch_cfgs(&self) -> &Vec<BranchCfg> {
        &self.branch_cfgs
    }

    pub fn num_params(&self) -> usize {
        let mut res = 0;
        for cfg in &self.branch_cfgs {
            res += cfg.num_params
        }
        res
    }

    pub fn num_branches(&self) -> usize {
        self.num_branches
    }

    pub fn num_branch_params(&self, branch_ix: usize) -> usize {
        self.branch_cfgs[branch_ix].num_params
    }

    // TODO: do not assume that all branches are the same!
    pub fn write_meta(&self, mcmc_cfg: &MCMCCfg) {
        let w = BufWriter::new(File::create(mcmc_cfg.meta_path()).unwrap());
        to_writer(
            w,
            &BranchMeta::from_cfg(
                &self.branch_cfgs[0],
                self.precision_prior_shape,
                self.precision_prior_scale,
            ),
        )
        .unwrap();
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
        let mut trace_file = None;
        if mcmc_cfg.trace {
            trace_file = Some(BufWriter::new(File::create(mcmc_cfg.trace_path()).unwrap()));
        }

        let mut rng = thread_rng();
        let num_individuals = train_data.num_individuals();
        let mut residual = self.residual(
            &train_data.x(),
            &Array::new(&train_data.y(), dim4!(num_individuals as u64, 1, 1, 1)),
        );
        let mut branch_ixs = (0..self.num_branches).collect::<Vec<usize>>();
        let mut report_interval = 1;

        info!(
            "Training net with {:} branches, {:} params",
            self.num_branches,
            self.num_params()
        );

        // initial loss
        self.record_mse(train_data, report_cfg.as_ref().unwrap().test_data);

        // report
        if verbose {
            self.report_training_state(0);
            report_interval = report_cfg.as_ref().unwrap().interval;
        }

        if mcmc_cfg.trace {
            to_writer(trace_file.as_mut().unwrap(), &self.branch_cfgs).unwrap();
            trace_file.as_mut().unwrap().write_all(b"\n").unwrap();
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
                    &train_data.x()[branch_ix],
                    dim4!(num_individuals as u64, cfg.num_markers as u64),
                );
                // load branch cfg
                let mut branch = B::from_cfg(cfg);
                // tell branch about global error precision
                branch.set_error_precision(self.error_precision);
                // TODO: save last prediction contribution for each branch to reduce compute
                let prev_pred = branch.predict(&x);
                residual = &residual + &prev_pred;
                let step_res = branch.hmc_step(&x, &residual, mcmc_cfg);
                self.note_hmc_step_result(&step_res);
                match step_res {
                    // update residual
                    HMCStepResult::Accepted(state) => residual -= state.y_pred,
                    // not accepted, just remove previous prediction
                    _ => residual -= prev_pred,
                }
                branch.sample_precisions(self.precision_prior_shape, self.precision_prior_scale);

                // dump branch cfg
                self.branch_cfgs[branch_ix] = branch.to_cfg();
            }

            // TODO:
            // this can be easily done without predicting again,
            // just by saving the last predictions of each branch
            // and combining them.
            self.record_mse(train_data, report_cfg.as_ref().unwrap().test_data);

            // update error precision
            self.error_precision = multi_param_precision_posterior(
                self.precision_prior_shape,
                self.precision_prior_scale,
                &residual,
                &mut rng,
            );

            // report
            if verbose && chain_ix % report_interval == 0 {
                self.report_training_state(chain_ix);
            }

            if mcmc_cfg.trace {
                to_writer(trace_file.as_mut().unwrap(), &self.branch_cfgs).unwrap();
                trace_file.as_mut().unwrap().write_all(b"\n").unwrap();
            }
        }

        info!("Completed training");
        // save training stats
        self.training_stats.to_file(&mcmc_cfg.outpath);
    }

    // TODO: predict using posterior predictive distribution instead of point estimate
    pub fn predict(&self, gen: &Genotypes) -> Vec<f32> {
        // I expect X to be column major
        let mut y_hat = Array::new(
            &vec![0.0; gen.num_individuals()],
            dim4![gen.num_individuals() as u64, 1, 1, 1],
        );
        // add bias
        y_hat += self.output_bias.af_bias();
        // add all branch predictions
        for branch_ix in 0..self.num_branches {
            let cfg = &self.branch_cfgs[branch_ix];
            let x = Array::new(
                &gen.x()[branch_ix],
                dim4!(gen.num_individuals() as u64, cfg.num_markers as u64),
            );
            y_hat += B::from_cfg(&cfg).predict(&x);
        }
        to_host(&y_hat)
    }

    pub fn predict_f64(&self, gen: &Genotypes) -> Vec<f64> {
        self.predict(gen).iter().map(|e| *e as f64).collect()
    }

    fn record_mse(&mut self, train_data: &Data, test_data: Option<&Data>) {
        self.training_stats.add_mse_train(self.mse(train_data));
        if let Some(tst) = test_data {
            self.training_stats.add_mse_test(self.mse(tst));
        }
    }

    fn residual(&self, x: &Vec<Vec<f32>>, y: &Array<f32>) -> Array<f32> {
        y - self.predict_device(x, y.elements())
    }

    // TODO: predict using posterior predictive distribution instead of point estimate
    fn predict_device(&self, x_test: &Vec<Vec<f32>>, num_individuals: usize) -> Array<f32> {
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
            y_hat += B::from_cfg(&cfg).predict(&x);
        }
        y_hat
    }

    pub fn rss(&self, data: &Data) -> f32 {
        let y_test_arr = Array::new(&data.y(), dim4!(data.num_individuals() as u64, 1, 1, 1));
        let y_hat = self.predict_device(&data.x(), data.num_individuals());
        let residual = y_test_arr - y_hat;
        super::gibbs_steps::sum_of_squares(&residual)
    }

    pub fn mse(&self, data: &Data) -> f32 {
        self.rss(data) / data.num_individuals() as f32
    }

    fn report_training_state(&self, iteration: usize) {
        if let Some(tst_mse) = &self.training_stats.mse_test {
            info!(
                "iteration: {:} \t | acc: {:.2} \t | early_rej: {:.2} \t | end_rej: {:.2} \t | mse(trn): {:.4} \t | mse(tst): {:.4}",
                iteration,
                self.training_stats.acceptance_rate(),
                self.training_stats.early_rejection_rate(),
                self.training_stats.end_rejection_rate(),
                self.training_stats.mse_train.last().unwrap(),
                tst_mse.last().unwrap());
        } else {
            info!(
                "iteration: {:} \t | acc: {:.2} \t | early_rej: {:.2} \t | end_rej: {:.2} \t | mse(trn): {:.4}",
                iteration,
                self.training_stats.acceptance_rate(),
                self.training_stats.early_rejection_rate(),
                self.training_stats.end_rejection_rate(),
                self.training_stats.mse_train.last().unwrap());
        }
    }

    fn note_hmc_step_result(&mut self, res: &HMCStepResult) {
        self.training_stats.add_hmc_step_result(res);
    }

    // fn reset_training_stats(&mut self) {
    //     self.training_stats = TrainingStats::new();
    // }
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
