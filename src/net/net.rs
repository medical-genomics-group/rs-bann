use super::branch::gradient::BranchLogDensityGradient;
use super::log_posterior_density::LogPosteriorDensity;
use super::model_type::ModelType;
use super::params::GlobalParams;
use super::{
    branch::branch::{Branch, BranchCfg, HMCStepResult},
    gibbs_steps::ridge_single_param_precision_posterior,
    mcmc_cfg::MCMCCfg,
    params::{ModelHyperparameters, NetworkPrecisionHyperparameters},
    train_stats::{ReportCfg, TrainingStats},
};
use crate::af_helpers::to_host;
use crate::data::data::Data;
use crate::data::genotypes::GroupedGenotypes;
use arrayfire::{dim4, sum_all, Array};
use bincode::{deserialize_from, serialize_into};
use log::{debug, info};
use rand::{prelude::SliceRandom, rngs::ThreadRng, thread_rng};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use serde_json::to_writer;
use std::path::Path;
use std::{
    fs::{create_dir_all, File},
    io::{BufReader, BufWriter, Write},
    marker::PhantomData,
};

#[derive(Serialize, Deserialize)]
pub struct OutputBias {
    // the part of the total residual error precision contributed by the output bias (lambda_e)
    pub(crate) error_precision: f32,
    // a sample from the posterior of the precision of the prior of the output bias (lambda_b)
    pub(crate) precision: f32,
    pub(crate) bias: f32,
}

impl OutputBias {
    fn update_global_params(&mut self, gp: &GlobalParams) {
        self.error_precision = gp.error_precision();
    }

    fn set_to_maximum_likelihood(&mut self, residual: &Array<f32>) {
        self.bias = arrayfire::sum_all(residual).0 / residual.elements() as f32;
    }

    fn sample_bias(&mut self, residual: &Array<f32>, n: usize, rng: &mut ThreadRng) {
        let (sum_r, _) = sum_all(residual);
        let nu = self.error_precision / (n as f32 * self.error_precision + self.precision);
        let mean = nu * sum_r;
        let std = (1. / (n as f32 * self.error_precision + self.precision)).sqrt();
        self.bias = Normal::new(mean, std).unwrap().sample(rng);
    }

    /// Sample lambda_theta of the output bias
    fn sample_prior_precision(
        &mut self,
        hyperparams: &NetworkPrecisionHyperparameters,
        rng: &mut ThreadRng,
    ) {
        self.precision = ridge_single_param_precision_posterior(
            hyperparams.output_layer_prior_shape(),
            hyperparams.output_layer_prior_shape(),
            self.bias,
            rng,
        );
    }

    fn af_bias(&self) -> Array<f32> {
        Array::new(&[self.bias], dim4!(1, 1, 1, 1))
    }
}

#[derive(Serialize, Deserialize)]
/// The full network model
pub struct Net<B: Branch> {
    hyperparams: NetworkPrecisionHyperparameters,
    num_branches: usize,
    branch_cfgs: Vec<BranchCfg>,
    output_bias: OutputBias,
    training_stats: TrainingStats,
    log_posterior_density: LogPosteriorDensity,
    global_params: GlobalParams,
    branch_type: PhantomData<B>,
}

impl<B: Branch> Net<B> {
    pub fn new(
        hyperparams: NetworkPrecisionHyperparameters,
        num_branches: usize,
        branch_cfgs: Vec<BranchCfg>,
        output_bias: OutputBias,
        global_params: GlobalParams,
    ) -> Self {
        Self {
            hyperparams,
            num_branches,
            branch_cfgs,
            output_bias,
            training_stats: TrainingStats::new(),
            log_posterior_density: LogPosteriorDensity::new(num_branches),
            global_params,
            branch_type: PhantomData,
        }
    }

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

    pub fn set_error_precision(&mut self, val: f32) {
        self.global_params.set_error_precision(val);
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

    pub fn write_hyperparams(&self, mcmc_cfg: &MCMCCfg) {
        let w = BufWriter::new(File::create(mcmc_cfg.hyperparam_path()).unwrap());
        to_writer(
            w,
            &ModelHyperparameters::new(&self.hyperparams, &self.branch_cfgs),
        )
        .unwrap();
    }

    fn initialize_stats<T: GroupedGenotypes>(&mut self, x: &T, y: &Array<f32>) -> Array<f32> {
        // add bias
        let mut residual = y - self.output_bias.af_bias();
        // add all branch predictions
        for branch_ix in 0..self.num_branches {
            let cfg = &mut self.branch_cfgs[branch_ix];
            cfg.update_global_params(&self.global_params);
            let branch = B::from_cfg(cfg);
            residual -= branch.predict(&x.x_group_af(branch_ix));
            self.update_lpd_from_branch(branch_ix, &branch, &residual);
        }

        residual
    }

    fn update_lpd_from_branch(
        &mut self,
        branch_ix: usize,
        branch: &impl Branch,
        residual: &Array<f32>,
    ) {
        self.log_posterior_density.update_from_branch(
            branch_ix,
            branch,
            residual,
            &self.hyperparams,
        );
    }

    pub fn perturb(&mut self, params_by: Option<f32>, precisions_by: Option<f32>) {
        if params_by.is_some() || precisions_by.is_some() {
            for branch_ix in 0..self.num_branches() {
                let cfg = &mut self.branch_cfgs[branch_ix];
                if let Some(by) = params_by {
                    cfg.perturb_params(by);
                }
                if let Some(by) = precisions_by {
                    cfg.perturb_precisions(by);
                }
            }
        }
    }

    pub fn train<T: GroupedGenotypes>(
        &mut self,
        train_data: &Data<T>,
        mcmc_cfg: &MCMCCfg,
        verbose: bool,
        report_cfg: Option<ReportCfg<T>>,
    ) {
        if mcmc_cfg.chain_length > mcmc_cfg.burn_in {
            self.create_model_dir(mcmc_cfg);
            self.create_effect_size_dir(mcmc_cfg);
        }

        let mut trace_file = None;
        if mcmc_cfg.trace {
            trace_file = Some(BufWriter::new(File::create(mcmc_cfg.trace_path()).unwrap()));
        }

        let mut rng = thread_rng();
        let num_individuals = train_data.num_individuals();
        let y_train = &Array::new(train_data.y(), dim4!(num_individuals as u64, 1, 1, 1));
        let mut residual = self.initialize_stats(&train_data.gen, y_train);
        let mut branch_ixs = (0..self.num_branches).collect::<Vec<usize>>();
        let mut report_interval = 1;

        info!(
            "Training net with {:} branches, {:} params",
            self.num_branches,
            self.num_params()
        );

        // initial loss
        self.record_perf(&residual, report_cfg.as_ref().unwrap().test_data);

        // report
        if verbose {
            self.report_training_state(0);
            report_interval = report_cfg.as_ref().unwrap().interval;
        }

        if mcmc_cfg.trace {
            to_writer(trace_file.as_mut().unwrap(), &self.branch_cfgs).unwrap();
            trace_file.as_mut().unwrap().write_all(b"\n").unwrap();
        }

        // save initial model if no burn in
        if mcmc_cfg.burn_in == 0 {
            self.save_model(0, mcmc_cfg);
        }

        for chain_ix in 1..=mcmc_cfg.chain_length {
            // shuffle order in which branches are trained
            branch_ixs.shuffle(&mut rng);
            for &branch_ix in &branch_ixs {
                debug!("Updating branch {:}", branch_ix);

                let cfg = &mut self.branch_cfgs[branch_ix];
                cfg.update_global_params(&self.global_params);

                // load marker data onto device
                let x = &train_data.x_branch_af(branch_ix);

                // load branch cfg
                let mut branch = B::from_cfg(cfg);
                if !(mcmc_cfg.gradient_descent_joint || mcmc_cfg.joint_hmc) {
                    branch.sample_error_precision(&residual, &self.hyperparams);
                    if !mcmc_cfg.fixed_param_precisions {
                        branch.sample_param_precisions(&self.hyperparams);
                    }
                }

                let prev_pred = branch.predict(x);
                residual = &residual + &prev_pred;

                let step_res = if mcmc_cfg.gradient_descent {
                    branch.gradient_descent(x, &residual, mcmc_cfg)
                } else if mcmc_cfg.gradient_descent_joint {
                    branch.gradient_descent_joint(x, &residual, mcmc_cfg, &self.hyperparams)
                } else if mcmc_cfg.joint_hmc {
                    branch.hmc_step_joint(x, &residual, mcmc_cfg, &self.hyperparams)
                } else {
                    branch.hmc_step(x, &residual, mcmc_cfg)
                };
                self.note_hmc_step_result(&step_res);
                match step_res {
                    // update residual
                    HMCStepResult::Accepted(state) => {
                        residual -= state.y_pred;
                        self.update_lpd_from_branch(branch_ix, &branch, &residual);
                    }
                    // not accepted, just remove previous prediction
                    _ => residual -= prev_pred,
                }

                // update global params & dump branch cfg
                let cfg = branch.to_cfg();
                self.global_params.update_from_branch_cfg(&cfg);
                self.branch_cfgs[branch_ix] = cfg;

                // compute effect sizes and save
                if chain_ix >= mcmc_cfg.burn_in {
                    self.save_effect_sizes(
                        &branch.effect_sizes(x, y_train),
                        chain_ix,
                        branch_ix,
                        mcmc_cfg,
                    )
                }

                self.report_training_state_debug(chain_ix, &residual);

                // sample output bias
                self.output_bias.update_global_params(&self.global_params);
                residual += self.output_bias.af_bias();

                if mcmc_cfg.sampled_output_bias {
                    self.output_bias
                        .sample_prior_precision(&self.hyperparams, &mut rng);
                    self.output_bias
                        .sample_bias(&residual, num_individuals, &mut rng);
                } else {
                    self.output_bias.set_to_maximum_likelihood(&residual);
                }

                residual -= self.output_bias.af_bias();
            }

            self.record_perf(&residual, report_cfg.as_ref().unwrap().test_data);

            // save current model if done with burn in
            if chain_ix >= mcmc_cfg.burn_in {
                let model_ix = chain_ix;
                self.save_model(model_ix, mcmc_cfg);
            }

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

    pub fn activations<T: GroupedGenotypes>(&self, gen: &T) -> Activations {
        let mut activations = Activations::new();
        for branch_ix in 0..self.num_branches() {
            let cfg = &self.branch_cfgs[branch_ix];
            let (_pre_activations, branch_activations) =
                B::from_cfg(cfg).forward_feed(&gen.x_group_af(branch_ix));
            activations.add_branch_activations(branch_activations);
        }
        activations
    }

    pub fn gradient<T: GroupedGenotypes>(&self, data: &Data<T>) -> Vec<BranchLogDensityGradient> {
        (0..self.num_branches())
            .map(|ix| {
                B::from_cfg(&self.branch_cfgs[ix])
                    .log_density_gradient(&data.x_branch_af(ix), &data.y_af())
            })
            .collect()
    }

    pub fn predict<T: GroupedGenotypes>(&self, gen: &T) -> Vec<f32> {
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
            y_hat += B::from_cfg(cfg).predict(&gen.x_group_af(branch_ix));
        }
        to_host(&y_hat)
    }

    pub fn predict_f64<T: GroupedGenotypes>(&self, gen: &T) -> Vec<f64> {
        self.predict(gen).iter().map(|e| *e as f64).collect()
    }

    fn save_model(&self, model_ix: usize, mcmc_cfg: &MCMCCfg) {
        let mut model_path = mcmc_cfg.models_path().join(model_ix.to_string());
        model_path.set_extension("bin");
        self.to_file(&model_path);
    }

    fn save_effect_sizes(
        &self,
        effect_sizes: &Array<f32>,
        model_ix: usize,
        branch_ix: usize,
        mcmc_cfg: &MCMCCfg,
    ) {
        let num_markers = effect_sizes.dims().get()[1] as usize;
        let file_path = mcmc_cfg
            .effect_sizes_path()
            .join(format!("{}_{}", model_ix, branch_ix));
        let mut wtr = csv::Writer::from_path(file_path).unwrap();
        to_host(&arrayfire::transpose(effect_sizes, false))
            .chunks(num_markers)
            .for_each(|row| wtr.write_record(row.iter().map(|e| e.to_string())).unwrap());
        wtr.flush().expect("Failed to flush csv writer");
    }

    fn create_model_dir(&self, mcmc_cfg: &MCMCCfg) {
        create_dir_all(mcmc_cfg.models_path()).expect("Failed to create models outdir");
    }

    fn create_effect_size_dir(&self, mcmc_cfg: &MCMCCfg) {
        create_dir_all(mcmc_cfg.effect_sizes_path()).expect("Failed to create effect size outdir");
    }

    fn record_perf<T: GroupedGenotypes>(
        &mut self,
        residual: &Array<f32>,
        test_data: Option<&Data<T>>,
    ) {
        self.training_stats
            .add_lpd(self.log_posterior_density.lpd());
        self.training_stats.add_mse_train(
            crate::af_helpers::sum_of_squares(residual) / residual.elements() as f32,
        );
        if let Some(tst) = test_data {
            self.training_stats.add_mse_test(self.mse(tst));
        }
    }

    // fn residual<T: GroupedGenotypes>(&self, x: &T, y: &Array<f32>) -> Array<f32> {
    //     y - self.predict_device(x, y.elements())
    // }

    // TODO: predict using posterior predictive distribution instead of point estimate
    fn predict_device<T: GroupedGenotypes>(
        &self,
        x_test: &T,
        num_individuals: usize,
    ) -> Array<f32> {
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
            y_hat += B::from_cfg(cfg).predict(&x_test.x_group_af(branch_ix));
        }
        y_hat
    }

    pub fn rss<T: GroupedGenotypes>(&self, data: &Data<T>) -> f32 {
        let y_test_arr = Array::new(data.y(), dim4!(data.num_individuals() as u64, 1, 1, 1));
        let y_hat = self.predict_device(&data.gen, data.num_individuals());
        let residual = y_test_arr - y_hat;
        crate::af_helpers::sum_of_squares(&residual)
    }

    pub fn mse<T: GroupedGenotypes>(&self, data: &Data<T>) -> f32 {
        self.rss(data) / data.num_individuals() as f32
    }

    pub fn branch_r2s<T: GroupedGenotypes>(&self, data: &Data<T>) -> Vec<f32> {
        let mut res = vec![0.0; self.num_branches];
        let y = data.y_af();
        for branch_ix in 0..self.num_branches {
            let cfg = &self.branch_cfgs[branch_ix];
            res[branch_ix] = B::from_cfg(cfg).r2(&data.x_branch_af(branch_ix), &y);
        }
        res
    }

    fn report_training_state_debug(&self, iteration: usize, residual: &Array<f32>) {
        debug!(
                "i: {:} \t | acc: {:.2} \t | early_rej: {:.2} \t | end_rej: {:.2} \t | mse(trn): {:.4} | lpd: {:.4}",
                iteration,
                self.training_stats.acceptance_rate(),
                self.training_stats.early_rejection_rate(),
                self.training_stats.end_rejection_rate(),
                crate::af_helpers::sum_of_squares(residual) / residual.elements() as f32,
                self.log_posterior_density.lpd()
            );
    }

    fn report_training_state(&self, iteration: usize) {
        if let Some(tst_mse) = &self.training_stats.mse_test {
            info!(
                "i: {:} \t | acc: {:.2} \t | early_rej: {:.2} \t | end_rej: {:.2} \t | mse(trn): {:.4} \t | mse(tst): {:.4} | lpd: {:.4}",
                iteration,
                self.training_stats.acceptance_rate(),
                self.training_stats.early_rejection_rate(),
                self.training_stats.end_rejection_rate(),
                self.training_stats.mse_train.last().unwrap(),
                tst_mse.last().unwrap(),
                self.log_posterior_density.lpd()
            );
        } else {
            info!(
                "i: {:} \t | acc: {:.2} \t | early_rej: {:.2} \t | end_rej: {:.2} \t | mse(trn): {:.4} | lpd: {:.4}",
                iteration,
                self.training_stats.acceptance_rate(),
                self.training_stats.early_rejection_rate(),
                self.training_stats.end_rejection_rate(),
                self.training_stats.mse_train.last().unwrap(),
                self.log_posterior_density.lpd()
            );
        }
    }

    fn note_hmc_step_result(&mut self, res: &HMCStepResult) {
        self.training_stats.add_hmc_step_result(res);
    }

    // fn reset_training_stats(&mut self) {
    //     self.training_stats = TrainingStats::new();
    // }
}

/// Activations of all network nodes.
///
/// Not used for training, only for IO purposes.
#[derive(Serialize)]
pub struct Activations {
    activations: Vec<Vec<Array<f32>>>,
}

impl Activations {
    pub(crate) fn new() -> Self {
        Self {
            activations: Vec::new(),
        }
    }

    pub(crate) fn add_branch_activations(&mut self, activations: Vec<Array<f32>>) {
        self.activations.push(activations);
    }

    pub fn to_json(&self, path: &Path) {
        to_writer(File::create(path).unwrap(), self).expect("Failed to write activations to json");
    }
}

#[cfg(test)]
mod tests {
    use crate::af_helpers::to_host;
    use arrayfire::{dim4, Array, MatProp};

    #[test]
    fn arrayfire_dot_expected_result() {
        // this should compute the sum of squares of two arrays, right?
        let a = Array::new(&[1.0, 2.0, 3.0], dim4!(3, 1, 1, 1));
        let b = Array::new(&[3.0, 2.0, 1.0], dim4!(3, 1, 1, 1));
        let exp = 10.0;
        let dotp = arrayfire::dot(&a, &b, MatProp::NONE, MatProp::NONE);
        assert_eq!(to_host(&dotp)[0], exp);
    }
}
