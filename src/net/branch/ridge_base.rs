use super::{
    super::gibbs_steps::ridge_multi_param_precision_posterior,
    super::model_type::ModelType,
    super::params::{BranchParams, BranchPrecisions},
    branch_cfg::BranchCfg,
    branch_cfg_builder::BranchCfgBuilder,
    branch_sampler::BranchSampler,
    branch_struct::BranchStruct,
    step_sizes::StepSizes,
    training_state::TrainingState,
};
use crate::af_helpers::{af_scalar, scalar_to_host, sum_of_squares};
use crate::net::activation_functions::*;
use crate::net::mcmc_cfg::MCMCCfg;
use crate::net::params::NetworkPrecisionHyperparameters;
use arrayfire::{sqrt, Array};
use rand::prelude::ThreadRng;

pub struct RidgeBaseBranch {
    pub(crate) num_params: usize,
    pub(crate) num_weights: usize,
    pub(crate) params: BranchParams,
    pub(crate) num_markers: usize,
    pub(crate) precisions: BranchPrecisions,
    pub(crate) layer_widths: Vec<usize>,
    pub(crate) num_layers: usize,
    pub(crate) rng: ThreadRng,
    pub(crate) training_state: TrainingState,
    pub(crate) activation_function: ActivationFunction,
}

crate::net::activation_functions::has_activation_function!(RidgeBaseBranch);
super::branch_struct::branch_struct!(RidgeBaseBranch);

impl BranchSampler for RidgeBaseBranch {
    fn summary_stat_fn_host(vals: &[f32]) -> f32 {
        crate::arr_helpers::sum_of_squares(vals)
    }

    fn summary_stat_fn(&self, vals: &Array<f32>) -> Array<f32> {
        af_scalar(sum_of_squares(vals))
    }

    fn model_type() -> ModelType {
        ModelType::RidgeBase
    }

    fn build_cfg(cfg_bld: BranchCfgBuilder) -> BranchCfg {
        cfg_bld.build_base()
    }

    fn std_scaled_step_sizes(&self, mcmc_cfg: &MCMCCfg) -> StepSizes {
        let const_factor = mcmc_cfg.hmc_step_size_factor;
        let mut wrt_weights = Vec::with_capacity(self.num_layers());
        let mut wrt_biases = Vec::with_capacity(self.num_layers() - 1);

        for index in 0..self.num_layers() {
            wrt_weights.push(Array::new(
                &vec![
                    const_factor * (1. / scalar_to_host(self.weight_precisions(index))).sqrt();
                    self.layer_weights(index).elements()
                ],
                self.layer_weights(index).dims(),
            ));
        }
        for index in 0..self.num_layers() - 1 {
            wrt_biases.push(
                arrayfire::constant(1.0f32, self.layer_biases(index).dims())
                    * (const_factor * (1.0f32 / arrayfire::sqrt(self.bias_precision(index)))),
            );
        }

        StepSizes {
            wrt_weights,
            wrt_biases,
            wrt_weight_precisions: None,
            wrt_bias_precisions: None,
            wrt_error_precision: None,
        }
    }

    fn izmailov_step_sizes(&mut self, mcmc_cfg: &MCMCCfg) -> StepSizes {
        let integration_length = mcmc_cfg.hmc_integration_length;
        let mut wrt_weights: Vec<Array<f32>> = Vec::with_capacity(self.num_layers());
        let mut wrt_biases = Vec::with_capacity(self.num_layers() - 1);

        for index in 0..self.num_layers() {
            wrt_weights.push(
                mcmc_cfg.hmc_step_size_factor * std::f32::consts::PI
                    / (2f32
                        * sqrt(&self.precisions().weight_precisions[index])
                        * integration_length as f32),
            );
        }

        for index in 0..self.num_layers() - 1 {
            wrt_biases.push(
                arrayfire::constant(
                    mcmc_cfg.hmc_step_size_factor,
                    self.layer_biases(index).dims(),
                ) * (std::f32::consts::PI
                    / (2.0f32
                        * arrayfire::sqrt(&self.precisions().bias_precisions[index])
                        * integration_length as f32)),
            );
        }

        StepSizes {
            wrt_weights,
            wrt_biases,
            wrt_weight_precisions: None,
            wrt_bias_precisions: None,
            wrt_error_precision: None,
        }
    }

    fn log_density_joint_wrt_local_weights(
        &self,
        params: &BranchParams,
        precisions: &BranchPrecisions,
        hyperparams: &NetworkPrecisionHyperparameters,
    ) -> Array<f32> {
        let mut log_density: Array<f32> = af_scalar(0.0);

        // weight terms
        for i in 0..self.output_layer_index() {
            let (shape, scale) = hyperparams.layer_prior_hyperparams(i, self.num_layers());
            log_density -= (sum_of_squares(params.layer_weights(i)) / 2.0f32 + 1.0f32 / scale)
                * precisions.layer_weight_precisions(i);
            let nvar = params.layer_weights(i).elements();
            log_density += (shape + (nvar as f32 - 2.0f32) / 2.0)
                * arrayfire::log(&precisions.layer_weight_precisions(i));
        }

        log_density
    }

    fn log_density_joint_wrt_output_weights(
        &self,
        params: &BranchParams,
        precisions: &BranchPrecisions,
        hyperparams: &NetworkPrecisionHyperparameters,
    ) -> Array<f32> {
        let mut log_density: Array<f32> = af_scalar(0.0);

        let i = self.output_layer_index();
        let (shape, scale) = hyperparams.layer_prior_hyperparams(i, self.num_layers());
        let global_sum_of_squares =
            sum_of_squares(params.layer_weights(i)) + self.output_weight_summary_stats().reg_sum();
        log_density -= ((0.5f32 * global_sum_of_squares) + 1.0 / scale)
            * precisions.layer_weight_precisions(i);

        log_density += (shape + (self.output_weight_summary_stats().num_params() - 2.0f32) / 2.0)
            * &arrayfire::log(precisions.layer_weight_precisions(i));

        log_density
    }

    fn log_density_wrt_weights(
        &self,
        params: &BranchParams,
        precisions: &BranchPrecisions,
    ) -> Array<f32> {
        let mut log_density: Array<f32> = af_scalar(0.0);

        // weight terms
        for i in 0..self.num_layers() {
            log_density -= sum_of_squares(params.layer_weights(i)) / 2.0
                * precisions.layer_weight_precisions(i);
        }

        log_density
    }

    fn log_density_gradient_wrt_weights(&self) -> Vec<Array<f32>> {
        let mut ldg_wrt_weights: Vec<Array<f32>> = Vec::with_capacity(self.num_layers);
        for layer_index in 0..self.num_layers() {
            ldg_wrt_weights.push(
                -(self.error_precision() * self.layer_d_rss_wrt_weights(layer_index)
                    + self.weight_precisions(layer_index) * self.layer_weights(layer_index)),
            );
        }
        ldg_wrt_weights
    }

    fn log_density_gradient_wrt_weight_precisions(
        &self,
        hyperparams: &NetworkPrecisionHyperparameters,
    ) -> Vec<Array<f32>> {
        let mut ldg_wrt_weight_precisions: Vec<Array<f32>> = Vec::with_capacity(self.num_layers);
        for layer_index in 0..self.output_layer_index() {
            let precisions: &Array<f32> = self.weight_precisions(layer_index);
            let params: &Array<f32> = self.layer_weights(layer_index);
            let (shape, scale) = hyperparams.layer_prior_hyperparams(layer_index, self.num_layers);
            ldg_wrt_weight_precisions.push(
                (2.0f32 * shape + params.elements() as f32 - 2.0) / (2.0f32 * precisions)
                    - (1.0f32 / scale)
                    - sum_of_squares(params) / 2.0,
            );
        }

        let layer_index = self.output_layer_index();
        let precisions: &Array<f32> = self.weight_precisions(layer_index);
        let params: &Array<f32> = self.layer_weights(layer_index);
        let (shape, scale) = hyperparams.layer_prior_hyperparams(layer_index, self.num_layers);
        ldg_wrt_weight_precisions.push(
            (2.0f32 * shape + self.output_weight_summary_stats().num_params() - 2.0f32)
                / (2.0f32 * precisions)
                - (1.0f32 / scale)
                - (sum_of_squares(params) + self.output_weight_summary_stats().reg_sum()) / 2.0f32,
        );

        ldg_wrt_weight_precisions
    }

    fn precision_posterior_host(
        &mut self,
        // k
        prior_shape: f32,
        // s or theta
        prior_scale: f32,
        summary_stat: f32,
        num_vals: usize,
    ) -> f32 {
        super::super::gibbs_steps::ridge_multi_param_precision_posterior_host_prepared_summary_stats(
            prior_shape,
            prior_scale,
            summary_stat,
            num_vals,
            self.rng(),
        )
    }

    /// Samples precision values from their posterior distribution in a Gibbs step.
    fn sample_prior_precisions(&mut self, hyperparams: &NetworkPrecisionHyperparameters) {
        for i in 0..self.output_layer_index() {
            let (prior_shape, prior_scale) =
                hyperparams.layer_prior_hyperparams(i, self.num_layers());
            self.precisions.weight_precisions[i] =
                af_scalar(ridge_multi_param_precision_posterior(
                    prior_shape,
                    prior_scale,
                    &self.params.weights[i],
                    &mut self.rng,
                ));
            self.precisions.bias_precisions[i] = af_scalar(ridge_multi_param_precision_posterior(
                prior_shape,
                prior_scale,
                &self.params.biases[i],
                &mut self.rng,
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::net::mcmc_cfg::MCMCCfg;
    use crate::net::params::{
        NetworkPrecisionHyperparameters, OutputWeightSummaryStats, PrecisionHyperparameters,
    };
    use arrayfire::{dim4, Array};
    // use arrayfire::{af_print, randu};

    use super::super::{
        branch_builder::BranchBuilder, branch_sampler::BranchSampler, branch_struct::BranchStruct,
    };
    use super::RidgeBaseBranch;

    use crate::af_helpers::{scalar_to_host, to_host};
    use crate::net::branch::momentum::BranchMomentum;
    use crate::net::params::BranchParams;

    // #[test]
    // fn af() {
    //     let num_rows: u64 = 5;
    //     let num_cols: u64 = 3;
    //     let dims = Dim4::new(&[num_rows, num_cols, 1, 1]);
    //     let a = randu::<f32>(dims);
    //     af_print!("Create a 5-by-3 matrix of random floats on the GPU", a);
    // }

    fn make_test_branch() -> RidgeBaseBranch {
        let exp_weights = [
            Array::new(&[0., 1., 2., 3., 4., 5.], dim4![3, 2, 1, 1]),
            Array::new(&[1., 2.], dim4![2, 1, 1, 1]),
            Array::new(&[2.], dim4![1, 1, 1, 1]),
        ];
        let exp_biases = [
            Array::new(&[0., 1.], dim4![1, 2, 1, 1]),
            Array::new(&[2.], dim4![1, 1, 1, 1]),
        ];

        BranchBuilder::new()
            .with_num_markers(3)
            .add_hidden_layer(2)
            .add_layer_biases(&exp_biases[0])
            .add_layer_weights(&exp_weights[0])
            .add_summary_weights(&exp_weights[1])
            .add_summary_bias(&exp_biases[1])
            .add_output_weight(&exp_weights[2])
            .build_ridge_base()
    }

    fn make_test_uniform_params(c: f32) -> BranchParams {
        let weights = [
            Array::new(&[c; 6], dim4![3, 2, 1, 1]),
            Array::new(&[c; 2], dim4![2, 1, 1, 1]),
            Array::new(&[c], dim4![1, 1, 1, 1]),
        ]
        .to_vec();
        let biases = [
            Array::new(&[c; 2], dim4![1, 2, 1, 1]),
            Array::new(&[c], dim4![1, 1, 1, 1]),
        ]
        .to_vec();
        let layer_widths = vec![2, 1, 1];
        let num_markers = 3;
        BranchParams {
            weights,
            biases,
            layer_widths,
            num_markers,
            // this simulates the situation in which the branches reg sum has been subtracted already, withing a sampling fn
            output_weight_summary_stats: OutputWeightSummaryStats::new_single_branch(0.0, 1),
        }
    }

    fn make_test_uniform_momenta(c: f32) -> BranchMomentum {
        let wrt_weights = [
            Array::new(&[c; 6], dim4![3, 2, 1, 1]),
            Array::new(&[c; 2], dim4![2, 1, 1, 1]),
            Array::new(&[c], dim4![1, 1, 1, 1]),
        ]
        .to_vec();
        let wrt_biases = [
            Array::new(&[c; 2], dim4![1, 2, 1, 1]),
            Array::new(&[c], dim4![1, 1, 1, 1]),
        ]
        .to_vec();
        BranchMomentum {
            wrt_weights,
            wrt_biases,
        }
    }

    fn make_test_branch_with_precision(precision: f32) -> RidgeBaseBranch {
        let exp_weights = [
            Array::new(&[0., 1., 2., 3., 4., 5.], dim4![3, 2, 1, 1]),
            Array::new(&[1., 2.], dim4![2, 1, 1, 1]),
            Array::new(&[2.], dim4![1, 1, 1, 1]),
        ];
        let exp_biases = [
            Array::new(&[0., 1.], dim4![1, 2, 1, 1]),
            Array::new(&[2.], dim4![1, 1, 1, 1]),
        ];

        BranchBuilder::new()
            .with_num_markers(3)
            .add_hidden_layer(2)
            .add_layer_biases(&exp_biases[0])
            .add_layer_weights(&exp_weights[0])
            .add_summary_weights(&exp_weights[1])
            .add_summary_bias(&exp_biases[1])
            .add_output_weight(&exp_weights[2])
            .with_initial_precision_value(precision)
            .build_ridge_base()
    }

    #[test]
    fn forward_feed() {
        let num_individuals = 4;
        let num_markers = 3;
        let branch = make_test_branch();
        let x_train: Array<f32> = Array::new(
            &[1., 0., 0., 2., 1., 1., 2., 0., 0., 2., 0., 1.],
            dim4![num_individuals, num_markers, 1, 1],
        );
        let (_pre_activations, activations) = branch.forward_feed(&x_train);

        // correct number of activations
        assert_eq!(activations.len(), branch.num_layers);

        // correct dimensions of activations
        for i in 0..(branch.num_layers) {
            assert_eq!(
                activations[i].dims(),
                dim4![num_individuals, branch.layer_widths[i] as u64, 1, 1]
            );
        }

        let exp_activations: Vec<Array<f32>> = vec![
            Array::new(
                &[
                    0.7615942,
                    0.9999092,
                    0.9640276,
                    0.9640276,
                    0.99999976,
                    0.9999999999998128,
                    0.99999994,
                    0.9999999999244973,
                ],
                dim4![4, 2, 1, 1],
            ),
            Array::new(
                &[0.99985373, 0.99990916, 0.9999024, 0.9999024],
                dim4![4, 1, 1, 1],
            ),
            Array::new(
                &[1.9997075, 1.9998183, 1.9998049, 1.9998049],
                dim4![4, 1, 1, 1],
            ),
        ];
        // correct values of activations
        for i in 0..(branch.num_layers) {
            assert_eq!(to_host(&activations[i]), to_host(&exp_activations[i]));
        }
    }
    #[test]
    fn log_density_joint() {
        let num_individuals = 4;
        let num_markers = 3;
        let branch = make_test_branch_with_precision(2.0);
        let x_train: Array<f32> = Array::new(
            &[1., 0., 0., 2., 1., 1., 2., 0., 0., 2., 0., 1.],
            dim4![num_individuals, num_markers, 1, 1],
        );
        let y_train: Array<f32> = Array::new(&[0.0, 2.0, 1.0, 1.5], dim4![4, 1, 1, 1]);
        let hyperparams = NetworkPrecisionHyperparameters {
            dense: PrecisionHyperparameters::new(3.0, 2.0),
            summary: PrecisionHyperparameters::new(3.0, 2.0),
            output: PrecisionHyperparameters::new(4.0, 5.0),
        };

        let rss = branch.rss(&x_train, &y_train);
        assert_eq!(rss, 5.248245);

        let ld_wrt_e = branch.log_density_joint_wrt_rss(
            branch.precisions(),
            rss,
            &hyperparams,
            num_individuals as usize,
        );

        assert_eq!(scalar_to_host(&ld_wrt_e), -2.182509);

        let ld_wrt_w = branch.log_density_joint_wrt_weights(
            branch.params(),
            branch.precisions(),
            &hyperparams,
        );

        assert_eq!(scalar_to_host(&ld_wrt_w), -58.428806);

        let ld_wrt_b =
            branch.log_density_joint_wrt_biases(branch.params(), branch.precisions(), &hyperparams);

        assert_eq!(scalar_to_host(&ld_wrt_b), -3.1876905);

        let ld = branch.log_density_joint(
            branch.params(),
            branch.precisions(),
            rss,
            &hyperparams,
            num_individuals as usize,
        );

        assert_eq!(ld, -63.799007);
    }

    #[test]
    fn log_density_gradient_joint() {
        let num_individuals = 4;
        let num_markers = 3;
        let mut branch = make_test_branch_with_precision(2.0);
        let x_train: Array<f32> = Array::new(
            &[1., 0., 0., 2., 1., 1., 2., 0., 0., 2., 0., 1.],
            dim4![num_individuals, num_markers, 1, 1],
        );
        let y_train: Array<f32> = Array::new(&[0.0, 2.0, 1.0, 1.5], dim4![4, 1, 1, 1]);
        let hyperparams = NetworkPrecisionHyperparameters {
            dense: PrecisionHyperparameters::new(3.0, 2.0),
            summary: PrecisionHyperparameters::new(3.0, 2.0),
            output: PrecisionHyperparameters::new(4.0, 5.0),
        };
        let ldg = branch.log_density_gradient_joint(&x_train, &y_train, &hyperparams);

        let exp_ldg_wrt_w = [
            Array::new(
                &[
                    -0.0010378566,
                    -2.00109287e+00,
                    -4.00002756e+00,
                    -6.00000000e+00,
                    -8.00000000e+00,
                    -1.00000000e+01,
                ],
                dim4![3, 2, 1, 1],
            ),
            Array::new(&[-2.0029104, -4.0035105], dim4!(2, 1, 1, 1)),
            Array::new(&[-10.997393], dim4!(1, 1, 1, 1)),
        ];

        for i in 0..(branch.num_layers) {
            assert_eq!(to_host(&ldg.wrt_weights[i]), to_host(&exp_ldg_wrt_w[i]));
        }

        let exp_ldg_wrt_b = [
            Array::new(&[-0.0010654309, -2.00000000e+00], dim4!(2, 1, 1, 1)),
            Array::new(&[-4.0035105], dim4!(1, 1, 1, 1)),
        ];

        for i in 0..(branch.num_layers - 1) {
            assert_eq!(to_host(&ldg.wrt_biases[i]), to_host(&exp_ldg_wrt_b[i]));
        }

        // wrt error precision
        assert_eq!(scalar_to_host(&ldg.wrt_error_precision), -0.32412243);

        let exp_ldg_wrt_w_prec = [
            Array::new(&[-25.5], dim4!(1)),
            Array::new(&[-1.5], dim4!(1)),
            Array::new(&[-0.45000005], dim4!(1)),
        ];

        for i in 0..(branch.num_layers) {
            assert_eq!(
                to_host(&ldg.wrt_weight_precisions[i]),
                to_host(&exp_ldg_wrt_w_prec[i])
            );
        }

        let exp_ldg_wrt_b_prec = [0.5, -1.25];

        for i in 0..(branch.num_layers - 1) {
            assert_eq!(
                scalar_to_host(&ldg.wrt_bias_precisions[i]),
                exp_ldg_wrt_b_prec[i]
            );
        }
    }

    #[test]
    fn log_density_gradient() {
        let num_individuals = 4;
        let num_markers = 3;
        let mut branch = make_test_branch();
        let x_train: Array<f32> = Array::new(
            &[1., 0., 0., 2., 1., 1., 2., 0., 0., 2., 0., 1.],
            dim4![num_individuals, num_markers, 1, 1],
        );
        let y_train: Array<f32> = Array::new(&[0.0, 2.0, 1.0, 1.5], dim4![4, 1, 1, 1]);
        let ldg = branch.log_density_gradient(&x_train, &y_train);

        // correct output length
        assert_eq!(ldg.wrt_weights.len(), branch.num_layers);
        assert_eq!(ldg.wrt_biases.len(), branch.num_layers - 1);

        // correct dimensions
        for i in 0..(branch.num_layers) {
            assert_eq!(ldg.wrt_weights[i].dims(), branch.layer_weights(i).dims());
        }
        for i in 0..(branch.num_layers - 1) {
            assert_eq!(ldg.wrt_biases[i].dims(), branch.layer_biases(i).dims());
        }

        let exp_ldg_wrt_w = [
            Array::new(
                &[-0.0005189283, -1.0005465, -2.0000138, -3.0, -4.0, -5.0],
                dim4![3, 2, 1, 1],
            ),
            Array::new(&[-1.0014552, -2.0017552], dim4![2, 1, 1, 1]),
            Array::new(&[-5.4986963], dim4![1, 1, 1, 1]),
        ];

        let exp_ldg_wrt_b = [
            Array::new(&[-0.00053271546, -1.2088213e-9], dim4![2, 1, 1, 1]),
            Array::new(&[-0.0017552058], dim4![1, 1, 1, 1]),
        ];

        // correct values
        for i in 0..(branch.num_layers) {
            assert_eq!(to_host(&ldg.wrt_weights[i]), to_host(&exp_ldg_wrt_w[i]));
        }
        for i in 0..(branch.num_layers - 1) {
            assert_eq!(to_host(&ldg.wrt_biases[i]), to_host(&exp_ldg_wrt_b[i]));
        }
    }

    #[test]
    fn uniform_step_sizes() {
        let branch = make_test_branch();
        let mut cfg = MCMCCfg::default();
        cfg.hmc_step_size_factor = 1.0;
        let step_sizes = branch.uniform_step_sizes(&cfg);
        for i in 0..(branch.num_layers - 1) {
            let mut obs = to_host(&step_sizes.wrt_weights[i]);
            assert_eq!(obs, vec![cfg.hmc_step_size_factor; obs.len()]);
            obs = to_host(&step_sizes.wrt_biases[i]);
            assert_eq!(obs, vec![cfg.hmc_step_size_factor; obs.len()]);
        }
        let obs = to_host(&step_sizes.wrt_weights[branch.num_layers - 1]);
        assert_eq!(obs, vec![cfg.hmc_step_size_factor; obs.len()]);
    }

    #[test]
    fn net_movement() {
        let branch = make_test_branch();
        let momenta = make_test_uniform_momenta(1.);
        let init_params = make_test_uniform_params(0.);
        assert!(branch.net_movement(&init_params, &momenta) > 0.0);
        let init_params = make_test_uniform_params(100.);
        assert!(branch.net_movement(&init_params, &momenta) < 0.0);
    }

    #[test]
    fn set_params() {
        let mut branch = make_test_branch_with_precision(2.0);

        // test branch has these params:
        // let exp_weights = [
        //     Array::new(&[0., 1., 2., 3., 4., 5.], dim4![3, 2, 1, 1]),
        //     Array::new(&[1., 2.], dim4![2, 1, 1, 1]),
        //     Array::new(&[2.], dim4![1, 1, 1, 1]),
        // ];
        // let exp_biases = [
        //     Array::new(&[0., 1.], dim4![1, 2, 1, 1]),
        //     Array::new(&[2.], dim4![1, 1, 1, 1]),
        // ];

        // weight setting
        branch.incr_weight(0, 1, 1, 1.0);
        assert_eq!(
            to_host(branch.layer_weights(0)),
            vec![0., 1., 2., 3., 5., 5.]
        );
        branch.decr_weight(0, 1, 1, 1.0);
        assert_eq!(
            to_host(branch.layer_weights(0)),
            vec![0., 1., 2., 3., 4., 5.]
        );

        // bias setting
        branch.incr_bias(1, 0, 2.);
        assert_eq!(to_host(branch.layer_biases(1)), vec![4.0]);
        branch.decr_bias(1, 0, 2.);
        assert_eq!(to_host(branch.layer_biases(1)), vec![2.0]);

        branch.incr_bias(0, 1, 2.);
        assert_eq!(to_host(branch.layer_biases(0)), vec![0., 3.0]);
        branch.decr_bias(0, 1, 2.);
        assert_eq!(to_host(branch.layer_biases(0)), vec![0., 1.0]);

        // weight precision setting
        branch.incr_weight_precision(0, 0, 1.0);
        assert_eq!(to_host(branch.layer_weight_precisions(0)), vec![3.0]);
        branch.decr_weight_precision(0, 0, 1.0);
        assert_eq!(to_host(branch.layer_weight_precisions(0)), vec![2.0]);

        // bias precision setting
        branch.incr_bias_precision(0, 1.0);
        assert_eq!(to_host(branch.layer_bias_precision(0)), vec![3.0]);
        branch.decr_bias_precision(0, 1.0);
        assert_eq!(to_host(branch.layer_bias_precision(0)), vec![2.0]);

        // error precision setting
        branch.incr_error_precision(1.0);
        assert_eq!(to_host(branch.error_precision()), vec![3.0]);
        branch.decr_error_precision(1.0);
        assert_eq!(to_host(branch.error_precision()), vec![2.0]);
    }
}
