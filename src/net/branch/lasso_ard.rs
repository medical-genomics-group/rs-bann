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
use crate::net::mcmc_cfg::MCMCCfg;
use crate::net::params::NetworkPrecisionHyperparameters;
use crate::{
    af_helpers::{af_scalar, sign, to_host},
    arr_helpers,
};
use crate::{
    af_helpers::{l1_norm, l1_norm_rows},
    net::activation_functions::*,
};
use arrayfire::{dim4, sqrt, tile, Array, MatProp};
use rand::prelude::ThreadRng;
use rand_distr::{Distribution, Gamma};

pub struct LassoArdBranch {
    pub(crate) num_params: usize,
    pub(crate) num_weights: usize,
    pub(crate) num_markers: usize,
    pub(crate) params: BranchParams,
    pub(crate) precisions: BranchPrecisions,
    pub(crate) layer_widths: Vec<usize>,
    pub(crate) num_layers: usize,
    pub(crate) rng: ThreadRng,
    pub(crate) training_state: TrainingState,
    pub(crate) activation_function: ActivationFunction,
}

super::super::activation_functions::has_activation_function!(LassoArdBranch);
super::branch_struct::branch_struct!(LassoArdBranch);

// Weights in this branch are grouped by the node they
// are going out of.
impl BranchSampler for LassoArdBranch {
    fn summary_stat_fn_host(vals: &[f32]) -> f32 {
        arr_helpers::sum_of_abs(vals)
    }

    fn summary_stat_fn(&self, vals: &Array<f32>) -> Array<f32> {
        af_scalar(l1_norm(vals))
    }

    fn model_type() -> ModelType {
        ModelType::LassoARD
    }

    fn build_cfg(cfg_bld: BranchCfgBuilder) -> BranchCfg {
        cfg_bld.build_ard()
    }

    // TODO: actually make some step sizes here
    fn std_scaled_step_sizes(&self, mcmc_cfg: &MCMCCfg) -> StepSizes {
        let _const_factor = mcmc_cfg.hmc_step_size_factor;
        let wrt_weights = Vec::with_capacity(self.num_layers());
        let wrt_biases = Vec::with_capacity(self.num_layers() - 1);

        StepSizes {
            wrt_weights,
            wrt_biases,
            wrt_weight_precisions: None,
            wrt_bias_precisions: None,
            wrt_error_precision: None,
        }
    }

    // TODO: this makes massive step sizes sometimes.
    fn izmailov_step_sizes(&mut self, mcmc_cfg: &MCMCCfg) -> StepSizes {
        let integration_length = mcmc_cfg.hmc_integration_length;
        let mut wrt_weights: Vec<Array<f32>> = Vec::with_capacity(self.num_layers());
        let mut wrt_biases = Vec::with_capacity(self.num_layers() - 1);

        // ard layers
        for index in 0..self.output_layer_index() {
            wrt_weights.push(tile(
                &(std::f32::consts::PI
                    / (2f32
                        * sqrt(&self.precisions().weight_precisions[index])
                        * integration_length as f32)),
                dim4!(1, self.layer_widths[index] as u64, 1, 1),
            ));
        }

        // output layer is base
        wrt_weights.push(
            std::f32::consts::PI
                / (2f32
                    * sqrt(&self.precisions().weight_precisions[self.output_layer_index()])
                    * integration_length as f32),
        );

        // there is only one bias precision per layer here
        for index in 0..self.output_layer_index() {
            let step_size = std::f32::consts::PI
                / (2.0f32
                    * arrayfire::sqrt(&self.precisions().bias_precisions[index])
                    * integration_length as f32);
            wrt_biases
                .push(arrayfire::constant(1.0f32, self.layer_biases(index).dims()) * step_size);
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
            log_density -= arrayfire::dot(
                &(l1_norm_rows(params.layer_weights(i)) + 1.0 / scale),
                precisions.layer_weight_precisions(i),
                MatProp::NONE,
                MatProp::NONE,
            );
            let nrows = params.layer_weights(i).dims().get()[0];
            let ncols = params.layer_weights(i).dims().get()[1];
            log_density += arrayfire::dot(
                &((shape + ncols as f32 - 1.0f32) * arrayfire::constant(1.0f32, dim4!(nrows))),
                &arrayfire::log(precisions.layer_weight_precisions(i)),
                MatProp::NONE,
                MatProp::NONE,
            );
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
        let global_sum_abs =
            l1_norm(params.layer_weights(i)) + self.output_weight_summary_stats().reg_sum();
        log_density -= (global_sum_abs + 1.0 / scale) * precisions.layer_weight_precisions(i);

        log_density += (shape + self.output_weight_summary_stats().num_params() - 1.0f32)
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
        for i in 0..self.output_layer_index() {
            log_density -= arrayfire::dot(
                &(l1_norm_rows(params.layer_weights(i))),
                precisions.layer_weight_precisions(i),
                MatProp::NONE,
                MatProp::NONE,
            );
        }

        let i = self.output_layer_index();
        log_density -= l1_norm(params.layer_weights(i)) * precisions.layer_weight_precisions(i);

        log_density
    }

    fn log_density_gradient_wrt_weights(&self) -> Vec<Array<f32>> {
        let mut ldg_wrt_weights: Vec<Array<f32>> = Vec::with_capacity(self.num_layers);
        // ard layer weights
        for layer_index in 0..self.output_layer_index() {
            let prec_m = arrayfire::tile(
                self.weight_precisions(layer_index),
                dim4!(1, self.layer_weights(layer_index).dims().get()[1], 1, 1),
            );
            ldg_wrt_weights.push(
                -(self.error_precision() * self.layer_d_rss_wrt_weights(layer_index)
                    + prec_m * sign(self.layer_weights(layer_index))),
            );
        }

        // output layer is base
        ldg_wrt_weights.push(
            -(self.error_precision() * self.layer_d_rss_wrt_weights(self.output_layer_index())
                + self.weight_precisions(self.output_layer_index())
                    * sign(self.layer_weights(self.output_layer_index()))),
        );

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
                (shape + precisions.elements() as f32 - 1.0) / precisions
                    - (1.0 / scale)
                    - l1_norm_rows(params),
            );
        }

        let layer_index = self.output_layer_index();
        let precisions: &Array<f32> = self.weight_precisions(layer_index);
        let params: &Array<f32> = self.layer_weights(layer_index);
        let (shape, scale) = hyperparams.layer_prior_hyperparams(layer_index, self.num_layers);
        ldg_wrt_weight_precisions.push(
            (shape + self.output_weight_summary_stats().num_params() - 1.0f32) / (precisions)
                - (1.0f32 / scale)
                - (l1_norm(params) + self.output_weight_summary_stats().reg_sum()),
        );

        ldg_wrt_weight_precisions
    }

    fn precision_posterior_host(
        &mut self,
        // k
        prior_shape: f32,
        // s or theta
        prior_scale: f32,
        reg_sum: f32,
        num_vals: usize,
    ) -> f32 {
        super::super::gibbs_steps::lasso_multi_param_precision_posterior_host_prepared_summary_stats(
            prior_shape,
            prior_scale,
            reg_sum,
            num_vals,
            self.rng(),
        )
    }

    /// Samples precision values from their posterior distribution in a Gibbs step.
    fn sample_prior_precisions(&mut self, hyperparams: &NetworkPrecisionHyperparameters) {
        for i in 0..self.output_layer_index() {
            let (prior_shape, prior_scale) =
                hyperparams.layer_prior_hyperparams(i, self.num_layers());

            // wrt weights
            let param_group_size = self.layer_width(i) as f32;
            let posterior_shape = param_group_size + prior_shape;
            // compute l1 norm of all rows
            self.precisions.weight_precisions[i] = Array::new(
                &to_host(&l1_norm_rows(&self.params.weights[i]))
                    .iter()
                    .map(|l1_norm| {
                        let posterior_scale = prior_scale / (1. + prior_scale * l1_norm);
                        Gamma::new(posterior_shape, posterior_scale)
                            .unwrap()
                            .sample(self.rng())
                    })
                    .collect::<Vec<f32>>(),
                self.precisions.weight_precisions[i].dims(),
            );

            // wrt biases (always l2 regularized)
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
    use arrayfire::{dim4, sum, Array};
    use assert_approx_eq::assert_approx_eq;
    // use arrayfire::{af_print, randu};

    use super::super::{
        branch_builder::BranchBuilder, branch_sampler::BranchSampler, branch_struct::BranchStruct,
    };
    use super::LassoArdBranch;

    use crate::af_helpers::{scalar_to_host, to_host};
    use crate::net::branch::momentum::BranchMomentum;
    use crate::net::mcmc_cfg::MCMCCfg;
    use crate::net::params::{
        BranchParams, NetworkPrecisionHyperparameters, OutputWeightSummaryStats,
        PrecisionHyperparameters,
    };

    fn assert_approx_eq_slice(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (ai, bi) in a.iter().zip(b.iter()) {
            assert_approx_eq!(ai, bi, tol);
        }
    }

    #[test]
    fn af_sign() {
        let a = Array::new(&[0f32, 2f32, -2f32], dim4![3, 1, 1, 1]);

        let neg = arrayfire::sign(&a);
        let pos = arrayfire::gt(&a, &0f32, false);
        let a_dims = *a.dims().get();
        let sign =
            arrayfire::constant!(0f32; a_dims[0], a_dims[1], a_dims[2], a_dims[3]) - neg + pos;

        assert_eq!(to_host(&sign), vec![0f32, 1f32, -1f32]);
    }

    // #[test]
    // fn af() {
    //     let num_rows: u64 = 5;
    //     let num_cols: u64 = 3;
    //     let dims = Dim4::new(&[num_rows, num_cols, 1, 1]);
    //     let a = randu::<f32>(dims);
    //     af_print!("Create a 5-by-3 matrix of random floats on the GPU", a);
    // }

    #[test]
    fn af_sum_axis() {
        // AF sum(_, 1) sums along columns, i.e. the output vector has entries equal to the number of rows
        // of the _ matrix.
        let a = Array::new(&[0f32, 1f32, 2f32, 3f32, 4f32, 5f32], dim4![3, 2, 1, 1]);
        assert_eq!(to_host(&sum(&a, 1)), vec![3f32, 5f32, 7f32]);
    }

    // this actually causes undefined behaviour.
    #[test]
    fn af_array_creation_broadcast() {
        let a = Array::new(&[0., 1., 3.], dim4![3, 2, 1, 1]);
        assert!(to_host(&a) != vec![0., 1., 3., 0., 1., 3.]);
    }

    fn make_test_branch() -> LassoArdBranch {
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
            .build_lasso_ard()
    }

    fn make_test_branch_with_precision(precision: f32) -> LassoArdBranch {
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
            .build_lasso_ard()
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

        assert_eq!(scalar_to_host(&ld_wrt_w), -30.150764);

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

        assert_eq!(ld, -35.520966);
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
                &[-0.0010378566, -2.001093, -2.0000277, -2.0, -2.0, -2.0],
                dim4![3, 2, 1, 1],
            ),
            Array::new(&[-2.0029104, -2.0035105], dim4!(2, 1, 1, 1)),
            Array::new(&[-8.997393], dim4!(1, 1, 1, 1)),
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
            Array::new(&[-1.0, -3.0, -5.0], dim4!(3)),
            Array::new(&[0.5, -0.5], dim4!(2)),
            Array::new(&[-0.20000005], dim4!(1)),
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
                &[
                    -0.0005189283,
                    -1.0005465,
                    -1.0000138,
                    -1.0000000010532997,
                    -1.00000000114826,
                    -1.000000000000059,
                ],
                dim4![3, 2, 1, 1],
            ),
            Array::new(&[-1.0014552, -1.0017552], dim4![2, 1, 1, 1]),
            Array::new(&[-4.4986963], dim4![1, 1, 1, 1]),
        ];

        let exp_ldg_wrt_b = [
            Array::new(&[-0.00053271546, -1.0000000011007801], dim4![2, 1, 1, 1]),
            Array::new(&[-2.0017552], dim4![1, 1, 1, 1]),
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
    fn num_log_density_gradient() {
        let num_individuals = 4;
        let num_markers = 3;
        let mut branch = make_test_branch();
        let x_train: Array<f32> = Array::new(
            &[1., 0., 0., 2., 1., 1., 2., 0., 0., 2., 0., 1.],
            dim4![num_individuals, num_markers, 1, 1],
        );
        let y_train: Array<f32> = Array::new(&[0.0, 2.0, 1.0, 1.5], dim4![4, 1, 1, 1]);
        let ldg = branch.numerical_log_density_gradient(&x_train, &y_train);

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
                &[
                    // here the first entry is also -1, because of the sharp transition of the gradient between theta=0 and theta != 1
                    // in the regularization term
                    -1.0005189,
                    -1.0005465,
                    -1.0000138,
                    -1.0000000010532997,
                    -1.00000000114826,
                    -1.000000000000059,
                ],
                dim4![3, 2, 1, 1],
            ),
            Array::new(&[-1.0014552, -1.0017552], dim4![2, 1, 1, 1]),
            Array::new(&[-4.4986963], dim4![1, 1, 1, 1]),
        ];

        let exp_ldg_wrt_b = [
            Array::new(&[-0.00053271546, -1.0000000011007801], dim4![2, 1, 1, 1]),
            Array::new(&[-2.0017552], dim4![1, 1, 1, 1]),
        ];

        // correct values
        for i in 0..(branch.num_layers) {
            assert_approx_eq_slice(
                &to_host(&ldg.wrt_weights[i]),
                &to_host(&exp_ldg_wrt_w[i]),
                1e-2f32,
            );
        }
        for i in 0..(branch.num_layers - 1) {
            assert_approx_eq_slice(
                &to_host(&ldg.wrt_biases[i]),
                &to_host(&exp_ldg_wrt_b[i]),
                1e-2f32,
            );
        }
    }

    #[test]
    fn uniform_step_sizes() {
        let branch = make_test_branch();
        let cfg = MCMCCfg {
            hmc_step_size_factor: 1.0,
            ..Default::default()
        };
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
}
