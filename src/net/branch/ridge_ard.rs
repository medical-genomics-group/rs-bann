use super::{
    super::gibbs_steps::ridge_multi_param_precision_posterior,
    super::model_type::ModelType,
    super::params::{BranchParams, BranchPrecisions},
    branch::{Branch, BranchCfg},
    branch_cfg_builder::BranchCfgBuilder,
    step_sizes::StepSizes,
    training_state::TrainingState,
};
use crate::net::mcmc_cfg::MCMCCfg;
use crate::net::params::NetworkPrecisionHyperparameters;
use crate::{
    af_helpers::{af_scalar, sum_of_squares, sum_of_squares_rows, to_host},
    net::params::OutputWeightSummaryStats,
};
use arrayfire::{dim4, sqrt, tile, Array, MatProp};
use rand::prelude::ThreadRng;
use rand::thread_rng;
use rand_distr::{Distribution, Gamma};

pub struct RidgeArdBranch {
    pub(crate) num_params: usize,
    pub(crate) num_weights: usize,
    pub(crate) num_markers: usize,
    pub(crate) params: BranchParams,
    pub(crate) precisions: BranchPrecisions,
    pub(crate) layer_widths: Vec<usize>,
    pub(crate) num_layers: usize,
    pub(crate) rng: ThreadRng,
    pub(crate) training_state: TrainingState,
}

// Weights in this branch are grouped by the node they
// are going out of.
impl Branch for RidgeArdBranch {
    fn summary_stat_fn_host(vals: &[f32]) -> f32 {
        crate::arr_helpers::sum_of_squares(vals)
    }

    fn summary_stat_fn(&self, vals: &Array<f32>) -> Array<f32> {
        af_scalar(sum_of_squares(vals))
    }

    fn model_type() -> ModelType {
        ModelType::RidgeARD
    }

    fn build_cfg(cfg_bld: BranchCfgBuilder) -> BranchCfg {
        cfg_bld.build_ard()
    }

    /// Creates Branch on device with BranchCfg from host memory.
    fn from_cfg(cfg: &BranchCfg) -> Self {
        let mut res = Self {
            num_params: cfg.num_params,
            num_weights: cfg.num_weights,
            num_markers: cfg.num_markers,
            num_layers: cfg.layer_widths.len(),
            layer_widths: cfg.layer_widths.clone(),
            precisions: BranchPrecisions::from_host(&cfg.precisions),
            params: BranchParams::from_host(&cfg.params),
            rng: thread_rng(),
            training_state: TrainingState::default(),
        };

        res.subtract_output_weight_summary_stat_from_global();

        res
    }

    fn rng_mut(&mut self) -> &mut ThreadRng {
        &mut self.rng
    }

    fn output_weight_summary_stats(&self) -> &OutputWeightSummaryStats {
        &self.params.output_weight_summary_stats
    }

    fn output_weight_summary_stats_mut(&mut self) -> &mut OutputWeightSummaryStats {
        &mut self.params.output_weight_summary_stats
    }

    fn training_state(&self) -> &TrainingState {
        &self.training_state
    }

    fn training_state_mut(&mut self) -> &mut TrainingState {
        &mut self.training_state
    }

    fn num_weights(&self) -> usize {
        self.num_weights
    }

    fn precisions(&self) -> &BranchPrecisions {
        &self.precisions
    }

    fn precisions_mut(&mut self) -> &mut BranchPrecisions {
        &mut self.precisions
    }

    fn set_precisions(&mut self, precisions: &BranchPrecisions) {
        self.precisions = precisions.clone();
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn rng(&mut self) -> &mut ThreadRng {
        &mut self.rng
    }

    fn set_params(&mut self, params: &BranchParams) {
        self.params = params.clone();
    }

    fn params_mut(&mut self) -> &mut BranchParams {
        &mut self.params
    }

    fn params(&self) -> &BranchParams {
        &self.params
    }

    fn num_params(&self) -> usize {
        self.num_params
    }

    fn num_markers(&self) -> usize {
        self.num_markers
    }

    fn layer_widths(&self) -> &Vec<usize> {
        &self.layer_widths
    }

    fn layer_width(&self, index: usize) -> usize {
        self.layer_widths[index]
    }

    fn set_error_precision(&mut self, val: f32) {
        self.precisions.error_precision = af_scalar(val);
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

        // base layers
        wrt_weights.push(
            std::f32::consts::PI
                / (2f32
                    * sqrt(&self.precisions().weight_precisions[self.output_layer_index()])
                    * integration_length as f32),
        );

        // there is only one bias precision per layer here
        for index in 0..self.output_layer_index() {
            wrt_biases.push(
                arrayfire::constant(1.0f32, self.layer_biases(index).dims())
                    * (std::f32::consts::PI
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
            log_density -= arrayfire::dot(
                &(sum_of_squares_rows(params.layer_weights(i)) / 2.0f32 + 1.0f32 / scale),
                precisions.layer_weight_precisions(i),
                MatProp::NONE,
                MatProp::NONE,
            );
            let nrows = params.layer_weights(i).dims().get()[0];
            let ncols = params.layer_weights(i).dims().get()[1];
            log_density += arrayfire::dot(
                &((shape + (ncols as f32 - 2.0f32) / 2.0)
                    * arrayfire::constant(1.0f32, dim4!(nrows))),
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
        for i in 0..self.output_layer_index() {
            log_density -= arrayfire::dot(
                &(0.5f32 * &sum_of_squares_rows(params.layer_weights(i))),
                precisions.layer_weight_precisions(i),
                MatProp::NONE,
                MatProp::NONE,
            );
        }

        let i = self.output_layer_index();
        log_density -= 0.5f32
            * sum_of_squares(params.layer_weights(i))
            * precisions.layer_weight_precisions(i);

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
                    + prec_m * self.layer_weights(layer_index)),
            );
        }

        // output layer is base
        ldg_wrt_weights.push(
            -(self.error_precision() * self.layer_d_rss_wrt_weights(self.output_layer_index())
                + self.weight_precisions(self.output_layer_index())
                    * self.layer_weights(self.output_layer_index())),
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
                (2.0f32 * shape + precisions.elements() as f32 - 2.0) / (2.0f32 * precisions)
                    - (1.0f32 / scale)
                    - sum_of_squares_rows(params) / 2.0f32,
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
            &mut self.rng,
        )
    }

    /// Samples precision values from their posterior distribution in a Gibbs step.
    fn sample_prior_precisions(&mut self, hyperparams: &NetworkPrecisionHyperparameters) {
        // wrt weights
        for i in 0..self.output_layer_index() {
            let (prior_shape, prior_scale) =
                hyperparams.layer_prior_hyperparams(i, self.num_layers());

            // weights
            let param_group_size = self.layer_width(i) as f32;
            let posterior_shape = param_group_size / 2. + prior_shape;
            self.precisions.weight_precisions[i] = Array::new(
                &to_host(&sum_of_squares_rows(&self.params.weights[i]))
                    .iter()
                    .map(|sum_squares| {
                        let posterior_scale = 2. * prior_scale / (2. + prior_scale * sum_squares);
                        Gamma::new(posterior_shape, posterior_scale)
                            .unwrap()
                            .sample(self.rng())
                    })
                    .collect::<Vec<f32>>(),
                self.precisions.weight_precisions[i].dims(),
            );

            // wrt biases
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
    use arrayfire::{dim4, sum, Array};
    use assert_approx_eq::assert_approx_eq;
    // use arrayfire::{af_print, randu};

    use super::super::{
        branch::Branch, branch_builder::BranchBuilder, branch_cfg_builder::BranchCfgBuilder,
    };
    use super::RidgeArdBranch;

    use crate::af_helpers::{scalar_to_host, to_host};
    use crate::net::branch::momentum::BranchMomentum;
    use crate::net::params::BranchParams;

    fn assert_approx_eq_slice(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (ai, bi) in a.iter().zip(b.iter()) {
            assert_approx_eq!(ai, bi, tol);
        }
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

    fn make_test_branch() -> RidgeArdBranch {
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
            .build_ridge_ard()
    }

    fn make_test_branch_by_cfg() -> RidgeArdBranch {
        let mut bld = BranchCfgBuilder::new()
            .with_num_markers(3)
            .with_initial_weights_value(0.1)
            .with_initial_bias_value(0.0);
        bld.add_hidden_layer(2);

        RidgeArdBranch::from_cfg(&RidgeArdBranch::build_cfg(bld))
    }

    fn make_test_branch_with_precision(precision: f32) -> RidgeArdBranch {
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
            .build_ridge_ard()
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
        let activations = branch.forward_feed(&x_train);

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
    fn effect_size_dim() {
        let num_individuals = 4;
        let num_markers = 3;
        let branch = make_test_branch();
        let x_train: Array<f32> = Array::new(
            &[1., 0., 0., 2., 1., 1., 2., 0., 0., 2., 0., 1.],
            dim4![num_individuals, num_markers, 1, 1],
        );
        let y_train: Array<f32> = Array::new(&[0.0, 2.0, 1.0, 1.5], dim4![4, 1, 1, 1]);
        let effect_sizes = branch.effect_sizes(&x_train, &y_train);
        assert_eq!(
            effect_sizes.dims(),
            dim4!(num_individuals, num_markers, 1, 1)
        );
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

        assert_eq!(scalar_to_host(&ld_wrt_w), -57.269924);

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

        assert_eq!(ld, -62.640125);
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
            Array::new(&[-3.25, -7.25, -13.25], dim4!(3)),
            Array::new(&[0.5, -1.0], dim4!(2)),
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
    fn log_density_gradient_by_cfg() {
        let num_individuals = 4;
        let num_markers = 3;
        let mut branch = make_test_branch_by_cfg();
        let x_train: Array<f32> = Array::new(
            &[1., 0., 0., 2., 1., 1., 2., 0., 0., 2., 0., 1.],
            dim4![num_individuals, num_markers, 1, 1],
        );
        let y_train: Array<f32> = Array::new(&[0.0, 2.0, 1.0, 1.5], dim4![4, 1, 1, 1]);
        let _ldg = branch.log_density_gradient(&x_train, &y_train);
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
                    -2.0000138,
                    -3.0000000010532997,
                    -4.00000000114826,
                    -5.000000000000059,
                ],
                dim4![3, 2, 1, 1],
            ),
            Array::new(&[-1.0014552, -2.0017552], dim4![2, 1, 1, 1]),
            Array::new(&[-5.4986963], dim4![1, 1, 1, 1]),
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
                    -0.0005189283,
                    -1.0005465,
                    -2.0000138,
                    -3.0000000010532997,
                    -4.00000000114826,
                    -5.000000000000059,
                ],
                dim4![3, 2, 1, 1],
            ),
            Array::new(&[-1.0014552, -2.0017552], dim4![2, 1, 1, 1]),
            Array::new(&[-5.4986963], dim4![1, 1, 1, 1]),
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
}
