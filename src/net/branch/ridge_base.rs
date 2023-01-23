use super::{
    super::gibbs_steps::ridge_multi_param_precision_posterior,
    super::model_type::ModelType,
    super::params::{BranchParams, BranchPrecisions},
    branch::{Branch, BranchCfg},
    branch_cfg_builder::BranchCfgBuilder,
    step_sizes::StepSizes,
    training_state::TrainingState,
};
use crate::af_helpers::{af_scalar, scalar_to_host, sum_of_squares};
use crate::net::mcmc_cfg::MCMCCfg;
use crate::net::params::NetworkPrecisionHyperparameters;
use arrayfire::{dim4, sqrt, Array};
use rand::prelude::ThreadRng;
use rand::thread_rng;

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
}

impl Branch for RidgeBaseBranch {
    fn model_type() -> ModelType {
        ModelType::RidgeBase
    }

    fn build_cfg(cfg_bld: BranchCfgBuilder) -> BranchCfg {
        cfg_bld.build_base()
    }

    /// Creates Branch on device with BranchCfg from host memory.
    fn from_cfg(cfg: &BranchCfg) -> Self {
        Self {
            num_params: cfg.num_params,
            num_weights: cfg.num_weights,
            num_markers: cfg.num_markers,
            num_layers: cfg.layer_widths.len(),
            layer_widths: cfg.layer_widths.clone(),
            precisions: BranchPrecisions::from_host(&cfg.precisions),
            params: BranchParams::from_param_vec(&cfg.params, &cfg.layer_widths, cfg.num_markers),
            rng: thread_rng(),
            training_state: TrainingState::default(),
        }
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

    fn num_markers(&self) -> usize {
        self.num_markers
    }

    fn layer_widths(&self) -> &Vec<usize> {
        &self.layer_widths
    }

    fn precisions(&self) -> &BranchPrecisions {
        &self.precisions
    }

    fn precisions_mut(&mut self) -> &mut BranchPrecisions {
        &mut self.precisions
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn rng(&mut self) -> &mut ThreadRng {
        &mut self.rng
    }

    fn params_mut(&mut self) -> &mut BranchParams {
        &mut self.params
    }

    fn set_params(&mut self, params: &BranchParams) {
        self.params = params.clone();
    }

    fn params(&self) -> &BranchParams {
        &self.params
    }

    fn num_params(&self) -> usize {
        self.num_params
    }

    fn layer_width(&self, index: usize) -> usize {
        self.layer_widths[index]
    }

    fn set_error_precision(&mut self, val: f32) {
        self.precisions.error_precision = af_scalar(val);
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
                std::f32::consts::PI
                    / (2f32
                        * sqrt(&self.precisions().weight_precisions[index])
                        * integration_length as f32),
            );
        }

        for index in 0..self.num_layers() - 1 {
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

    fn log_density_joint_wrt_weights(
        &self,
        params: &BranchParams,
        precisions: &BranchPrecisions,
        hyperparams: &NetworkPrecisionHyperparameters,
    ) -> Array<f32> {
        let mut log_density: Array<f32> = af_scalar(0.0);

        // weight terms
        for i in 0..self.num_layers() {
            let (shape, scale) = hyperparams.layer_prior_hyperparams(i, self.num_layers());
            log_density -= (sum_of_squares(params.layer_weights(i)) / 2.0f32 + 1.0f32 / scale)
                * precisions.layer_weight_precisions(i);
            let nvar = params.layer_weights(i).elements();
            log_density += (shape + (2.0 * nvar as f32 - 2.0f32) / 2.0)
                * arrayfire::log(&precisions.layer_weight_precisions(i));
        }

        log_density
    }

    fn log_density(&self, params: &BranchParams, precisions: &BranchPrecisions, rss: f32) -> f32 {
        let mut log_density: f32 = scalar_to_host(&(-0.5f32 * &precisions.error_precision * rss));
        for i in 0..self.num_layers() {
            log_density -= 0.5
                * arrayfire::sum_all(
                    &(&precisions.weight_precisions[i]
                        * &(params.layer_weights(i) * params.layer_weights(i))),
                )
                .0;
        }
        for i in 0..self.num_layers() - 1 {
            log_density -= 0.5
                * arrayfire::sum_all(
                    &(params.layer_biases(i)
                        * params.layer_biases(i)
                        * &precisions.bias_precisions[i]),
                )
                .0;
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
        for layer_index in 0..self.num_layers() {
            let precisions: &Array<f32> = self.weight_precisions(layer_index);
            let params: &Array<f32> = self.layer_weights(layer_index);
            let (shape, scale) = hyperparams.layer_prior_hyperparams(layer_index, self.num_layers);
            ldg_wrt_weight_precisions.push(
                (2.0f32 * shape + precisions.elements() as f32 - 2.0) / (2.0f32 * precisions)
                    - (1.0f32 / scale)
                    - sum_of_squares(params),
            );
        }
        ldg_wrt_weight_precisions
    }

    fn precision_posterior_host(
        // k
        prior_shape: f32,
        // s or theta
        prior_scale: f32,
        param_vals: &[f32],
        rng: &mut ThreadRng,
    ) -> f32 {
        super::super::gibbs_steps::lasso_multi_param_precision_posterior_host(
            prior_shape,
            prior_scale,
            param_vals,
            rng,
        )
    }

    /// Samples precision values from their posterior distribution in a Gibbs step.
    fn sample_prior_precisions(&mut self, hyperparams: &NetworkPrecisionHyperparameters) {
        // output precision is sampled jointly for all branches
        for i in 0..self.num_layers() - 2 {
            self.precisions.weight_precisions[i] = Array::new(
                &[ridge_multi_param_precision_posterior(
                    hyperparams.dense_layer_prior_shape(),
                    hyperparams.dense_layer_prior_scale(),
                    &self.params.weights[i],
                    &mut self.rng,
                )],
                dim4!(1, 1, 1, 1),
            );
        }
        for i in 0..self.num_layers() - 2 {
            self.precisions.bias_precisions[i] = af_scalar(ridge_multi_param_precision_posterior(
                hyperparams.dense_layer_prior_shape(),
                hyperparams.dense_layer_prior_scale(),
                &self.params.biases[i],
                &mut self.rng,
            ));
        }

        // sample summary layer weights in base manner
        let summary_layer_index = self.summary_layer_index();
        self.precisions.weight_precisions[summary_layer_index] = Array::new(
            &[ridge_multi_param_precision_posterior(
                hyperparams.summary_layer_prior_shape(),
                hyperparams.summary_layer_prior_scale(),
                &self.params.weights[summary_layer_index],
                &mut self.rng,
            )],
            self.precisions.weight_precisions[summary_layer_index].dims(),
        );

        // sample summary layer biases with summary layer hyperparams
        self.precisions.bias_precisions[summary_layer_index] =
            af_scalar(ridge_multi_param_precision_posterior(
                hyperparams.summary_layer_prior_shape(),
                hyperparams.summary_layer_prior_scale(),
                &self.params.biases[summary_layer_index],
                &mut self.rng,
            ));
    }
}

#[cfg(test)]
mod tests {
    use crate::net::mcmc_cfg::MCMCCfg;
    use arrayfire::{dim4, Array};
    // use arrayfire::{af_print, randu};

    use super::super::{branch::Branch, branch_builder::BranchBuilder};
    use super::RidgeBaseBranch;

    use crate::af_helpers::to_host;
    use crate::net::branch::momentum::BranchMomentum;
    use crate::net::params::BranchParams;

    // #[test]
    // fn test_af() {
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
        BranchParams { weights, biases }
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
    fn test_forward_feed() {
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

    // TODO: this test is flaky, depending on the hardware it is run on. Should use approx
    // comparisons instead.
    // #[test]
    // fn test_backpropagation() {
    //     let num_individuals = 4;
    //     let num_markers = 3;
    //     let branch = make_test_branch();
    //     let x_train: Array<f32> = Array::new(
    //         &[1., 0., 0., 2., 1., 1., 2., 0., 0., 2., 0., 1.],
    //         dim4![num_individuals, num_markers, 1, 1],
    //     );
    //     let y_train: Array<f32> = Array::new(&[0.0, 2.0, 1.0, 1.5], dim4![4, 1, 1, 1]);
    //     let (weights_gradient, bias_gradient) = branch.backpropagate(&x_train, &y_train);

    //     // correct number of gradients
    //     assert_eq!(weights_gradient.len(), branch.num_layers);
    //     assert_eq!(bias_gradient.len(), branch.num_layers - 1);

    //     // correct dimensions of gradients
    //     for i in 0..(branch.num_layers) {
    //         assert_eq!(weights_gradient[i].dims(), branch.weights(i).dims());
    //     }
    //     for i in 0..(branch.num_layers - 1) {
    //         assert_eq!(bias_gradient[i].dims(), branch.biases(i).dims());
    //     }

    //     let exp_weight_grad = [
    //         Array::new(
    //             &[
    //                 0.0005189283,
    //                 0.00054650265,
    //                 1.37817915e-5,
    //                 1.1157868e-9,
    //                 1.3018558e-9,
    //                 0.0,
    //             ],
    //             dim4![3, 2, 1, 1],
    //         ),
    //         Array::new(&[0.0014552199, 0.0017552056], dim4![2, 1, 1, 1]),
    //         Array::new(&[3.4986966], dim4![1, 1, 1, 1]),
    //     ];

    //     let exp_bias_grad = [
    //         Array::new(&[0.00053271546, 1.2088213e-9], dim4![2, 1, 1, 1]),
    //         Array::new(&[0.0017552058], dim4![1, 1, 1, 1]),
    //     ];

    //     // correct values of gradient
    //     for i in 0..(branch.num_layers) {
    //         assert_eq!(to_host(&weights_gradient[i]), to_host(&exp_weight_grad[i]));
    //     }
    //     for i in 0..(branch.num_layers - 1) {
    //         assert_eq!(to_host(&bias_gradient[i]), to_host(&exp_bias_grad[i]));
    //     }
    // }

    #[test]
    fn test_log_density_gradient() {
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
            Array::new(&[-0.00053271546, -1.0], dim4![2, 1, 1, 1]),
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
    fn test_uniform_step_sizes() {
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
    fn test_net_movement() {
        let branch = make_test_branch();
        let momenta = make_test_uniform_momenta(1.);
        let init_params = make_test_uniform_params(0.);
        assert!(branch.net_movement(&init_params, &momenta) > 0.0);
        let init_params = make_test_uniform_params(100.);
        assert!(branch.net_movement(&init_params, &momenta) < 0.0);
    }
}
