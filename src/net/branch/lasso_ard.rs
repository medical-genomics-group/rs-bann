use super::{
    super::af_helpers::sign,
    super::gibbs_steps::lasso_multi_param_precision_posterior,
    super::model_type::ModelType,
    super::params::{BranchParams, BranchPrecisions},
    branch::{Branch, BranchCfg, BranchLogDensityGradient},
    branch_cfg_builder::BranchCfgBuilder,
    step_sizes::StepSizes,
};
use crate::{net::params::NetworkPrecisionHyperparameters, to_host};
use arrayfire::{abs, dim4, matmul, sqrt, sum, sum_all, tile, Array, MatProp};
use rand::prelude::ThreadRng;
use rand::thread_rng;
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
}

// Weights in this branch are grouped by the node they
// are going out of.
impl Branch for LassoArdBranch {
    fn model_type() -> ModelType {
        ModelType::ARD
    }

    fn build_cfg(cfg_bld: BranchCfgBuilder) -> BranchCfg {
        cfg_bld.build_ard()
    }

    /// Creates Branch on device with BranchCfg from host memory.
    fn from_cfg(cfg: &BranchCfg) -> Self {
        Self {
            num_params: cfg.num_params,
            num_weights: cfg.num_weights,
            num_markers: cfg.num_markers,
            num_layers: cfg.layer_widths.len(),
            layer_widths: cfg.layer_widths.clone(),
            precisions: cfg.precisions.clone(),
            params: BranchParams::from_param_vec(&cfg.params, &cfg.layer_widths, cfg.num_markers),
            rng: thread_rng(),
        }
    }

    /// Dumps all branch info into a BranchCfg object stored in host memory.
    fn to_cfg(&self) -> BranchCfg {
        BranchCfg {
            num_params: self.num_params,
            num_weights: self.num_weights,
            num_markers: self.num_markers,
            layer_widths: self.layer_widths.clone(),
            params: self.params.param_vec(),
            precisions: self.precisions.clone(),
        }
    }

    fn precisions(&self) -> &BranchPrecisions {
        &self.precisions
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
        self.precisions.error_precision = val;
    }

    // TODO: actually make some step sizes here
    fn std_scaled_step_sizes(&self, _const_factor: f32) -> StepSizes {
        let wrt_weights = Vec::with_capacity(self.num_layers());
        let wrt_biases = Vec::with_capacity(self.num_layers() - 1);

        StepSizes {
            wrt_weights,
            wrt_biases,
        }
    }

    // TODO: this makes massive step sizes sometimes.
    fn izmailov_step_sizes(&mut self, integration_length: usize) -> StepSizes {
        let mut wrt_weights: Vec<Array<f32>> = Vec::with_capacity(self.num_layers());
        let mut wrt_biases = Vec::with_capacity(self.num_layers() - 1);

        for index in 0..self.num_layers() {
            wrt_weights.push(tile(
                &(std::f32::consts::PI
                    / (2f32
                        * sqrt(&self.precisions().weight_precisions[index])
                        * integration_length as f32)),
                dim4!(1, self.layer_widths[index] as u64, 1, 1),
            ));
        }

        // there is only one bias precision per layer here
        for index in 0..self.num_layers() - 1 {
            wrt_biases.push(Array::new(
                &vec![
                    std::f32::consts::PI
                        / (2.
                            * &self.precisions().bias_precisions[index].sqrt()
                            * integration_length as f32);
                    self.biases(index).elements()
                ],
                self.biases(index).dims(),
            ));
        }

        StepSizes {
            wrt_weights,
            wrt_biases,
        }
    }

    // TODO: write test
    fn log_density(&self, params: &BranchParams, precisions: &BranchPrecisions, rss: f32) -> f32 {
        let mut log_density: f32 = -0.5 * precisions.error_precision * rss;

        for i in 0..self.summary_layer_index() {
            log_density -= sum_all(&matmul(
                &(abs(params.weights(i))),
                self.weight_precisions(i),
                MatProp::TRANS,
                MatProp::NONE,
            ))
            .0;
        }
        // summary layer (is always Base)
        log_density -= sum_all(
            &(self.weight_precisions(self.summary_layer_index())
                * &(abs(params.weights(self.summary_layer_index())))),
        )
        .0;

        // output layer
        log_density -= sum_all(
            &(self.weight_precisions(self.output_layer_index())
                * &(abs(params.weights(self.output_layer_index())))),
        )
        .0;

        for i in 0..self.num_layers() - 1 {
            log_density -= precisions.bias_precisions[i] * sum_all(&(abs(params.biases(i)))).0;
        }

        log_density
    }

    fn log_density_gradient(
        &self,
        x_train: &Array<f32>,
        y_train: &Array<f32>,
    ) -> BranchLogDensityGradient {
        let (d_rss_wrt_weights, d_rss_wrt_biases) = self.backpropagate(x_train, y_train);
        let mut ldg_wrt_weights: Vec<Array<f32>> = Vec::with_capacity(self.num_layers);
        let mut ldg_wrt_biases: Vec<Array<f32>> = Vec::with_capacity(self.num_layers - 1);

        for layer_index in 0..self.num_layers() {
            let prec_m = arrayfire::tile(
                self.weight_precisions(layer_index),
                dim4!(1, self.weights(layer_index).dims().get()[1], 1, 1),
            );

            ldg_wrt_weights.push(
                -(self.error_precision() * &d_rss_wrt_weights[layer_index]
                    + prec_m * sign(self.weights(layer_index))),
            );
        }

        for layer_index in 0..self.num_layers - 1 {
            ldg_wrt_biases.push(
                -self.bias_precision(layer_index) * sign(self.biases(layer_index))
                    - self.error_precision() * &d_rss_wrt_biases[layer_index],
            );
        }

        BranchLogDensityGradient {
            wrt_weights: ldg_wrt_weights,
            wrt_biases: ldg_wrt_biases,
        }
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
        // this iterates over layers
        // the last two layers (summary and output)
        // are sampled in the base layer way, otherwise there is overwhelming
        // influence of the hyperparameters.
        for i in 0..self.num_layers() - 2 {
            let param_group_size = self.layer_width(i) as f32;
            let posterior_shape = param_group_size + hyperparams.dense_layer_prior_shape();
            // compute l1 norm of all rows
            self.precisions.weight_precisions[i] = Array::new(
                &to_host(&sum(&abs(&self.params.weights[i]), 1))
                    .iter()
                    .map(|l1_norm| {
                        let posterior_scale = hyperparams.dense_layer_prior_scale()
                            / (1. + hyperparams.dense_layer_prior_scale() * l1_norm);
                        Gamma::new(posterior_shape, posterior_scale)
                            .unwrap()
                            .sample(self.rng())
                    })
                    .collect::<Vec<f32>>(),
                self.precisions.weight_precisions[i].dims(),
            );
        }

        // output precision is sampled jointly for all branches, not here

        for i in 0..self.num_layers() - 2 {
            self.precisions.bias_precisions[i] = lasso_multi_param_precision_posterior(
                hyperparams.dense_layer_prior_shape(),
                hyperparams.dense_layer_prior_scale(),
                &self.params.biases[i],
                &mut self.rng,
            );
        }

        // sample summary layer weights in base manner
        let summary_layer_index = self.summary_layer_index();
        self.precisions.weight_precisions[summary_layer_index] = Array::new(
            &[lasso_multi_param_precision_posterior(
                hyperparams.summary_layer_prior_shape(),
                hyperparams.summary_layer_prior_scale(),
                &self.params.weights[summary_layer_index],
                &mut self.rng,
            )],
            self.precisions.weight_precisions[summary_layer_index].dims(),
        );

        // sample summary layer biases with summary layer hyperparams
        self.precisions.bias_precisions[summary_layer_index] =
            lasso_multi_param_precision_posterior(
                hyperparams.summary_layer_prior_shape(),
                hyperparams.summary_layer_prior_scale(),
                &self.params.biases[summary_layer_index],
                &mut self.rng,
            );
    }
}

#[cfg(test)]
mod tests {
    use arrayfire::{dim4, sum, Array};
    use assert_approx_eq::assert_approx_eq;
    // use arrayfire::{af_print, randu};

    use super::super::{branch::Branch, branch_builder::BranchBuilder};
    use super::LassoArdBranch;

    use crate::net::branch::momenta::BranchMomenta;
    use crate::net::params::BranchParams;
    use crate::to_host;

    fn assert_approx_eq_slice(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (ai, bi) in a.iter().zip(b.iter()) {
            assert_approx_eq!(ai, bi, tol);
        }
    }

    #[test]
    fn test_af_sign() {
        let a = Array::new(&[0f32, 2f32, -2f32], dim4![3, 1, 1, 1]);

        let neg = arrayfire::sign(&a);
        let pos = arrayfire::gt(&a, &0f32, false);
        let a_dims = *a.dims().get();
        let sign =
            arrayfire::constant!(0f32; a_dims[0], a_dims[1], a_dims[2], a_dims[3]) - neg + pos;

        assert_eq!(to_host(&sign), vec![0f32, 1f32, -1f32]);
    }

    // #[test]
    // fn test_af() {
    //     let num_rows: u64 = 5;
    //     let num_cols: u64 = 3;
    //     let dims = Dim4::new(&[num_rows, num_cols, 1, 1]);
    //     let a = randu::<f32>(dims);
    //     af_print!("Create a 5-by-3 matrix of random floats on the GPU", a);
    // }

    #[test]
    fn test_af_sum_axis() {
        // AF sum(_, 1) sums along columns, i.e. the output vector has entries equal to the number of rows
        // of the _ matrix.
        let a = Array::new(&[0f32, 1f32, 2f32, 3f32, 4f32, 5f32], dim4![3, 2, 1, 1]);
        assert_eq!(to_host(&sum(&a, 1)), vec![3f32, 5f32, 7f32]);
    }

    // this actually causes undefined behaviour.
    #[test]
    fn test_af_array_creation_broadcast() {
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

    fn make_test_uniform_momenta(c: f32) -> BranchMomenta {
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
        BranchMomenta {
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
        let branch = make_test_branch();
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
            assert_eq!(ldg.wrt_weights[i].dims(), branch.weights(i).dims());
        }
        for i in 0..(branch.num_layers - 1) {
            assert_eq!(ldg.wrt_biases[i].dims(), branch.biases(i).dims());
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
            Array::new(&[-1.0017552], dim4![1, 1, 1, 1]),
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
    fn test_num_log_density_gradient() {
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
            assert_eq!(ldg.wrt_weights[i].dims(), branch.weights(i).dims());
        }
        for i in 0..(branch.num_layers - 1) {
            assert_eq!(ldg.wrt_biases[i].dims(), branch.biases(i).dims());
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
            Array::new(&[-1.0005327, -1.0000000011007801], dim4![2, 1, 1, 1]),
            Array::new(&[-1.0017552], dim4![1, 1, 1, 1]),
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
    fn test_uniform_step_sizes() {
        let branch = make_test_branch();
        let val = 1.0;
        let step_sizes = branch.uniform_step_sizes(val);
        for i in 0..(branch.num_layers - 1) {
            let mut obs = to_host(&step_sizes.wrt_weights[i]);
            assert_eq!(obs, vec![val; obs.len()]);
            obs = to_host(&step_sizes.wrt_biases[i]);
            assert_eq!(obs, vec![val; obs.len()]);
        }
        let obs = to_host(&step_sizes.wrt_weights[branch.num_layers - 1]);
        assert_eq!(obs, vec![val; obs.len()]);
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
