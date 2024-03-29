//! Sampling from the precision posterior distributions.

use crate::af_helpers::{l1_norm, sum_of_squares};
use arrayfire::Array;
use log::debug;
use rand::{rngs::ThreadRng, Rng};
use rand_distr::{Distribution, Gamma};

pub(crate) fn ridge_single_param_precision_posterior(
    // k
    prior_shape: f32,
    // s or theta
    prior_scale: f32,
    param_val: f32,
    rng: &mut ThreadRng,
) -> f32 {
    let square = param_val * param_val;
    let posterior_shape = prior_shape + 0.5;
    let posterior_scale = 2. * prior_scale / (2. + prior_scale * square);
    Gamma::new(posterior_shape, posterior_scale)
        .unwrap()
        .sample(rng)
}

pub(crate) fn lasso_multi_param_precision_posterior(
    // k
    prior_shape: f32,
    // s or theta
    prior_scale: f32,
    param_vals: &Array<f32>,
    rng: &mut ThreadRng,
) -> f32 {
    let num_params = param_vals.elements();
    let posterior_shape = prior_shape + num_params as f32;
    let posterior_scale = prior_scale / (1. + prior_scale * l1_norm(param_vals));
    Gamma::new(posterior_shape, posterior_scale)
        .unwrap()
        .sample(rng)
}

pub(crate) fn lasso_multi_param_precision_posterior_host_prepared_summary_stats(
    // k
    prior_shape: f32,
    // s or theta
    prior_scale: f32,
    sum_of_abs: f32,
    num_vals: usize,
    rng: &mut ThreadRng,
) -> f32 {
    let num_params = num_vals;
    let posterior_shape = prior_shape + num_params as f32;
    let l1_norm: f32 = sum_of_abs;
    let posterior_scale = prior_scale / (1. + prior_scale * l1_norm);
    Gamma::new(posterior_shape, posterior_scale)
        .unwrap()
        .sample(rng)
}

// pub(crate) fn lasso_multi_param_precision_posterior_host(
//     // k
//     prior_shape: f32,
//     // s or theta
//     prior_scale: f32,
//     param_vals: &[f32],
//     rng: &mut ThreadRng,
// ) -> f32 {
//     let num_params = param_vals.len();
//     let posterior_shape = prior_shape + num_params as f32;
//     let l1_norm: f32 = param_vals.iter().map(|e| e.abs()).sum();
//     let posterior_scale = prior_scale / (1. + prior_scale * l1_norm);
//     Gamma::new(posterior_shape, posterior_scale)
//         .unwrap()
//         .sample(rng)
// }

pub(crate) fn ridge_multi_param_precision_posterior(
    // k
    prior_shape: f32,
    // s or theta
    prior_scale: f32,
    param_vals: &Array<f32>,
    rng: &mut impl Rng,
) -> f32 {
    let num_params = param_vals.elements();
    let posterior_shape = prior_shape + num_params as f32 / 2.;
    let posterior_scale = 2. * prior_scale / (2. + prior_scale * sum_of_squares(param_vals));
    debug!(
        "Posterior shape: {:?}; Posterior scale: {:?}",
        posterior_shape, posterior_scale
    );
    Gamma::new(posterior_shape, posterior_scale)
        .unwrap()
        .sample(rng)
}

// pub(crate) fn ridge_multi_param_precision_posterior_host(
//     // k
//     prior_shape: f32,
//     // s or theta
//     prior_scale: f32,
//     param_vals: &[f32],
//     rng: &mut ThreadRng,
// ) -> f32 {
//     let num_params = param_vals.len();
//     let sum_of_squares: f32 = param_vals.iter().map(|e| e * e).sum();
//     let posterior_shape = prior_shape + num_params as f32 / 2.;
//     let posterior_scale = 2. * prior_scale / (2. + prior_scale * sum_of_squares);
//     Gamma::new(posterior_shape, posterior_scale)
//         .unwrap()
//         .sample(rng)
// }

pub(crate) fn ridge_multi_param_precision_posterior_host_prepared_summary_stats(
    // k
    prior_shape: f32,
    // s or theta
    prior_scale: f32,
    summary_stat: f32,
    num_vals: usize,
    rng: &mut ThreadRng,
) -> f32 {
    let num_params = num_vals;
    let sum_of_squares: f32 = summary_stat;
    let posterior_shape = prior_shape + num_params as f32 / 2.;
    let posterior_scale = 2. * prior_scale / (2. + prior_scale * sum_of_squares);
    Gamma::new(posterior_shape, posterior_scale)
        .unwrap()
        .sample(rng)
}
