use arrayfire::{Array, MatProp};
use rand::rngs::ThreadRng;
use rand_distr::{Distribution, Gamma};

pub(crate) fn sum_of_squares(arr: &Array<f64>) -> f64 {
    let mut sum_of_squares = vec![0.0];
    arrayfire::dot(
        &arrayfire::flat(arr),
        &arrayfire::flat(arr),
        MatProp::NONE,
        MatProp::NONE,
    )
    .host(&mut sum_of_squares);
    sum_of_squares[0]
}

pub(crate) fn single_param_precision_posterior(
    // k
    prior_shape: f64,
    // s or theta
    prior_scale: f64,
    param_val: f64,
    rng: &mut ThreadRng,
) -> f64 {
    let square = param_val * param_val;
    let posterior_shape = prior_shape + 0.5;
    let posterior_scale = 2. * prior_scale / (2. + prior_scale * square);
    Gamma::new(posterior_shape, posterior_scale)
        .unwrap()
        .sample(rng)
}

pub(crate) fn multi_param_precision_posterior(
    // k
    prior_shape: f64,
    // s or theta
    prior_scale: f64,
    param_vals: &Array<f64>,
    rng: &mut ThreadRng,
) -> f64 {
    let num_params = param_vals.elements();
    let posterior_shape = prior_shape + num_params as f64 / 2.;
    let posterior_scale = 2. * prior_scale / (2. + prior_scale * sum_of_squares(param_vals));
    Gamma::new(posterior_shape, posterior_scale)
        .unwrap()
        .sample(rng)
}
