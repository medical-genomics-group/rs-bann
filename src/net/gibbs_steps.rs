use rand::rngs::ThreadRng;
use rand_distr::{Distribution, Gamma};

fn single_param_precision_posterior(
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

fn multi_param_precision_posterior(
    // k
    prior_shape: f64,
    // s or theta
    prior_scale: f64,
    param_vals: Vec<f64>,
    rng: &mut ThreadRng,
) -> f64 {
    let num_params = param_vals.len();
    let mut sum_of_squares = 0.0;
    for v in param_vals {
        sum_of_squares += v * v;
    }
    let posterior_shape = prior_shape + num_params as f64 / 2.;
    let posterior_scale = 2. * prior_scale / (2. + prior_scale * sum_of_squares);
    Gamma::new(posterior_shape, posterior_scale)
        .unwrap()
        .sample(rng)
}
