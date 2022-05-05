//! The network implementation

use ndarray::{arr1, s, Array1, ArrayView1};
use rand::prelude::ThreadRng;
use rand::{thread_rng, Rng};
use rs_bedvec::bedvec::BedVecCM;
use rs_bedvec::io::BedReader;
use rs_hmc::momentum::Momentum;
use rs_hmc::momentum::MultivariateStandardNormalMomentum;

type A = Array1<f32>;

#[inline(always)]
fn activation_fn(x: f32) -> f32 {
    f32::tanh(x)
}

#[inline(always)]
fn activation_fn_derivative(x: f32) -> f32 {
    1. - f32::tanh(x).powf(2.)
}

/// A group of markers
pub struct MarkerGroup {
    residual: A,
    w1: A,
    b1: f32,
    w2: f32,
    lambda_w1: f32,
    lambda_b1: f32,
    lambda_w2: f32,
    lambda_e: f32,
    bed_reader: BedReader,
    num_markers: usize,
    rng: ThreadRng,
    momentum_sampler: MultivariateStandardNormalMomentum,
    marker_data: Option<BedVecCM>,
}

impl MarkerGroup {
    pub fn new(
        residual: A,
        w1: A,
        b1: f32,
        w2: f32,
        bed_reader: BedReader,
        num_markers: usize,
    ) -> Self {
        assert_eq!(
            w1.len(),
            num_markers,
            "num_markers has to equal length of w1!"
        );
        Self {
            residual,
            w1,
            b1,
            w2,
            lambda_w1: 1.,
            lambda_b1: 1.,
            lambda_w2: 1.,
            lambda_e: 1.,
            bed_reader,
            num_markers,
            rng: thread_rng(),
            momentum_sampler: MultivariateStandardNormalMomentum::new(num_markers + 2),
            marker_data: None,
        }
    }

    pub fn set_params(&mut self, param_vec: &Array1<f32>) {
        let b1_index = 0;
        let w1_index_first = 1;
        let w1_index_last = self.num_markers;
        let w2_index = w1_index_last + 1;
        self.b1 = param_vec[b1_index];
        self.w1 = param_vec
            .slice(s![w1_index_first..=w1_index_last])
            .to_owned();
        self.w2 = param_vec[w2_index];
    }

    pub fn load_marker_data(&mut self) {
        self.marker_data = Some(self.bed_reader.read_into_bedvec());
    }

    pub fn forget_marker_data(&mut self) {
        self.marker_data = None;
    }

    pub fn forward_feed_with_set_params(&self) -> A {
        // this does not include b2, because it is not group specific
        (self
            .marker_data
            .as_ref()
            .unwrap()
            .right_multiply_par(self.w1.as_slice().unwrap())
            + self.b1)
            .mapv(activation_fn)
            * self.w2
    }

    fn forward_feed(&self, b1: f32, w1: &ArrayView1<f32>, w2: f32) -> A {
        // this does not include b2, because it is not group specific
        (self
            .marker_data
            .as_ref()
            .unwrap()
            .right_multiply_par(w1.as_slice().unwrap())
            + b1)
            .mapv(activation_fn)
            * w2
    }

    fn rss(&self, b1: f32, w1: &ArrayView1<f32>, w2: f32) -> f32 {
        let r = &self.residual - self.forward_feed(b1, w1, w2);
        r.dot(&r)
    }

    // logarithm of the parameter density (-U)
    // this has to accept a parameter vector
    fn log_density(&self, param_vec: &A) -> f32 {
        let b1_index = 0;
        let w1_index_first = 1;
        let w1_index_last = self.num_markers;
        let w2_index = w1_index_last + 1;
        let b1 = param_vec[b1_index];
        let w1 = param_vec.slice(s![w1_index_first..=w1_index_last]);
        let w2 = param_vec[w2_index];
        let b1_part = -self.lambda_b1 / 2. * b1 * b1;
        let w1_part = -self.lambda_w1 / 2. * w1.dot(&w1);
        let w2_part = -self.lambda_w2 / 2. * w2 * w2;
        let rss_part = -self.lambda_e / 2. * self.rss(b1, &w1, w2);
        b1_part + w1_part + w2_part + rss_part
    }

    fn log_density_gradient(&self, param_vec: &A) -> A {
        let b1_index = 0;
        let w1_index_first = 1;
        let w1_index_last = self.num_markers;
        let w2_index = w1_index_last + 1;
        let b1 = param_vec[0];
        let w1 = param_vec.slice(s![w1_index_first..=w1_index_last]);
        let w2 = param_vec[w2_index];
        let x_times_w1 = self
            .marker_data
            .as_ref()
            .unwrap()
            .right_multiply_par(w1.as_slice().unwrap());
        let z = &x_times_w1 + b1;
        let a = (x_times_w1 + b1).mapv(activation_fn);
        let y_hat = &a * self.w2;
        let h_prime_of_z = z.mapv(activation_fn_derivative);
        let drss_dyhat = -self.lambda_e * (y_hat - &self.residual);
        let mut gradient: A = Array1::zeros(2 + w1.len());

        gradient[b1_index] = -self.lambda_b1 * b1 + w2 * drss_dyhat.dot(&h_prime_of_z);
        gradient
            .slice_mut(s![w1_index_first..=w1_index_last])
            .assign(
                &(-self.lambda_w1 * &w1
                    + (self
                        .marker_data
                        .as_ref()
                        .unwrap()
                        .left_multiply_simd_v1_par(
                            (&drss_dyhat * w2 * h_prime_of_z).as_slice().unwrap(),
                        ))),
            );
        gradient[w2_index] = -self.lambda_w2 * w2 + drss_dyhat.dot(&a);
        gradient
    }

    pub fn param_vec(&self) -> A {
        let mut p = Vec::with_capacity(self.w1.len() + 2);
        p.push(self.b1);
        p.extend(&self.w1);
        p.push(self.w2);
        arr1(&p)
    }

    // Take single sample using HMC
    // TODO: could to max tries and reduce step size if unsuccessful
    pub fn sample_params(&mut self, step_size: f32, integration_length: usize) -> A {
        let start_position = self.param_vec();
        loop {
            let mut position = start_position.clone();
            let start_momentum: A = self.momentum_sampler.sample();
            let mut momentum = start_momentum.clone();
            momentum.scaled_add(step_size * 0.5, &self.log_density_gradient(&position));
            for _step in 0..integration_length {
                let acc_prob = self.acceptance_probability(
                    &position,
                    &momentum,
                    &start_position,
                    &start_momentum,
                );
                dbg!(&position);
                dbg!(self.log_density(&position));
                dbg!(acc_prob);
                let m_d = self.momentum_sampler.log_density(&momentum);
                dbg!(m_d);
                self.leapfrog(&mut position, &mut momentum, step_size);
            }
            let acc_prob =
                self.acceptance_probability(&position, &momentum, &start_position, &start_momentum);
            if self.accept(acc_prob) {
                return position;
            }
        }
    }

    fn accept(&mut self, acceptance_probability: f32) -> bool {
        self.rng.gen_range(0.0..1.0) < acceptance_probability
    }

    fn acceptance_probability(
        &self,
        new_position: &A,
        new_momentum: &A,
        initial_position: &A,
        initial_momentum: &A,
    ) -> f32 {
        let log_acc_probability = self.neg_hamiltonian(new_position, new_momentum)
            - self.neg_hamiltonian(initial_position, initial_momentum);
        if log_acc_probability >= 0. {
            return 1.;
        }
        log_acc_probability.exp()
    }

    // this is -H = (-U) + (-K)
    fn neg_hamiltonian(&self, position: &A, momentum: &A) -> f32 {
        self.log_density(position) + self.momentum_sampler.log_density(momentum)
    }

    // TODO: make only one gradient computation here,
    // do the half step before and after completing the integration length
    fn leapfrog(&self, position: &mut A, momentum: &mut A, step_size: f32) {
        position.scaled_add(step_size, momentum);
        momentum.scaled_add(step_size, &self.log_density_gradient(position));
    }

    fn gradient_step(&mut self, step_size: f32) {
        let mut param_vec = self.param_vec();
        let gradient = self.log_density_gradient(&param_vec);
        dbg!(&param_vec, &gradient);
        self.set_params(&(param_vec + (gradient * step_size)));
    }
}

// Will have multiple groups
// Each group should be trained independently
// e.g. each group will have it's own sampler
pub struct Net {}

impl Net {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marker_group_param_sampling() {
        let reader = BedReader::new("resources/test/four_by_two.bed", 4, 2);
        let mut mg = MarkerGroup::new(
            arr1(&[-0.587_430_3, 0.020_813_8, 0.346_810_51, 0.283_149_64]),
            arr1(&[1., 1.]),
            1.,
            1.,
            reader,
            2,
        );
        mg.load_marker_data();
        dbg!("starting loop");
        for i in 0..3 {
            dbg!(i);
            let res = mg.sample_params(70., 100);
            mg.set_params(&res);
            dbg!(mg.forward_feed_with_set_params());
        }

        mg.forget_marker_data();
        assert_eq!(mg.param_vec(), arr1(&[0., 0., 0., 0., 0.]));
    }

    #[test]
    fn test_marker_group_gradient() {
        let reader = BedReader::new("resources/test/four_by_two.bed", 4, 2);
        let mut mg = MarkerGroup::new(
            arr1(&[-0.587_430_3, 0.020_813_8, 0.346_810_51, 0.283_149_64]),
            arr1(&[0., 1.]),
            0.,
            1.,
            reader,
            2,
        );
        mg.load_marker_data();
        let density_before_step = mg.log_density(&mg.param_vec());
        mg.gradient_step(0.01);
        let density_after_step = mg.log_density(&mg.param_vec());
        mg.forget_marker_data();
        dbg!(density_before_step, density_after_step);
        assert!(density_before_step < density_after_step);
    }
}
