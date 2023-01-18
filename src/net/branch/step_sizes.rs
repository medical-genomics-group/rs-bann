use arrayfire::Array;
use std::fmt;

#[derive(Clone)]
pub struct StepSizes {
    pub wrt_weights: Vec<Array<f32>>,
    pub wrt_biases: Vec<Array<f32>>,
    pub wrt_weight_precisions: Option<Vec<Array<f32>>>,
    pub wrt_bias_precisions: Option<Vec<Array<f32>>>,
    pub wrt_error_precision: Option<Array<f32>>,
}

impl fmt::Debug for StepSizes {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.param_vec())
    }
}

impl StepSizes {
    fn param_vec(&self) -> Vec<f32> {
        let mut host_vec = Vec::new();
        host_vec.resize(self.num_params(), 0.);
        let mut insert_ix: usize = 0;
        for i in 0..self.wrt_weights.len() {
            let len = self.wrt_weights[i].elements();
            self.wrt_weights[i].host(&mut host_vec[insert_ix..insert_ix + len]);
            insert_ix += len;
        }
        for i in 0..self.wrt_biases.len() {
            let len = self.wrt_biases[i].elements();
            self.wrt_biases[i].host(&mut host_vec[insert_ix..insert_ix + len]);
            insert_ix += len;
        }
        host_vec
    }

    fn num_params(&self) -> usize {
        let mut res: usize = 0;
        for i in 0..self.wrt_weights.len() {
            res += self.wrt_weights[i].elements();
        }
        for i in 0..self.wrt_biases.len() {
            res += self.wrt_biases[i].elements();
        }
        res
    }

    // // TODO: better to use some dual averaging scheme?
    // // TODO: test this!
    // // TODO: this will almost definitely violate reversibility and therefore detailed balance.
    // // It might still work, but is not guaranteed to?
    // fn update_with_second_derivative(
    //     &mut self,
    //     prev_momenta: &BranchMomenta,
    //     prev_gradients: &BranchLogDensityGradient,
    //     curr_gradients: &BranchLogDensityGradient,
    // ) {
    //     for i in 0..self.wrt_weights.len() {
    //         // distance traveled
    //         let delta_t = &self.wrt_weights[i] * &prev_momenta.wrt_weights[i];
    //         // change in first derivative value
    //         let delta_f = &curr_gradients.wrt_weights[i] - &prev_gradients.wrt_weights[i];
    //         // TODO: the argument of the sqrt might need a negative sign
    //         self.wrt_weights[i] = 1 / (arrayfire::sqrt(&(delta_f / delta_t)));
    //     }
    //     for i in 0..self.wrt_biases.len() {
    //         // distance traveled
    //         let delta_t = &self.wrt_biases[i] * &prev_momenta.wrt_biases[i];
    //         // change in first derivative value
    //         let delta_f = &curr_gradients.wrt_biases[i] - &prev_gradients.wrt_biases[i];
    //         self.wrt_biases[i] = 1 / (arrayfire::sqrt(&(delta_f / delta_t)));
    //     }
    // }
}
