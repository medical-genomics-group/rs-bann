use arrayfire::{dim4, Array};

pub trait NetParamGradient {
    fn wrt_weights(&self) -> &Vec<Array<f32>>;
    fn wrt_biases(&self) -> &Vec<Array<f32>>;
}

pub trait NetPrecisionGradient {
    fn wrt_weight_precisions(&self) -> &Vec<Array<f32>>;
    fn wrt_bias_precisions(&self) -> &Vec<Array<f32>>;
    fn wrt_error_precision(&self) -> &Array<f32>;
}

/// Gradients of the log density w.r.t. the network parameters and precisions
#[derive(Clone)]
pub struct BranchLogDensityGradientJoint {
    pub wrt_weights: Vec<Array<f32>>,
    pub wrt_biases: Vec<Array<f32>>,
    pub wrt_weight_precisions: Vec<Array<f32>>,
    pub wrt_bias_precisions: Vec<Array<f32>>,
    pub wrt_error_precision: Array<f32>,
}

impl NetParamGradient for BranchLogDensityGradientJoint {
    fn wrt_biases(&self) -> &Vec<Array<f32>> {
        &self.wrt_biases
    }

    fn wrt_weights(&self) -> &Vec<Array<f32>> {
        &self.wrt_weights
    }
}

impl NetPrecisionGradient for BranchLogDensityGradientJoint {
    fn wrt_bias_precisions(&self) -> &Vec<Array<f32>> {
        &self.wrt_bias_precisions
    }

    fn wrt_weight_precisions(&self) -> &Vec<Array<f32>> {
        &self.wrt_weight_precisions
    }

    fn wrt_error_precision(&self) -> &Array<f32> {
        &self.wrt_error_precision
    }
}

impl BranchLogDensityGradientJoint {
    fn num_params(&self) -> usize {
        let mut res: usize = 1;
        for i in 0..self.wrt_weights.len() {
            res += self.wrt_weights[i].elements();
        }
        for i in 0..self.wrt_biases.len() {
            res += self.wrt_biases[i].elements();
        }
        for i in 0..self.wrt_weight_precisions.len() {
            res += self.wrt_weight_precisions[i].elements();
        }
        for i in 0..self.wrt_bias_precisions.len() {
            res += self.wrt_bias_precisions[i].elements();
        }
        res
    }

    pub(crate) fn param_vec(&self) -> Vec<f32> {
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
        for i in 0..self.wrt_weight_precisions.len() {
            let len = self.wrt_weight_precisions[i].elements();
            self.wrt_weight_precisions[i].host(&mut host_vec[insert_ix..insert_ix + len]);
            insert_ix += len;
        }
        for i in 0..self.wrt_bias_precisions.len() {
            let len = self.wrt_bias_precisions[i].elements();
            self.wrt_bias_precisions[i].host(&mut host_vec[insert_ix..insert_ix + len]);
            insert_ix += len;
        }
        self.wrt_error_precision
            .host(&mut host_vec[insert_ix..insert_ix + 1]);
        host_vec
    }
}

/// Gradients of the log density w.r.t. the network parameters.
#[derive(Clone)]
pub struct BranchLogDensityGradient {
    pub wrt_weights: Vec<Array<f32>>,
    pub wrt_biases: Vec<Array<f32>>,
}

impl NetParamGradient for BranchLogDensityGradient {
    fn wrt_biases(&self) -> &Vec<Array<f32>> {
        &self.wrt_biases
    }

    fn wrt_weights(&self) -> &Vec<Array<f32>> {
        &self.wrt_weights
    }
}

impl BranchLogDensityGradient {
    pub fn from_param_vec(
        param_vec: &[f32],
        layer_widths: &Vec<usize>,
        num_markers: usize,
    ) -> Self {
        let mut wrt_weights: Vec<Array<f32>> = vec![];
        let mut wrt_biases: Vec<Array<f32>> = vec![];
        let mut prev_width = num_markers;
        let mut read_ix: usize = 0;
        for width in layer_widths {
            let num_weights = prev_width * width;
            wrt_weights.push(Array::new(
                &param_vec[read_ix..read_ix + num_weights],
                dim4!(prev_width as u64, *width as u64, 1, 1),
            ));
            prev_width = *width;
            read_ix += num_weights;
        }
        for width in &layer_widths[..layer_widths.len() - 1] {
            let num_biases = width;
            wrt_biases.push(Array::new(
                &param_vec[read_ix..read_ix + num_biases],
                dim4!(1, *width as u64, 1, 1),
            ));
            read_ix += num_biases;
        }
        Self {
            wrt_weights,
            wrt_biases,
        }
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

    pub(crate) fn param_vec(&self) -> Vec<f32> {
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
}
