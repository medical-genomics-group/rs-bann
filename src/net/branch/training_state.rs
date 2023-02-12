use arrayfire::{constant, dim4, Array};
pub struct TrainingState {
    rss: Array<f32>,
    d_rss_wrt_weights: Vec<Array<f32>>,
    d_rss_wrt_biases: Vec<Array<f32>>,
}

impl TrainingState {
    pub fn default() -> Self {
        Self {
            rss: constant(0.0, dim4!(1, 1, 1, 1)),
            d_rss_wrt_weights: vec![constant(0.0, dim4!(1, 1, 1, 1))],
            d_rss_wrt_biases: vec![constant(0.0, dim4!(1, 1, 1, 1))],
        }
    }

    pub fn rss(&self) -> &Array<f32> {
        &self.rss
    }

    pub fn d_rss_wrt_weights(&self) -> &Vec<Array<f32>> {
        &self.d_rss_wrt_weights
    }

    pub fn d_rss_wrt_biases(&self) -> &Vec<Array<f32>> {
        &self.d_rss_wrt_biases
    }

    pub fn set_rss(&mut self, new: &Array<f32>) {
        self.rss = new.clone();
    }

    // TODO: instead of cloning, directly modify the entries of training state on the fly
    pub fn set_d_rss_wrt_weights(&mut self, new: &[Array<f32>]) {
        self.d_rss_wrt_weights = new.to_vec();
    }

    pub fn set_d_rss_wrt_biases(&mut self, new: &[Array<f32>]) {
        self.d_rss_wrt_biases = new.to_vec();
    }
}
