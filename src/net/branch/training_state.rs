pub struct TrainingState {
    last_rss: Array<f32>,
}

impl TrainingState {
    pub fn last_rss(&self) -> &Array<f32> {
        &self.last_rss
    }

    pub fn set_last_rss(&mut self, new: &Array<f32>) {
        self.last_rss = *new;
    }
}
