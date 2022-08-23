use serde::Serialize;

#[derive(Clone, Serialize)]
pub(crate) struct Trajectory {
    // flattened param vecs over time
    params: Vec<Vec<f64>>,
}

impl Trajectory {
    pub fn new() -> Self {
        Self { params: Vec::new() }
    }

    pub fn add(&mut self, a: Vec<f64>) {
        self.params.push(a);
    }
}
