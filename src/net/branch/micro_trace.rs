use serde::Serialize;

#[derive(Clone, Serialize)]
pub(crate) struct MicroTrace {
    // flattened param vecs over time
    params: Vec<Vec<f64>>,
}

impl MicroTrace {
    pub fn new() -> Self {
        MicroTrace { params: Vec::new() }
    }

    pub fn add(&mut self, a: Vec<f64>) {
        self.params.push(a);
    }
}
