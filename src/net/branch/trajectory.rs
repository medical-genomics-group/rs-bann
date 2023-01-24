use serde::Serialize;

#[derive(Clone, Serialize)]
pub(crate) struct Trajectory {
    // flattened param vecs over time
    params: Vec<Vec<f32>>,
    precisions: Vec<Vec<f32>>,
    ldg: Vec<Vec<f32>>,
    num_ldg: Vec<Vec<f32>>,
    hamiltonian: Vec<f32>,
}

impl Trajectory {
    pub fn new() -> Self {
        Self {
            params: Vec::new(),
            precisions: Vec::new(),
            ldg: Vec::new(),
            hamiltonian: Vec::new(),
            num_ldg: Vec::new(),
        }
    }

    pub fn add_precisions(&mut self, a: Vec<f32>) {
        self.precisions.push(a);
    }

    pub fn add_params(&mut self, a: Vec<f32>) {
        self.params.push(a);
    }

    pub fn add_ldg(&mut self, a: Vec<f32>) {
        self.ldg.push(a);
    }

    pub fn add_num_ldg(&mut self, a: Vec<f32>) {
        self.num_ldg.push(a);
    }

    pub fn add_hamiltonian(&mut self, v: f32) {
        self.hamiltonian.push(v);
    }
}
