use serde::Serialize;

#[derive(Clone, Serialize)]
pub(crate) struct Trajectory {
    // flattened param vecs over time
    params: Vec<Vec<f64>>,
    ldg: Vec<Vec<f64>>,
    num_ldg: Vec<Vec<f64>>,
    hamiltonian: Vec<f64>,
}

impl Trajectory {
    pub fn new() -> Self {
        Self {
            params: Vec::new(),
            ldg: Vec::new(),
            hamiltonian: Vec::new(),
            num_ldg: Vec::new(),
        }
    }

    pub fn add_params(&mut self, a: Vec<f64>) {
        self.params.push(a);
    }

    pub fn add_ldg(&mut self, a: Vec<f64>) {
        self.ldg.push(a);
    }

    pub fn add_num_ldg(&mut self, a: Vec<f64>) {
        self.num_ldg.push(a);
    }

    pub fn add_hamiltonian(&mut self, v: f64) {
        self.hamiltonian.push(v);
    }
}
