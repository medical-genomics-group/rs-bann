/// Parameters for MCMC sampling.
pub struct MCMCCfg {
    pub hmc_step_size_factor: f64,
    pub hmc_max_hamiltonian_error: f64,
    pub hmc_integration_length: usize,
    pub hmc_step_size_mode: StepSizeMode,
    pub chain_length: usize,
    pub trace_file: Option<String>,
}

pub enum StepSizeMode {
    Uniform,
    Random,
    StdScaled,
    Izmailov,
}
