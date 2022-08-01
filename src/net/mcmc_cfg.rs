/// Parameters for MCMC sampling.
pub struct MCMCCfg {
    pub hmc_step_size: Option<f64>,
    pub hmc_max_hamiltonian_error: f64,
    pub hmc_integration_length: usize,
    pub chain_length: usize,
}
