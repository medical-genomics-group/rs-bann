/// Parameters for MCMC sampling.
pub struct MCMCCfg {
    pub hmc_step_size: f64,
    pub hmc_max_hamiltonian_error: f64,
    pub hmc_integration_length: usize,
    pub hmc_random_step_sizes: bool,
    pub chain_length: usize,
}
