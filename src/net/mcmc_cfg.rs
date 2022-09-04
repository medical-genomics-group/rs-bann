use std::{
    fmt,
    path::{Path, PathBuf},
    str::FromStr,
};

/// Parameters for MCMC sampling.
pub struct MCMCCfg {
    pub hmc_step_size_factor: f64,
    pub hmc_max_hamiltonian_error: f64,
    pub hmc_integration_length: usize,
    pub hmc_step_size_mode: StepSizeMode,
    pub chain_length: usize,
    pub outpath: String,
    pub trace: bool,
    pub trajectories: bool,
    pub num_grad_traj: bool,
}

impl MCMCCfg {
    pub fn create_out(&self) {
        if !Path::new(&self.outpath).exists() {
            std::fs::create_dir_all(&self.outpath).expect("Could not create output directory!");
        }
    }

    pub fn meta_path(&self) -> PathBuf {
        Path::new(&self.outpath).join("meta")
    }

    pub fn trace_path(&self) -> PathBuf {
        Path::new(&self.outpath).join("trace")
    }

    pub fn trajectories_path(&self) -> PathBuf {
        Path::new(&self.outpath).join("traj")
    }
}

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum StepSizeMode {
    Uniform,
    Random,
    StdScaled,
    Izmailov,
}
