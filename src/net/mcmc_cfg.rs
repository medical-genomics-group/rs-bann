use serde::Serialize;

use std::{
    fmt::{Display, Formatter},
    path::{Path, PathBuf},
};

/// Parameters for MCMC sampling.
pub struct MCMCCfg {
    pub hmc_step_size_factor: f32,
    pub hmc_max_hamiltonian_error: f32,
    pub hmc_integration_length: usize,
    pub hmc_step_size_mode: StepSizeMode,
    pub chain_length: usize,
    pub outpath: String,
    pub trace: bool,
    pub trajectories: bool,
    pub num_grad_traj: bool,
    pub num_grad: bool,
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

    pub fn args_path(&self) -> PathBuf {
        Path::new(&self.outpath).join("args.json")
    }
}

#[derive(clap::ValueEnum, Clone, Debug, Serialize)]
pub enum StepSizeMode {
    Uniform,
    Random,
    StdScaled,
    Izmailov,
}

impl Display for StepSizeMode {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
        // or, alternatively:
        // fmt::Debug::fmt(self, f)
    }
}
