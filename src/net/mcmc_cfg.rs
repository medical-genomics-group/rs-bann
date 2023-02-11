use serde::{Deserialize, Serialize};

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
    pub burn_in: usize,
    pub outpath: String,
    pub trace: bool,
    pub trajectories: bool,
    pub num_grad_traj: bool,
    pub num_grad: bool,
    /// use gradient descent instead of hmc
    pub gradient_descent: bool,
    /// sample branch parameters and their precisions jointly
    /// instead of sampling the precisions in a gibbs step
    pub joint_hmc: bool,
}

impl Default for MCMCCfg {
    fn default() -> Self {
        Self {
            hmc_step_size_factor: 1.0,
            hmc_max_hamiltonian_error: 10.0,
            hmc_integration_length: 100,
            hmc_step_size_mode: StepSizeMode::Izmailov,
            chain_length: 100,
            burn_in: 0,
            outpath: "./".to_string(),
            trace: false,
            trajectories: false,
            num_grad_traj: false,
            num_grad: false,
            // use gradient descent instead of hmc
            gradient_descent: false,
            joint_hmc: false,
        }
    }
}

impl MCMCCfg {
    pub fn create_out(&self) {
        if !Path::new(&self.outpath).exists() {
            std::fs::create_dir_all(&self.outpath).expect("Could not create output directory!");
        }
    }

    pub fn hyperparam_path(&self) -> PathBuf {
        Path::new(&self.outpath).join("hyperparams")
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

    pub fn models_path(&self) -> PathBuf {
        Path::new(&self.outpath).join("models")
    }

    pub fn effect_sizes_path(&self) -> PathBuf {
        Path::new(&self.outpath).join("effect_sizes")
    }
}

#[derive(clap::ValueEnum, Clone, Debug, Serialize, Deserialize)]
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
