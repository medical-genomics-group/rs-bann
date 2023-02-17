use serde::{Deserialize, Serialize};

use std::{
    fmt::{Display, Formatter},
    path::{Path, PathBuf},
};

pub struct MCMCCfgBuilder {
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
    /// use gradient descent on both parameteres and precisions, instead of hmc
    pub gradient_descent_joint: bool,
    /// sample branch parameters and their precisions jointly
    /// instead of sampling the precisions in a gibbs step
    pub joint_hmc: bool,
    pub fixed_precisions: bool,
}

impl MCMCCfgBuilder {
    pub fn default() -> Self {
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
            gradient_descent_joint: false,
            joint_hmc: false,
            fixed_precisions: false,
        }
    }

    pub fn with_hmc_step_size_factor(mut self, arg: f32) -> Self {
        self.hmc_step_size_factor = arg;
        self
    }

    pub fn with_hmc_max_hamiltonian_error(mut self, arg: f32) -> Self {
        self.hmc_max_hamiltonian_error = arg;
        self
    }

    pub fn with_hmc_integration_length(mut self, arg: usize) -> Self {
        self.hmc_integration_length = arg;
        self
    }

    pub fn with_hmc_step_size_mode(mut self, arg: StepSizeMode) -> Self {
        self.hmc_step_size_mode = arg;
        self
    }

    pub fn with_chain_length(mut self, arg: usize) -> Self {
        self.chain_length = arg;
        self
    }

    pub fn with_burn_in(mut self, arg: usize) -> Self {
        self.burn_in = arg;
        self
    }

    pub fn with_outpath(mut self, arg: String) -> Self {
        self.outpath = arg;
        self
    }

    pub fn with_trace(mut self, arg: bool) -> Self {
        self.trace = arg;
        self
    }

    pub fn with_trajectories(mut self, arg: bool) -> Self {
        self.trajectories = arg;
        self
    }

    pub fn with_num_grad_traj(mut self, arg: bool) -> Self {
        self.num_grad_traj = arg;
        self
    }

    pub fn with_num_grad(mut self, arg: bool) -> Self {
        self.num_grad = arg;
        self
    }

    pub fn with_gradient_descent(mut self, arg: bool) -> Self {
        self.gradient_descent = arg;
        self
    }

    pub fn with_gradient_descent_joint(mut self, arg: bool) -> Self {
        self.gradient_descent_joint = arg;
        self
    }

    pub fn with_joint_hmc(mut self, arg: bool) -> Self {
        self.joint_hmc = arg;
        self
    }

    pub fn with_fixed_precisions(mut self, arg: bool) -> Self {
        self.fixed_precisions = arg;
        self
    }

    pub fn build(&self) -> MCMCCfg {
        if self.fixed_precisions && (self.joint_hmc || self.gradient_descent_joint) {
            panic!("Fixed precisions and joint hmc / gd are mutually exclusive");
        }
        MCMCCfg {
            hmc_step_size_factor: self.hmc_step_size_factor,
            hmc_max_hamiltonian_error: self.hmc_max_hamiltonian_error,
            hmc_integration_length: self.hmc_integration_length,
            hmc_step_size_mode: self.hmc_step_size_mode.clone(),
            chain_length: self.chain_length,
            burn_in: self.burn_in,
            outpath: self.outpath.clone(),
            trace: self.trace,
            trajectories: self.trajectories,
            num_grad_traj: self.num_grad_traj,
            num_grad: self.num_grad,
            gradient_descent: self.gradient_descent,
            gradient_descent_joint: self.gradient_descent_joint,
            joint_hmc: self.joint_hmc,
            fixed_precisions: self.fixed_precisions,
        }
    }
}

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
    /// use gradient descent on both parameteres and precisions, instead of hmc
    pub gradient_descent_joint: bool,
    /// sample branch parameters and their precisions jointly
    /// instead of sampling the precisions in a gibbs step
    pub joint_hmc: bool,
    pub fixed_precisions: bool,
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
            gradient_descent_joint: false,
            joint_hmc: false,
            fixed_precisions: false,
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
