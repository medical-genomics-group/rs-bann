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

impl fmt::Display for StepSizeMode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            StepSizeMode::Uniform => "StepSizeMode::Uniform",
            StepSizeMode::Random => "StepSizeMode::Random",
            StepSizeMode::StdScaled => "StepSizeMode::StdScaled",
            StepSizeMode::Izmailov => "StepSizeMode::Izmailov",
        };
        write!(f, "{}", s)
    }
}

impl FromStr for StepSizeMode {
    type Err = StepSizeModeFromStrError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "StepSizeMode::Uniform" => Ok(StepSizeMode::Uniform),
            "StepSizeMode::Random" => Ok(StepSizeMode::Random),
            "StepSizeMode::StdScaled" => Ok(StepSizeMode::StdScaled),
            "StepSizeMode::Izmailov" => Ok(StepSizeMode::Izmailov),
            _ => Err(StepSizeModeFromStrError),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StepSizeModeFromStrError;

impl fmt::Display for StepSizeModeFromStrError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "failed to construct StepSizeMode from string.")
    }
}

impl std::error::Error for StepSizeModeFromStrError {}
