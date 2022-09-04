use clap::{Args, Parser, Subcommand};
use log::info;
use rs_bann::net::mcmc_cfg::StepSizeMode;
use serde::{Deserialize, Serialize};
use serde_json::to_writer_pretty;
use std::{fs::File, path::Path};

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
#[clap(propagate_version = true)]
pub(crate) struct Cli {
    #[clap(subcommand)]
    pub(crate) cmd: SubCmd,
}

#[derive(clap::ValueEnum, Clone, Debug)]
pub(crate) enum ModelType {
    ARD,
    Base,
    StdNormal,
}

#[derive(Subcommand)]
pub(crate) enum SubCmd {
    /// Simulate marker and phenotype data
    Simulate(SimulateArgs),
    /// Train Model
    Train(TrainArgs),
}

#[derive(Args, Debug, Serialize, Deserialize)]
pub(crate) struct SimulateArgs {
    /// path to output dir. Will be created if it does not exist
    pub outdir: String,

    /// number of input features per branch (markers)
    pub num_markers_per_branch: usize,

    /// number of branches (markers)
    pub num_branches: usize,

    /// number of samples (individuals)
    pub num_individuals: usize,

    /// width of hidden layer
    pub hidden_layer_width: usize,

    /// number of hidden layers in branches
    pub branch_depth: usize,

    /// heritability (determines amount of Gaussian noise added), must be in [0, 1]
    pub heritability: Option<f64>,
}

impl SimulateArgs {
    pub fn to_file(&self, path: &Path) {
        info!("Creating: {:?}", path);
        to_writer_pretty(File::create(path).unwrap(), self).unwrap();
    }
}

#[derive(Args, Debug)]
pub(crate) struct TrainArgs {
    /// Prior structure of model.
    #[clap(value_enum)]
    pub model_type: ModelType,

    /// input directory with train.bin and test.bin files
    pub indir: String,

    /// width of hidden layer
    pub hidden_layer_width: usize,

    /// number of hidden layers in branches
    pub branch_depth: usize,

    /// full model chain length
    pub chain_length: usize,

    /// hmc max hamiltonian error
    pub max_hamiltonian_error: f64,

    /// hmc integration length
    pub integration_length: usize,

    /// hmc step size, acts as a modifying factor on random step sizes if enabled
    pub step_size: f64,

    #[clap(default_value_t = 1)]
    /// training stats report interval
    pub report_interval: usize,

    #[clap(default_value_t = 1.)]
    /// prior shape
    pub prior_shape: f64,

    #[clap(default_value_t = 1.)]
    /// prior scale
    pub prior_scale: f64,

    /// Output path
    #[clap(short, long)]
    pub outpath: String,

    ///  Step size mode
    #[clap(value_enum, default_value_t = StepSizeMode::Uniform)]
    pub step_size_mode: StepSizeMode,

    /// enable debug prints
    #[clap(short, long)]
    pub debug_prints: bool,

    /// standardize input data
    #[clap(short, long)]
    pub standardize: bool,

    /// Output trace
    #[clap(long)]
    pub trace: bool,

    /// Output hmc trajectories
    #[clap(long)]
    pub trajectories: bool,

    /// Output numerical gradients
    /// CAUTION: this is extremely expensive, do not run this in production.
    #[clap(long)]
    pub num_grad_traj: bool,
}

/// A small bayesian neural network implementation based on ArrayFire.
#[derive(Args, Debug)]
pub(crate) struct AFArgs {
    /// number of input feature (markers)
    pub num_markers: usize,

    /// number of samples (individuals)
    pub num_individuals: usize,

    /// width of hidden layer
    pub hidden_layer_width: usize,

    /// hmc integration length
    pub integration_length: usize,

    /// chain length (number of hmc samples)
    pub chain_length: usize,

    /// max hamiltonian error
    pub max_hamiltonian_error: f64,

    /// hmc step size, acts as a modifying factor on random or scaled step size if enables
    pub step_size: f64,

    /// enable random step sizes
    #[clap(short, long)]
    pub random_step_sizes: bool,

    /// enable step sizes scales by prior standard deviation.
    /// Takes precedence over random_step_sizes if enabled.
    #[clap(short, long)]
    pub std_scaled_step_sizes: bool,

    /// enable debug prints
    #[clap(short, long)]
    pub debug_prints: bool,
}
