use clap::{Args, Parser, Subcommand};
use log::info;
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

#[derive(Subcommand)]
pub(crate) enum SubCmd {
    /// Simulate marker and phenotype data
    Simulate(SimulateArgs),
    /// Run BaseModel
    BaseModel(BaseModelArgs),
    /// Run StdNormalModel
    StdNormalModel(StdNormalModelArgs),
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

/// A small bayesian neural network implementation.
/// Number of markers per branch: fixed
/// Depth of branches: same for all branches
/// Width of branch layers: same within branches, dynamic between branches
#[derive(Args, Debug)]
pub(crate) struct BaseModelArgs {
    /// input directory with train.bin and test.bin files
    pub indir: String,

    /// width of hidden layer
    pub hidden_layer_width: usize,

    /// number of hidden layers in branches
    pub branch_depth: usize,

    /// prior shape
    pub prior_shape: f64,

    /// prior scale
    pub prior_scale: f64,

    /// full model chain length
    pub chain_length: usize,

    /// hmc max hamiltonian error
    pub max_hamiltonian_error: f64,

    /// hmc integration length
    pub integration_length: usize,

    /// hmc step size, acts as a modifying factor on random step sizes if enabled
    pub step_size: f64,

    /// training stats report interval
    pub report_interval: usize,

    /// enable random step sizes
    #[clap(short, long)]
    pub random_step_sizes: bool,

    /// Set step sizes to pi / (2 * prior_std_deviation * integration_length).
    /// Takes precedence over other step size flags.
    #[clap(short, long)]
    pub izmailov_step_sizes: bool,

    /// enable step sizes scales by prior standard deviation.
    /// Takes precedence of random_step_sizes if enabled.
    #[clap(short, long)]
    pub precision_scaled_step_sizes: bool,

    /// enable debug prints
    #[clap(short, long)]
    pub debug_prints: bool,

    /// standardize input data
    #[clap(short, long)]
    pub standardize: bool,

    /// Path to trace file.
    /// Trace output will only be generated if this path is specified.
    #[clap(short, long)]
    pub trace_file_path: Option<String>,

    /// Path to micro trace file.
    /// Complete HMC trajectories (all states visited during a single update) will be saved here.
    /// Micro trace output will only be generated if this path is specified.
    #[clap(short, long)]
    pub micro_trace_file_path: Option<String>,
}

/// A small bayesian neural network implementation.
/// Number of markers per branch: fixed
/// Depth of branches: same for all branches
/// Width of branch layers: same within branches, dynamic between branches
/// Weight and bias priors are std normals.
#[derive(Args, Debug)]
pub(crate) struct StdNormalModelArgs {
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

    /// training stats report interval
    pub report_interval: usize,

    /// enable random step sizes
    #[clap(short, long)]
    pub random_step_sizes: bool,

    /// Set step sizes to pi / (2 * prior_std_deviation * integration_length).
    /// Takes precedence over other step size flags.
    #[clap(short, long)]
    pub izmailov_step_sizes: bool,

    /// enable step sizes scales by prior standard deviation.
    /// Takes precedence of random_step_sizes if enabled.
    #[clap(short, long)]
    pub precision_scaled_step_sizes: bool,

    /// enable debug prints
    #[clap(short, long)]
    pub debug_prints: bool,

    /// standardize input data
    #[clap(short, long)]
    pub standardize: bool,

    /// Path to trace file.
    /// Trace output will only be generated if this path is specified.
    #[clap(short, long)]
    pub trace_file_path: Option<String>,

    /// Path to micro trace file.
    /// Complete HMC trajectories (all states visited during a single update) will be saved here.
    /// Micro trace output will only be generated if this path is specified.
    #[clap(short, long)]
    pub micro_trace_file_path: Option<String>,
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
