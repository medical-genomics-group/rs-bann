use clap::{Args, Parser, Subcommand};

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
}

#[derive(Args, Debug)]
pub(crate) struct SimulateArgs {
    /// path to output file(s)
    pub path: String,

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

/// A small bayesian neural network implementation.
/// Number of markers per branch: fixed
/// Depth of branches: same for all branches
/// Width of branch layers: same within branches, dynamic between branches
#[derive(Args, Debug)]
pub(crate) struct BaseModelArgs {
    /// path to train data
    pub train_data: String,

    /// path to test data
    pub test_data: String,

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

    /// enable step sizes scales by prior standard deviation.
    /// Takes precedence of random_step_sizes if enabled.
    #[clap(short, long)]
    pub std_scaled_step_sizes: bool,

    /// enable debug prints
    #[clap(short, long)]
    pub debug_prints: bool,
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
