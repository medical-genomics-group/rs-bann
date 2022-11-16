use clap::{Args, Parser, Subcommand};
use log::info;
use rs_bann::net::{mcmc_cfg::StepSizeMode, net::ModelType};
use serde::{Deserialize, Serialize};
use serde_json::to_writer_pretty;
use std::{fs::File, io::BufReader, path::Path};

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
#[clap(propagate_version = true)]
pub(crate) struct Cli {
    #[clap(subcommand)]
    pub(crate) cmd: SubCmd,
}

#[derive(Subcommand)]
pub(crate) enum SubCmd {
    /// Group markers by LD.
    GroupCentered(GroupCenteredArgs),
    /// Simulate phenotype data given marker data.
    ///
    /// Branch width is fixed to 1/2 the number of input nodes in a branch.
    SimulateY(SimulateYArgs),
    /// Simulate marker and phenotype data under a network model.
    SimulateXY(SimulateXYArgs),
    /// Train new model.
    TrainNew(TrainNewArgs),
    /// Train prespecified model.
    Train(TrainArgs),
    /// Use trained model to predict phenotypes.
    Predict(PredictArgs),
    /// Use trained model to compute r2 values for each model branch.
    BranchR2(BranchR2Args),
}

#[derive(Args, Debug, Serialize, Deserialize)]
pub(crate) struct GroupCenteredArgs {
    /// path to input (just the file stem without .bim and .ld suffixes)
    pub inpath: String,

    /// path to output directory
    #[clap(short, long, default_value = "./")]
    pub outdir: String,
}

#[derive(Args, Debug, Serialize, Deserialize)]
pub(crate) struct SimulateYArgs {
    /// path to output dir. Dir with the simulated data will be created there.
    #[clap(short, long, default_value = "./")]
    pub outdir: String,

    /// path to train .bed file
    pub train_bed: String,

    /// path to test .bed file
    pub test_bed: String,

    /// path to file with marker groupings. Should have two columns: marker_index, group_index
    pub groups: String,

    /// Prior structure of model.
    #[clap(value_enum)]
    pub model_type: ModelType,

    /// number of hidden layers in branches
    pub branch_depth: usize,

    /// variance of network params upon initialization
    #[clap(long, default_value_t = 1.0)]
    pub init_param_variance: f32,

    /// shape of gamma prior for network param initialization
    #[clap(long)]
    pub init_gamma_shape: Option<f32>,

    /// scale of gamma prior for network param initialization
    #[clap(long)]
    pub init_gamma_scale: Option<f32>,

    /// heritability (determines amount of Gaussian noise added), must be in [0, 1]
    #[clap(default_value_t = 1.0)]
    pub heritability: f32,

    /// write data to json files, e.g. for easier parsing into python
    #[clap(long)]
    pub json_data: bool,
}

impl SimulateYArgs {
    pub fn to_file(&self, path: &Path) {
        info!("Creating: {:?}", path);
        to_writer_pretty(File::create(path).unwrap(), self).unwrap();
    }
}

#[derive(Args, Debug, Serialize, Deserialize)]
pub(crate) struct SimulateXYArgs {
    /// path to output dir. Dir with the simulated data will be created there.
    #[clap(short, long, default_value = "./")]
    pub outdir: String,

    /// Prior structure of model.
    #[clap(value_enum)]
    pub model_type: ModelType,

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

    /// variance of network params upon initialization
    #[clap(long, default_value_t = 1.0)]
    pub init_param_variance: f32,

    /// shape of gamma prior for network param initialization
    #[clap(long)]
    pub init_gamma_shape: Option<f32>,

    /// scale of gamma prior for network param initialization
    #[clap(long)]
    pub init_gamma_scale: Option<f32>,

    /// heritability (determines amount of Gaussian noise added), must be in [0, 1]
    #[clap(default_value_t = 1.0)]
    pub heritability: f32,

    /// write data to json files, e.g. for easier parsing into python
    #[clap(long)]
    pub json_data: bool,
}

impl SimulateXYArgs {
    pub fn to_file(&self, path: &Path) {
        info!("Creating: {:?}", path);
        to_writer_pretty(File::create(path).unwrap(), self).unwrap();
    }
}

#[derive(Args, Debug, Serialize)]
pub(crate) struct TrainArgs {
    /// Prior structure of model.
    #[clap(value_enum)]
    pub model_type: ModelType,

    /// model file
    pub model_file: String,

    /// input directory with train.gen, train.phen, and optionally test.gen, test.phen
    #[clap(short, long, default_value = "./")]
    pub indir: String,

    /// full model chain length
    pub chain_length: usize,

    /// hmc max hamiltonian error
    #[clap(default_value_t = 10., long)]
    pub max_hamiltonian_error: f32,

    /// hmc integration length
    pub integration_length: usize,

    /// hmc step size, acts as a modifying factor on random step sizes if enabled
    #[clap(default_value_t = 0.1, long)]
    pub step_size: f32,

    #[clap(default_value_t = 1, long)]
    /// training stats report interval
    pub report_interval: usize,

    #[clap(short, long, default_value = "./")]
    /// Output path. Outdir will be created there.
    pub outpath: String,

    ///  Step size mode
    #[clap(value_enum, default_value_t = StepSizeMode::Izmailov, long)]
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

    /// Use numerical gradients instead of analytical for integration.
    /// CAUTION: this is extremely expensive, do not run this in production.
    #[clap(long)]
    pub num_grad: bool,

    /// Use gradient descent instead of HMC.
    /// CAUTION: this does only lead to point estimates for all parameters
    #[clap(long)]
    pub gradient_descent: bool,

    /// Set error precision of model before training.
    #[clap(long)]
    pub error_precision: Option<f32>,

    /// Number of burn-in samples that will be discarded.
    #[clap(default_value_t = 0, long)]
    pub burn_in: usize,
}

impl TrainArgs {
    pub fn to_file(&self, path: &Path) {
        info!("Creating: {:?}", path);
        to_writer_pretty(File::create(path).unwrap(), self).unwrap();
    }
}

#[derive(Args, Debug, Serialize, Deserialize)]
pub(crate) struct TrainNewArgs {
    /// Prior structure of model.
    #[clap(value_enum)]
    pub model_type: ModelType,

    /// input directory with train.bin and test.bin files
    #[clap(short, long, default_value = "./")]
    pub indir: String,

    /// width of hidden layer
    pub hidden_layer_width: usize,

    /// number of hidden layers in branches
    pub branch_depth: usize,

    /// full model chain length
    pub chain_length: usize,

    /// hmc max hamiltonian error
    #[clap(default_value_t = 10., long)]
    pub max_hamiltonian_error: f32,

    /// hmc integration length
    pub integration_length: usize,

    /// hmc step size, acts as a modifying factor on random step sizes if enabled
    #[clap(default_value_t = 0.1, long)]
    pub step_size: f32,

    #[clap(default_value_t = 1, long)]
    /// training stats report interval
    pub report_interval: usize,

    #[clap(default_value_t = 1., long)]
    /// shape hyperparam of prior distribution of precision of dense layer params
    pub dpk: f32,

    #[clap(default_value_t = 1., long)]
    /// scale hyperparam of prior distribution of precision of dense layer params
    pub dps: f32,

    #[clap(default_value_t = 1., long)]
    /// shape hyperparam of prior distribution of precision of summary layer params
    pub spk: f32,

    #[clap(default_value_t = 1., long)]
    /// scale hyperparam of prior distribution of precision of summary layer params
    pub sps: f32,

    #[clap(default_value_t = 1., long)]
    /// shape hyperparam of prior distribution of precision of ouput layer params
    pub opk: f32,

    #[clap(default_value_t = 1., long)]
    /// scale hyperparam of prior distribution of precision of output layer params
    pub ops: f32,

    #[clap(short, long, default_value = "./")]
    /// Output path. Outdir will be created there.
    pub outpath: String,

    ///  Step size mode
    #[clap(value_enum, default_value_t = StepSizeMode::Izmailov, long)]
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

    /// Output numerical gradients.
    /// CAUTION: this is extremely expensive, do not run this in production.
    #[clap(long)]
    pub num_grad_traj: bool,

    /// Use numerical gradients instead of analytical for integration.
    /// CAUTION: this is extremely expensive, do not run this in production.
    #[clap(long)]
    pub num_grad: bool,

    /// Use gradient descent instead of HMC.
    /// CAUTION: this does only lead to point estimates for all parameters
    #[clap(long)]
    pub gradient_descent: bool,

    /// Set error precision of model before training.
    #[clap(long)]
    pub error_precision: Option<f32>,

    /// Number of burn-in samples that will be discarded.
    #[clap(default_value_t = 0, long)]
    pub burn_in: usize,
}

impl TrainNewArgs {
    pub fn from_file(path: &Path) -> Self {
        let file = File::open(path).expect("Failed to open args.json");
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).unwrap()
    }

    pub fn to_file(&self, path: &Path) {
        info!("Creating: {:?}", path);
        to_writer_pretty(File::create(path).unwrap(), self).unwrap();
    }
}

/// Use trained model to predict phenotype values.
/// This returns one prediction for each sampled model
/// stored in the .models dir generated in a `rs-bann train-new` run.
/// The results are sent to stdout in csv format, where each row holds
/// the predictions generated with one sampled model for all input samples.
#[derive(Args, Debug, Serialize)]
pub(crate) struct PredictArgs {
    // TODO: this should accept a bed file.
    /// Path to input data file.
    /// Should contain a rs-bann Genotypes instance.
    pub input_data: String,

    /// Path to models generated with `train-new` or `train` command
    #[clap(short, long, default_value = "./models")]
    pub model_path: String,

    /// standardize input data
    #[clap(short, long)]
    pub standardize: bool,
}

// impl PredictArgs {
//     pub fn to_file(&self, path: &Path) {
//         info!("Creating: {:?}", path);
//         to_writer_pretty(File::create(path).unwrap(), self).unwrap();
//     }
// }

/// Use trained model to compute r2 values for each model branch separately.
/// This returns one r2 for each branch and sampled model in a .models dir
/// generated in a `rs-bann train-new` run.
#[derive(Args, Debug, Serialize)]
pub(crate) struct BranchR2Args {
    // TODO: this should accept a bed file.
    /// Path to input .gen file.
    /// Should contain a rs-bann Genotypes instance.
    pub gen: String,

    /// Path to input .phen file.
    /// Should contain a rs-bann Phenotypes instance.
    pub phen: String,

    /// Path to models generated with `train-new` or `train` command
    #[clap(short, long, default_value = "./models")]
    pub model_path: String,

    /// standardize input data
    #[clap(short, long)]
    pub standardize: bool,
}
