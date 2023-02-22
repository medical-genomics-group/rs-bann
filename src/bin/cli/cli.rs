use clap::*;
use log::info;
use rs_bann::net::{mcmc_cfg::StepSizeMode, model_type::ModelType};
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
    /// Group markers by genes.
    GroupByGenes(GroupByGenesArgs),
    /// Group markers by LD.
    GroupByLD(GroupCenteredArgs),
    /// Simulate phenotype data given marker data.
    ///
    /// Branch width is fixed to 1/2 the number of input nodes in a branch.
    SimulateY(SimulateYArgs),
    /// Simulate marker and phenotype data under a network model.
    SimulateXY(SimulateXYArgs),
    /// Train new model on data in .bed format
    TrainNew {
        #[clap(flatten)]
        input_args: TrainIOArgs,
        #[clap(flatten)]
        mcmc_args: MCMCArgs,
        #[clap(flatten)]
        model_args: TrainNewModelArgs,
    },
    /// Train prespecified model.
    Train {
        #[clap(flatten)]
        input_args: TrainIOArgs,
        #[clap(flatten)]
        mcmc_args: MCMCArgs,
        #[clap(flatten)]
        model_args: TrainOldModelArgs,
    },
    /// Use trained model to predict phenotypes.
    Predict(PredictArgs),
    /// Use trained model to compute r2 values for each model branch.
    BranchR2(BranchR2Args),
    /// Report node activations in trained model.
    Activations(ActivationArgs),
    /// Print backends available to ArrayFire.
    AvailableBackends,
}

#[derive(Args, Debug, Serialize, Deserialize)]
pub(crate) struct TrainIOArgs {
    /// dir + filestem of train data .bed, .bim, .fam files (the input genotypes)
    pub bfile_train: String,

    /// filepath of training data phenotypes
    pub p_train: String,

    /// dir + filestem of test data .bed, .bim, .fam files (the input genotypes)
    #[clap(long)]
    pub bfile_test: Option<String>,

    /// filepath of test data phenotypes
    #[clap(long)]
    pub p_test: Option<String>,

    /// path to grouping file
    pub groups: String,

    /// Output path. Outdir will be created there.
    #[clap(short, long, default_value = "./")]
    pub outpath: String,
}

#[derive(Args, Debug, Serialize, Deserialize)]
pub(crate) struct MCMCArgs {
    /// full model chain length
    pub chain_length: usize,

    /// hmc integration length
    pub integration_length: usize,

    /// hmc max hamiltonian error
    #[clap(default_value_t = 10., long)]
    pub max_hamiltonian_error: f32,

    /// hmc step size, acts as a modifying factor on random step sizes if enabled
    #[clap(default_value_t = 0.1, long)]
    pub step_size: f32,

    #[clap(default_value_t = 1, long)]
    /// training stats report interval
    pub report_interval: usize,

    /// fixed all prior precisions to the given value.
    #[clap(long)]
    pub fixed_param_precision: Option<f32>,

    ///  Step size mode
    #[clap(value_enum, default_value_t = StepSizeMode::Izmailov, long)]
    pub step_size_mode: StepSizeMode,

    /// enable debug prints
    #[clap(short, long)]
    pub debug_prints: bool,

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
    /// CAUTION: this only leads to point estimates for all parameters
    #[clap(long)]
    pub gradient_descent: bool,

    /// Use gradient descent to optimize both params and precisions, instead of HMC.
    /// CAUTION: this only leads to point estimates for all parameters
    #[clap(long)]
    pub gradient_descent_joint: bool,

    /// Number of burn-in samples that will be discarded.
    #[clap(default_value_t = 0, long)]
    pub burn_in: usize,

    /// Sample parameters and their precisions jointly, instead sampling the precisions in a Gibbs step.
    #[clap(short, long)]
    pub joint_hmc: bool,
}

#[derive(Args, Debug, Serialize, Deserialize)]
pub(crate) struct GroupByGenesArgs {
    /// path to input .bim file
    pub bim: String,

    /// path to intput .gff file
    pub gff: String,

    /// Distance defining the window from start and end of each gene in which SNPs will be grouped
    pub margin: usize,

    /// minimum group size
    #[clap(long, default_value_t = 1)]
    pub min_group_size: usize,

    /// path to output directory
    #[clap(short, long, default_value = "./")]
    pub outdir: String,
}

#[derive(Args, Debug, Serialize, Deserialize)]
pub(crate) struct GroupCenteredArgs {
    /// path to input (just the file stem without .bim and .ld suffixes)
    pub inpath: String,

    /// minimum group size
    #[clap(long, default_value_t = 1)]
    pub min_group_size: usize,

    /// path to output directory
    #[clap(short, long, default_value = "./")]
    pub outdir: String,
}

#[derive(Args, Debug, Serialize, Deserialize)]
pub(crate) struct SimulateYArgs {
    /// dir + filestem of train data .bed, .bim, .fam files (the input genotypes)
    pub bfile_train: String,

    /// dir + filestem of test data .bed, .bim, .fam files (the input genotypes)
    pub bfile_test: String,

    /// path to grouping file
    pub groups: String,

    /// prior structure of model
    #[clap(value_enum)]
    pub model_type: ModelType,

    /// number of hidden layers in branches
    #[clap(short, long, default_value_t = 0)]
    pub depth: usize,

    /// path to output dir. Dir with the simulated data will be created there.
    #[clap(short, long, default_value = "./")]
    pub outdir: String,

    /// proportion of effective markers
    #[clap(short, long, default_value_t = 1.0)]
    pub proportion_effective: f32,

    /// variance of network params upon initialization
    #[clap(long, default_value_t = 0.05)]
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

    /// enable debug prints
    #[clap(long)]
    pub debug: bool,
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

    /// proportion of effective markers
    #[clap(short, long, default_value_t = 1.0)]
    pub proportion_effective: f32,

    /// width of summary layer. By default equal to hidden layer width
    #[clap(long)]
    pub summary_layer_width: Option<usize>,

    /// variance of network params upon initialization
    #[clap(long, default_value_t = 0.05)]
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
pub(crate) struct TrainOldModelArgs {
    /// Prior structure of model.
    #[clap(value_enum)]
    pub model_type: ModelType,

    /// model file
    pub model_file: String,

    /// perturb model parameters before training by specified amount
    #[clap(long)]
    pub perturb_params: Option<f32>,

    /// perturb model precisions before training by specified amount
    #[clap(long)]
    pub perturb_precisions: Option<f32>,
}

impl TrainOldModelArgs {
    pub fn to_file(&self, path: &Path) {
        info!("Creating: {:?}", path);
        to_writer_pretty(File::create(path).unwrap(), self).unwrap();
    }
}

#[derive(Args, Debug, Serialize, Deserialize)]
pub(crate) struct TrainNewModelArgs {
    /// Prior structure of model.
    #[clap(value_enum)]
    pub model_type: ModelType,

    /// number of hidden layers in branches
    pub branch_depth: usize,

    /// sets the width of all hidden layers to the input size of the branch
    /// times this value
    #[clap(long, default_value_t = 0.5)]
    pub relative_hidden_layer_width: f32,

    /// fixes the width of all hidden layers, if set. Takes priority over `relative_hidden_layer_width`
    #[clap(long)]
    pub fixed_hidden_layer_width: Option<usize>,

    /// sets the width of all summary layers to the hidden layer size of the branch
    /// times this value.
    #[clap(long, default_value_t = 1.0)]
    pub relative_summary_layer_width: f32,

    /// fixes the width of all summary layers, if set. Takes priority over `relative_summary_layer_width`
    #[clap(long)]
    pub fixed_summary_layer_width: Option<usize>,

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
}

impl TrainNewModelArgs {
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
    /// Stem of input .bed + .dim or .bed + .bim + .fam files
    pub bfile: String,

    /// Path to .grouping file
    pub groups: String,

    /// Path to models generated with `train-new` or `train` command
    #[clap(short, long, default_value = "./models")]
    pub model_path: String,
}

/// Determine activations in trained model network given a specific input.
/// This returns one activation file for each sampled model
/// stored in the .models dir generated in a `rs-bann train-new` run.
#[derive(Args, Debug, Serialize)]
pub(crate) struct ActivationArgs {
    /// Stem of input .bed + .dim or .bed + .bim + .fam files
    pub bfile: String,

    /// Path to .grouping file
    pub groups: String,

    /// Path to models generated with `train-new` or `train` command
    #[clap(short, long, default_value = "./models")]
    pub model_path: String,
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
    /// Stem of input .bed + .dim or .bed + .bim + .fam files
    pub bfile: String,

    /// Path to input .phen file.
    /// Should contain a rs-bann Phenotypes instance.
    pub phen: String,

    /// Path to .grouping file
    pub groups: String,

    /// Path to models generated with `train-new` or `train` command
    #[clap(short, long, default_value = "./models")]
    pub model_path: String,
}
