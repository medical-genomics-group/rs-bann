use arrayfire::{dim4, randn, Array};
use clap::Parser;
use log::info;
use ndarray::arr1;
use rand::thread_rng;
use rand_distr::{Binomial, Distribution, Uniform};
use rs_bann::net::{
    architectures::BlockNetCfg,
    branch::{branch::HMCStepResult, branch_builder::BranchBuilder},
    mcmc_cfg::{MCMCCfg, StepSizeMode},
    net::{Data, ReportCfg},
};
use rs_bann::network::MarkerGroup;
use rs_bedvec::io::BedReader;

/// A small bayesian neural network implementation.
/// Number of markers per branch: fixed
/// Depth of branches: same for all branches
/// Width of branch layers: same within branches, dynamic between branches
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct BlockNetArgs {
    /// number of input features per branch (markers)
    num_markers_per_branch: usize,

    /// number of branches (markers)
    num_branches: usize,

    /// number of samples (individuals)
    num_individuals: usize,

    /// width of hidden layer
    hidden_layer_width: usize,

    /// number of hidden layers in branches
    branch_depth: usize,

    /// prior shape
    prior_shape: f64,

    /// prior scale
    prior_scale: f64,

    /// full model chain length
    chain_length: usize,

    /// hmc max hamiltonian error
    max_hamiltonian_error: f64,

    /// hmc integration length
    integration_length: usize,

    /// hmc step size, acts as a modifying factor on random step sizes if enabled
    step_size: f64,

    /// training stats report interval
    report_interval: usize,

    /// enable random step sizes
    #[clap(short, long)]
    random_step_sizes: bool,

    /// enable step sizes scales by prior standard deviation.
    /// Takes precedence of random_step_sizes if enabled.
    #[clap(short, long)]
    std_scaled_step_sizes: bool,

    /// enable debug prints
    #[clap(short, long)]
    debug_prints: bool,
}

/// A small bayesian neural network implementation based on ArrayFire.
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct AFArgs {
    /// number of input feature (markers)
    num_markers: usize,

    /// number of samples (individuals)
    num_individuals: usize,

    /// width of hidden layer
    hidden_layer_width: usize,

    /// hmc integration length
    integration_length: usize,

    /// chain length (number of hmc samples)
    chain_length: usize,

    /// max hamiltonian error
    max_hamiltonian_error: f64,

    /// hmc step size, acts as a modifying factor on random or scaled step size if enables
    step_size: f64,

    /// enable random step sizes
    #[clap(short, long)]
    random_step_sizes: bool,

    /// enable step sizes scales by prior standard deviation.
    /// Takes precedence of random_step_sizes if enabled.
    #[clap(short, long)]
    std_scaled_step_sizes: bool,

    /// enable debug prints
    #[clap(short, long)]
    debug_prints: bool,
}

fn main() {
    test_block_net();
    // test_crate_af();
}

// TODO:
// Unless the groups get to large, I can do everything on col major files
// and col major bedvecs.
// the preprocessing routine only has to split the large .bed into groups
// following some annotation input.
fn preprocess() {
    unimplemented!();
}

// The following lower bounds for memory consumption are expected,
// if only a subset of all samples is loaded at a time
// n    pg  mem[mb]
// 10k  1k  10**4 * 1x10**3 * 0.25 = 10**7 * 0.75 b =  7.5 Mb
// I need fast random reading of this data.
fn train() {
    unimplemented!();
}

fn predict() {
    unimplemented!();
}

// tests block net architecture
fn test_block_net() {
    let args = BlockNetArgs::parse();

    if args.debug_prints {
        simple_logger::init_with_level(log::Level::Debug).unwrap();
    } else {
        simple_logger::init_with_level(log::Level::Info).unwrap();
    }

    info!("Starting block net test");

    info!("Building true net");
    let mut true_net_cfg = BlockNetCfg::new()
        .with_depth(args.branch_depth)
        .with_precision_prior(args.prior_shape, args.prior_scale)
        .with_initial_random_range(2.0);
    for _ in 0..args.num_branches {
        true_net_cfg.add_branch(args.num_markers_per_branch, args.hidden_layer_width);
    }
    let true_net = true_net_cfg.build_net();

    let mut rng = thread_rng();

    info!("Making random marker data");
    let gt_per_branch = args.num_markers_per_branch * args.num_individuals;
    let mut x_train: Vec<Vec<f64>> = vec![vec![0.0; gt_per_branch]; args.num_branches];
    let mut x_test: Vec<Vec<f64>> = vec![vec![0.0; gt_per_branch]; args.num_branches];
    for branch_ix in 0..args.num_branches {
        for marker_ix in 0..args.num_markers_per_branch {
            let maf = Uniform::from(0.0..0.5).sample(&mut rng);
            (0..args.num_individuals).for_each(|i| {
                x_train[branch_ix][marker_ix * args.num_individuals + i] =
                    Binomial::new(2, maf).unwrap().sample(&mut rng) as f64
            });
            let maf = Uniform::from(0.0..0.5).sample(&mut rng);
            (0..args.num_individuals).for_each(|i| {
                x_test[branch_ix][marker_ix * args.num_individuals + i] =
                    Binomial::new(2, maf).unwrap().sample(&mut rng) as f64
            });
        }
    }

    info!("Making phenotype data");
    let y_train = true_net.predict(&x_train, args.num_individuals);
    let y_test = true_net.predict(&x_test, args.num_individuals);

    info!("Building net to train");
    let mut net_cfg = BlockNetCfg::new()
        .with_depth(args.branch_depth)
        .with_precision_prior(args.prior_shape, args.prior_scale);
    for _ in 0..args.num_branches {
        net_cfg.add_branch(args.num_markers_per_branch, args.hidden_layer_width);
    }
    let mut net = net_cfg.build_net();

    let step_size_mode = if args.std_scaled_step_sizes {
        StepSizeMode::StdScaled
    } else if args.random_step_sizes {
        StepSizeMode::Random
    } else {
        StepSizeMode::Uniform
    };

    info!(
        "Built net with {:} params per branch.",
        net.num_branch_params(0)
    );

    info!("Training net");
    let mcmc_cfg = MCMCCfg {
        hmc_step_size_factor: args.step_size,
        hmc_max_hamiltonian_error: args.max_hamiltonian_error,
        hmc_integration_length: args.integration_length,
        hmc_step_size_mode: step_size_mode,
        chain_length: args.chain_length,
    };

    let train_data = Data::new(&x_train, &y_train);
    let test_data = Data::new(&x_test, &y_test);
    let report_cfg = ReportCfg::new(args.report_interval, Some(&test_data));

    net.train(&train_data, &mcmc_cfg, true, Some(report_cfg));
}

// tests single branch impl
fn test_crate_af() {
    let args = AFArgs::parse();

    if args.debug_prints {
        simple_logger::init_with_level(log::Level::Debug).unwrap();
    } else {
        simple_logger::init_with_level(log::Level::Info).unwrap();
    }

    info!("Starting af test");

    // make random data
    let w0: Array<f64> = randn(dim4![
        args.num_markers as u64,
        args.hidden_layer_width as u64,
        1,
        1
    ]);
    let w1: Array<f64> = randn(dim4![args.hidden_layer_width as u64, 1, 1, 1]);
    let w2: Array<f64> = randn(dim4![1, 1, 1, 1]);
    let b0: Array<f64> = randn(dim4![1, args.hidden_layer_width as u64, 1, 1]);
    let b1: Array<f64> = randn(dim4![1, 1, 1, 1]);
    let true_net = BranchBuilder::new()
        .with_num_markers(args.num_markers as usize)
        .add_hidden_layer(args.hidden_layer_width as usize)
        .add_layer_weights(&w0)
        .add_layer_biases(&b0)
        .add_summary_weights(&w1)
        .add_summary_bias(&b1)
        .add_output_weight(&w2)
        .build();

    let x_train: Array<f64> = randn(dim4![
        args.num_individuals as u64,
        args.num_markers as u64,
        1,
        1
    ]);
    let y_train = true_net.predict(&x_train);
    let x_test: Array<f64> = randn(dim4![
        args.num_individuals as u64,
        args.num_markers as u64,
        1,
        1
    ]);
    let y_test = true_net.predict(&x_test);

    let mut train_net = BranchBuilder::new()
        .with_num_markers(args.num_markers as usize)
        .add_hidden_layer(args.hidden_layer_width as usize)
        .with_initial_weights_value(1.)
        .with_initial_bias_value(1.)
        .build();

    // train
    let mut accepted_samples: u64 = 0;

    let step_size_mode = if args.std_scaled_step_sizes {
        StepSizeMode::StdScaled
    } else if args.random_step_sizes {
        StepSizeMode::Random
    } else {
        StepSizeMode::Uniform
    };

    let mcmc_cfg = MCMCCfg {
        hmc_step_size_factor: args.step_size,
        hmc_max_hamiltonian_error: args.max_hamiltonian_error,
        hmc_integration_length: args.integration_length,
        chain_length: args.chain_length,
        hmc_step_size_mode: step_size_mode,
    };

    for i in 0..args.chain_length {
        if matches!(
            train_net.hmc_step(&x_train, &y_train, &mcmc_cfg),
            HMCStepResult::Accepted
        ) {
            accepted_samples += 1;
            info!(
                "iteration: {:?} \t| loss (train): {:?} \t| loss (test): {:?}",
                i,
                train_net.rss(&x_train, &y_train),
                train_net.rss(&x_test, &y_test)
            );
        }
    }

    let acceptance_rate: f64 = accepted_samples as f64 / args.chain_length as f64;
    info!("Finished. Overall acceptance rate: {:?}", acceptance_rate);
}

fn test_crate_ndarray() {
    let reader = BedReader::new("resources/test/four_by_two.bed", 4, 2);
    let mut mg = MarkerGroup::new(
        arr1(&[-0.587_430_3, 0.020_813_8, 0.346_810_51, 0.283_149_64]),
        arr1(&[1., 1.]),
        1.,
        1.,
        reader,
        2,
    );
    mg.load_marker_data();
    let mut prev_pos = arr1(&[-0.587_430_3, 0.020_813_8, 0.346_810_51, 0.283_149_64]);
    let n_samples = 1000;
    let mut n_rejected = 0;
    for _i in 0..n_samples {
        let (new_pos, u_turn_at) = mg.sample_params(10);
        if new_pos == prev_pos {
            n_rejected += 1;
        }
        if let Some(ix) = u_turn_at {
            println!("{:?}\t{:?}", new_pos, ix);
        } else {
            println!("{:?}\tnan", new_pos);
        }
        prev_pos = new_pos.clone();
        mg.set_params(&new_pos);
    }
    mg.forget_marker_data();
    dbg!(n_rejected as f64 / n_samples as f64);
}
