mod cli;

use clap::Parser;
use cli::cli::{Cli, ModelType, SimulateArgs, SubCmd, TrainArgs};
use log::info;
use rand::thread_rng;
use rand_distr::{Binomial, Distribution, Normal, Uniform};
use rs_bann::net::{
    architectures::BlockNetCfg,
    branch::{ard_branch::ArdBranch, base_branch::BaseBranch, std_normal_branch::StdNormalBranch},
    data::Data,
    mcmc_cfg::MCMCCfg,
    train_stats::ReportCfg,
};
use statrs::statistics::Statistics;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

fn main() {
    match Cli::parse().cmd {
        SubCmd::Simulate(args) => simulate(args),
        SubCmd::Train(args) => train(args),
    }
}

// TODO:
// Unless the groups get to large, I can do everything on col major files
// and col major bedvecs.
// the preprocessing routine only has to split the large .bed into groups
// following some annotation input.
// fn preprocess() {
//     unimplemented!();
// }

// // The following lower bounds for memory consumption are expected,
// // if only a subset of all samples is loaded at a time
// // n    pg  mem[mb]
// // 10k  1k  10**4 * 1x10**3 * 0.25 = 10**7 * 0.75 b =  7.5 Mb
// // I need fast random reading of this data.
// fn train() {
//     unimplemented!();
// }

// fn predict() {
//     unimplemented!();
// }

fn simulate(args: SimulateArgs) {
    simple_logger::init_with_level(log::Level::Info).unwrap();

    if let Some(val) = args.heritability {
        if !(val >= 0. && val <= 1.) {
            panic!("Heritability must be within [0, 1].");
        }
    }
    let path = Path::new(&args.outdir);
    if !path.exists() {
        std::fs::create_dir_all(path).expect("Could not create output directory!");
    }
    let train_path = path.join("train.bin");
    let test_path = path.join("test.bin");
    let args_path = path.join("args.json");
    let params_path = path.join("model.params");

    info!("Building model");
    let mut net_cfg = BlockNetCfg::<BaseBranch>::new()
        .with_depth(args.branch_depth)
        .with_initial_random_range(2.0);
    for _ in 0..args.num_branches {
        net_cfg.add_branch(args.num_markers_per_branch, args.hidden_layer_width);
    }
    let net = net_cfg.build_net();

    info!("Saving model params");
    info!("Creating: {:?}", params_path);
    let mut net_params_file = BufWriter::new(File::create(params_path).unwrap());
    for branch_ix in 0..args.num_branches {
        writeln!(
            &mut net_params_file,
            "{:?}",
            net.branch_cfg(branch_ix).params()
        )
        .expect("Failed to write model params");
    }

    let mut rng = thread_rng();

    info!("Generating random marker data");
    let gt_per_branch = args.num_markers_per_branch * args.num_individuals;
    let mut x_train: Vec<Vec<f32>> = vec![vec![0.0; gt_per_branch]; args.num_branches];
    let mut x_test: Vec<Vec<f32>> = vec![vec![0.0; gt_per_branch]; args.num_branches];
    let mut x_means: Vec<Vec<f32>> =
        vec![vec![0.0; args.num_markers_per_branch]; args.num_branches];
    let mut x_stds: Vec<Vec<f32>> = vec![vec![0.0; args.num_markers_per_branch]; args.num_branches];
    for branch_ix in 0..args.num_branches {
        for marker_ix in 0..args.num_markers_per_branch {
            let maf = Uniform::from(0.0..0.5).sample(&mut rng);
            x_means[branch_ix][marker_ix] = 2. * maf;
            x_stds[branch_ix][marker_ix] = (2. * maf * (1. - maf)).sqrt();

            let binom = Binomial::new(2, maf).unwrap();
            (0..args.num_individuals).for_each(|i| {
                x_train[branch_ix][marker_ix * args.num_individuals + i] =
                    binom.sample(&mut rng) as f32
            });

            let maf = Uniform::from(0.0..0.5).sample(&mut rng);
            let binom = Binomial::new(2, maf).unwrap();
            (0..args.num_individuals).for_each(|i| {
                x_test[branch_ix][marker_ix * args.num_individuals + i] =
                    binom.sample(&mut rng) as f32
            });
        }
    }

    info!("Making phenotype data");
    let mut y_train = net.predict(&x_train, args.num_individuals);
    let mut y_test = net.predict(&x_test, args.num_individuals);

    if let Some(h) = args.heritability {
        let s2_train = (&y_train).variance();
        let train_residual_variance = s2_train * (1. / h - 1.);
        let rv_train_dist = Normal::new(0.0, train_residual_variance.sqrt()).unwrap();
        y_train
            .iter_mut()
            .for_each(|e| *e += rv_train_dist.sample(&mut rng) as f32);
        info!("Train data: Added residual variance of {:2} to variance explained {:2} (total variance: {:2})", train_residual_variance, s2_train, (&y_train).variance());

        let s2_test = (&y_test).variance();
        let test_residual_variance = s2_test * (1. / h - 1.);
        let rv_test_dist = Normal::new(0.0, test_residual_variance.sqrt()).unwrap();
        y_test
            .iter_mut()
            .for_each(|e| *e += rv_test_dist.sample(&mut rng) as f32);
        info!("Test data: Added residual variance of {:2} to variance explained {:2} (total variance: {:2})", test_residual_variance, s2_test, (&y_test).variance());
    }

    Data::new(
        x_train,
        y_train,
        x_means.clone(),
        x_stds.clone(),
        args.num_markers_per_branch,
        args.num_individuals,
        false,
    )
    .to_file(&train_path);
    Data::new(
        x_test,
        y_test,
        x_means,
        x_stds,
        args.num_markers_per_branch,
        args.num_individuals,
        false,
    )
    .to_file(&test_path);

    args.to_file(&args_path);
}

fn train(args: TrainArgs) {
    if args.debug_prints {
        simple_logger::init_with_level(log::Level::Debug).unwrap();
    } else {
        simple_logger::init_with_level(log::Level::Info).unwrap();
    }

    let args_path = Path::new(&args.outpath).join("args.json");
    args.to_file(&args_path);

    info!("Loading data");
    let mut train_data = Data::from_file(&Path::new(&args.indir).join("train.bin"));
    let mut test_data = Data::from_file(&Path::new(&args.indir).join("test.bin"));

    if args.standardize {
        info!("Standardizing data");
        train_data.standardize();
        test_data.standardize();
    }

    let mcmc_cfg = MCMCCfg {
        hmc_step_size_factor: args.step_size,
        hmc_max_hamiltonian_error: args.max_hamiltonian_error,
        hmc_integration_length: args.integration_length,
        hmc_step_size_mode: args.step_size_mode,
        chain_length: args.chain_length,
        outpath: args.outpath,
        trace: args.trace,
        trajectories: args.trajectories,
        num_grad_traj: args.num_grad_traj,
    };
    mcmc_cfg.create_out();

    let report_cfg = ReportCfg::new(args.report_interval, Some(&test_data));

    info!("Building net");

    match args.model_type {
        ModelType::Base => {
            let mut net_cfg = BlockNetCfg::<BaseBranch>::new()
                .with_depth(args.branch_depth)
                .with_precision_prior(args.prior_shape, args.prior_scale);

            for _ in 0..train_data.num_branches() {
                net_cfg.add_branch(train_data.num_markers_per_branch(), args.hidden_layer_width);
            }
            let mut net = net_cfg.build_net();
            info!(
                "Built net with {:} params per branch.",
                net.num_branch_params(0)
            );
            net.write_meta(&mcmc_cfg);
            info!("Training net");
            net.train(&train_data, &mcmc_cfg, true, Some(report_cfg));
        }
        ModelType::ARD => {
            let mut net_cfg = BlockNetCfg::<ArdBranch>::new()
                .with_depth(args.branch_depth)
                .with_precision_prior(args.prior_shape, args.prior_scale);

            for _ in 0..train_data.num_branches() {
                net_cfg.add_branch(train_data.num_markers_per_branch(), args.hidden_layer_width);
            }
            let mut net = net_cfg.build_net();
            info!(
                "Built net with {:} params per branch.",
                net.num_branch_params(0)
            );
            net.write_meta(&mcmc_cfg);
            info!("Training net");
            net.train(&train_data, &mcmc_cfg, true, Some(report_cfg));
        }
        ModelType::StdNormal => {
            let mut net_cfg = BlockNetCfg::<StdNormalBranch>::new().with_depth(args.branch_depth);

            for _ in 0..train_data.num_branches() {
                net_cfg.add_branch(train_data.num_markers_per_branch(), args.hidden_layer_width);
            }
            let mut net = net_cfg.build_net();
            info!(
                "Built net with {:} params per branch.",
                net.num_branch_params(0)
            );
            net.write_meta(&mcmc_cfg);
            info!("Training net");
            net.train(&train_data, &mcmc_cfg, true, Some(report_cfg));
        }
    };
}

// // tests single branch impl
// fn test_crate_af() {
//     let args = AFArgs::parse();

//     if args.debug_prints {
//         simple_logger::init_with_level(log::Level::Debug).unwrap();
//     } else {
//         simple_logger::init_with_level(log::Level::Info).unwrap();
//     }

//     info!("Starting af test");

//     // make random data
//     let w0: Array<f32> = randn(dim4![
//         args.num_markers as u64,
//         args.hidden_layer_width as u64,
//         1,
//         1
//     ]);
//     let w1: Array<f32> = randn(dim4![args.hidden_layer_width as u64, 1, 1, 1]);
//     let w2: Array<f32> = randn(dim4![1, 1, 1, 1]);
//     let b0: Array<f32> = randn(dim4![1, args.hidden_layer_width as u64, 1, 1]);
//     let b1: Array<f32> = randn(dim4![1, 1, 1, 1]);
//     let true_net = BranchBuilder::new()
//         .with_num_markers(args.num_markers as usize)
//         .add_hidden_layer(args.hidden_layer_width as usize)
//         .add_layer_weights(&w0)
//         .add_layer_biases(&b0)
//         .add_summary_weights(&w1)
//         .add_summary_bias(&b1)
//         .add_output_weight(&w2)
//         .build();

//     let x_train: Array<f32> = randn(dim4![
//         args.num_individuals as u64,
//         args.num_markers as u64,
//         1,
//         1
//     ]);
//     let y_train = true_net.predict(&x_train);
//     let x_test: Array<f32> = randn(dim4![
//         args.num_individuals as u64,
//         args.num_markers as u64,
//         1,
//         1
//     ]);
//     let y_test = true_net.predict(&x_test);

//     let mut train_net = BranchBuilder::new()
//         .with_num_markers(args.num_markers as usize)
//         .add_hidden_layer(args.hidden_layer_width as usize)
//         .with_initial_weights_value(1.)
//         .with_initial_bias_value(1.)
//         .build();

//     // train
//     let mut accepted_samples: u64 = 0;

//     let step_size_mode = if args.std_scaled_step_sizes {
//         StepSizeMode::StdScaled
//     } else if args.random_step_sizes {
//         StepSizeMode::Random
//     } else {
//         StepSizeMode::Uniform
//     };

//     let mcmc_cfg = MCMCCfg {
//         hmc_step_size_factor: args.step_size,
//         hmc_max_hamiltonian_error: args.max_hamiltonian_error,
//         hmc_integration_length: args.integration_length,
//         chain_length: args.chain_length,
//         hmc_step_size_mode: step_size_mode,
//     };

//     for i in 0..args.chain_length {
//         if matches!(
//             train_net.hmc_step(&x_train, &y_train, &mcmc_cfg),
//             HMCStepResult::Accepted
//         ) {
//             accepted_samples += 1;
//             info!(
//                 "iteration: {:?} \t| loss (train): {:?} \t| loss (test): {:?}",
//                 i,
//                 train_net.rss(&x_train, &y_train),
//                 train_net.rss(&x_test, &y_test)
//             );
//         }
//     }

//     let acceptance_rate: f32 = accepted_samples as f32 / args.chain_length as f32;
//     info!("Finished. Overall acceptance rate: {:?}", acceptance_rate);
// }

// fn test_crate_ndarray() {
//     let reader = BedReader::new("resources/test/four_by_two.bed", 4, 2);
//     let mut mg = MarkerGroup::new(
//         arr1(&[-0.587_430_3, 0.020_813_8, 0.346_810_51, 0.283_149_64]),
//         arr1(&[1., 1.]),
//         1.,
//         1.,
//         reader,
//         2,
//     );
//     mg.load_marker_data();
//     let mut prev_pos = arr1(&[-0.587_430_3, 0.020_813_8, 0.346_810_51, 0.283_149_64]);
//     let n_samples = 1000;
//     let mut n_rejected = 0;
//     for _i in 0..n_samples {
//         let (new_pos, u_turn_at) = mg.sample_params(10);
//         if new_pos == prev_pos {
//             n_rejected += 1;
//         }
//         if let Some(ix) = u_turn_at {
//             println!("{:?}\t{:?}", new_pos, ix);
//         } else {
//             println!("{:?}\tnan", new_pos);
//         }
//         prev_pos = new_pos.clone();
//         mg.set_params(&new_pos);
//     }
//     mg.forget_marker_data();
//     dbg!(n_rejected as f32 / n_samples as f32);
// }
