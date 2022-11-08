mod cli;

use clap::Parser;
use cli::cli::{
    Cli, GroupCenteredArgs, SimulateXYArgs, SimulateYArgs, SubCmd, TrainArgs, TrainNewArgs,
};
use log::{info, warn};
use rand::thread_rng;
use rand_distr::{Binomial, Distribution, Normal, Uniform};
use rs_bann::group::{centered::CorrGraph, external::ExternalGrouping, grouping::MarkerGrouping};
use rs_bann::net::{
    architectures::BlockNetCfg,
    branch::{
        ard_branch::ArdBranch, base_branch::BaseBranch, branch::Branch,
        std_normal_branch::StdNormalBranch,
    },
    data::{Data, GenotypesBuilder, PhenStats, Phenotypes},
    mcmc_cfg::MCMCCfg,
    net::{ModelType, Net},
    train_stats::ReportCfg,
};
use serde_json::to_writer;
use statrs::statistics::Statistics;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

fn main() {
    match Cli::parse().cmd {
        SubCmd::SimulateY(args) => match args.model_type {
            ModelType::Base => simulate_y::<BaseBranch>(args),
            ModelType::ARD => simulate_y::<ArdBranch>(args),
            ModelType::StdNormal => simulate_y::<StdNormalBranch>(args),
        },
        SubCmd::SimulateXY(args) => match args.model_type {
            ModelType::Base => simulate_xy::<BaseBranch>(args),
            ModelType::ARD => simulate_xy::<ArdBranch>(args),
            ModelType::StdNormal => simulate_xy::<StdNormalBranch>(args),
        },
        SubCmd::TrainNew(args) => train_new(args),
        SubCmd::Train(args) => train(args),
        SubCmd::GroupCentered(args) => group_centered(args),
    }
}

fn group_centered(args: GroupCenteredArgs) {
    let mut bim_path = PathBuf::from(&args.inpath);
    bim_path.set_extension("bim");
    let mut corr_path = PathBuf::from(&args.inpath);
    corr_path.set_extension("ld");
    let mut outpath = Path::new(&args.outdir).join(bim_path.file_stem().unwrap());
    outpath.set_extension("centered_grouping");
    CorrGraph::from_plink_ld(&corr_path, &bim_path)
        .centered_grouping()
        .to_file(&outpath);
}

fn simulate_y<B>(args: SimulateYArgs)
where
    B: Branch,
{
    simple_logger::init_with_level(log::Level::Info).unwrap();

    if !(args.heritability >= 0. && args.heritability <= 1.) {
        panic!("Heritability must be within [0, 1].");
    }

    let train_bed_path = Path::new(&args.train_bed);
    let test_bed_path = Path::new(&args.test_bed);

    let mut path = Path::new(&args.outdir).join(format!(
        "{}_{}_d{}_h{}_v{}",
        train_bed_path.file_stem().unwrap().to_string_lossy(),
        args.model_type,
        args.branch_depth,
        args.heritability,
        args.init_param_variance
    ));

    if let (Some(k), Some(s)) = (args.init_gamma_shape, args.init_gamma_scale) {
        path = Path::new(&args.outdir).join(format!(
            "{}_{}_d{}_h{}_k{}_s{}",
            train_bed_path.file_stem().unwrap().to_string_lossy(),
            args.model_type,
            args.branch_depth,
            args.heritability,
            k,
            s
        ));
    }

    if !path.exists() {
        std::fs::create_dir_all(&path).expect("Could not create output directory!");
    }
    let train_path = path.join("train.bin");
    let test_path = path.join("test.bin");
    let args_path = path.join("args.json");
    let params_path = path.join("model.params");
    let model_path = path.join("model.bin");

    // TODO: depth should be set such that the largest branch still has n > p.
    // Although, that is for models I want to train. For simulation it doesn't
    // matter I guess.
    info!("Building model");
    let mut net_cfg = if let (Some(k), Some(s)) = (args.init_gamma_shape, args.init_gamma_scale) {
        BlockNetCfg::<B>::new()
            .with_depth(args.branch_depth)
            .with_init_gamma_params(k, s)
            .with_precision_prior(k, s)
    } else {
        BlockNetCfg::<B>::new()
            .with_depth(args.branch_depth)
            .with_init_param_variance(args.init_param_variance)
    };

    // load groups
    let grouping_path = Path::new(&args.groups);
    let grouping = ExternalGrouping::from_file(&grouping_path);

    // width is fixed to half the number of input nodes
    for size in grouping.group_sizes() {
        net_cfg.add_branch(size, size / 2);
    }
    let net = net_cfg.build_net();

    info!("Saving model");
    net.to_file(&model_path);

    info!("Saving model params");
    info!("Creating: {:?}", params_path);
    let mut net_params_file = BufWriter::new(File::create(params_path).unwrap());
    to_writer(&mut net_params_file, net.branch_cfgs()).unwrap();
    net_params_file.write_all(b"\n").unwrap();

    // load marker data from .bed
    let gen_train = GenotypesBuilder::new()
        .with_x_from_bed(&train_bed_path, &grouping)
        .build()
        .unwrap();
    let gen_test = GenotypesBuilder::new()
        .with_x_from_bed(&test_bed_path, &grouping)
        .build()
        .unwrap();

    info!("Making phenotype data");
    let mut y_train = net.predict(&gen_train);
    let mut y_test = net.predict(&gen_test);

    let mut train_residual_variance: f64 = 0.0;
    let mut test_residual_variance: f64 = 0.0;

    if args.heritability != 1. {
        let mut rng = thread_rng();
        let s2_train = (&y_train.iter().map(|e| *e as f64).collect::<Vec<f64>>()).variance();
        train_residual_variance = s2_train * (1. / args.heritability as f64 - 1.);
        let rv_train_dist = Normal::new(0.0, train_residual_variance.sqrt()).unwrap();
        y_train
            .iter_mut()
            .for_each(|e| *e += rv_train_dist.sample(&mut rng) as f32);
        let s2_test = (&y_test.iter().map(|e| *e as f64).collect::<Vec<f64>>()).variance();
        test_residual_variance = s2_test * (1. / args.heritability as f64 - 1.);
        let rv_test_dist = Normal::new(0.0, test_residual_variance.sqrt()).unwrap();
        y_test
            .iter_mut()
            .for_each(|e| *e += rv_test_dist.sample(&mut rng) as f32);
    }

    let train_data = Data::new(gen_train, Phenotypes::new(y_train.clone()));
    let test_data = Data::new(gen_test, Phenotypes::new(y_test.clone()));

    PhenStats::new(
        (&y_test.iter().map(|e| *e as f64).collect::<Vec<f64>>()).mean() as f32,
        (&y_test.iter().map(|e| *e as f64).collect::<Vec<f64>>()).variance() as f32,
        test_residual_variance as f32,
        net.mse(&test_data),
    )
    .to_file(&path.join("test_phen_stats.json"));

    PhenStats::new(
        (&y_train.iter().map(|e| *e as f64).collect::<Vec<f64>>()).mean() as f32,
        (&y_train.iter().map(|e| *e as f64).collect::<Vec<f64>>()).variance() as f32,
        train_residual_variance as f32,
        net.mse(&train_data),
    )
    .to_file(&path.join("train_phen_stats.json"));

    train_data.to_file(&train_path);
    test_data.to_file(&test_path);

    if args.json_data {
        train_data.to_json(&path.join("train.json"));
        test_data.to_json(&path.join("test.json"));
    }

    args.to_file(&args_path);
}

fn simulate_xy<B>(args: SimulateXYArgs)
where
    B: Branch,
{
    simple_logger::init_with_level(log::Level::Info).unwrap();

    if !(args.heritability >= 0. && args.heritability <= 1.) {
        panic!("Heritability must be within [0, 1].");
    }

    let mut path = Path::new(&args.outdir).join(format!(
        "{}_b{}_w{}_d{}_m{}_n{}_h{}_v{}",
        args.model_type,
        args.num_branches,
        args.hidden_layer_width,
        args.branch_depth,
        args.num_markers_per_branch,
        args.num_individuals,
        args.heritability,
        args.init_param_variance
    ));

    if let (Some(k), Some(s)) = (args.init_gamma_shape, args.init_gamma_scale) {
        path = Path::new(&args.outdir).join(format!(
            "{}_b{}_w{}_d{}_m{}_n{}_h{}_k{}_s{}",
            args.model_type,
            args.num_branches,
            args.hidden_layer_width,
            args.branch_depth,
            args.num_markers_per_branch,
            args.num_individuals,
            args.heritability,
            k,
            s
        ));
    }

    if !path.exists() {
        std::fs::create_dir_all(&path).expect("Could not create output directory!");
    }
    let train_path = path.join("train.bin");
    let test_path = path.join("test.bin");
    let args_path = path.join("args.json");
    let params_path = path.join("model.params");
    let model_path = path.join("model.bin");

    info!("Building model");
    let mut net_cfg = if let (Some(k), Some(s)) = (args.init_gamma_shape, args.init_gamma_scale) {
        BlockNetCfg::<B>::new()
            .with_depth(args.branch_depth)
            .with_init_gamma_params(k, s)
            .with_precision_prior(k, s)
    } else {
        BlockNetCfg::<B>::new()
            .with_depth(args.branch_depth)
            .with_init_param_variance(args.init_param_variance)
    };
    for _ in 0..args.num_branches {
        net_cfg.add_branch(args.num_markers_per_branch, args.hidden_layer_width);
    }
    let net = net_cfg.build_net();

    info!("Saving model");
    net.to_file(&model_path);

    info!("Saving model params");
    info!("Creating: {:?}", params_path);
    let mut net_params_file = BufWriter::new(File::create(params_path).unwrap());
    to_writer(&mut net_params_file, net.branch_cfgs()).unwrap();
    net_params_file.write_all(b"\n").unwrap();

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

            let binom = Binomial::new(2, maf as f64).unwrap();
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

    let gen_train = GenotypesBuilder::new()
        .with_x(
            x_train,
            vec![args.num_markers_per_branch; args.num_branches],
            args.num_individuals,
        )
        .build()
        .unwrap();
    let gen_test = GenotypesBuilder::new()
        .with_x(
            x_test,
            vec![args.num_markers_per_branch; args.num_branches],
            args.num_individuals,
        )
        .build()
        .unwrap();

    info!("Making phenotype data");
    let mut y_train = net.predict(&gen_train);
    let mut y_test = net.predict(&gen_test);

    let mut train_residual_variance: f64 = 0.0;
    let mut test_residual_variance: f64 = 0.0;

    if args.heritability != 1. {
        let s2_train = (&y_train.iter().map(|e| *e as f64).collect::<Vec<f64>>()).variance();
        train_residual_variance = s2_train * (1. / args.heritability as f64 - 1.);
        let rv_train_dist = Normal::new(0.0, train_residual_variance.sqrt()).unwrap();
        y_train
            .iter_mut()
            .for_each(|e| *e += rv_train_dist.sample(&mut rng) as f32);
        let s2_test = (&y_test.iter().map(|e| *e as f64).collect::<Vec<f64>>()).variance();
        test_residual_variance = s2_test * (1. / args.heritability as f64 - 1.);
        let rv_test_dist = Normal::new(0.0, test_residual_variance.sqrt()).unwrap();
        y_test
            .iter_mut()
            .for_each(|e| *e += rv_test_dist.sample(&mut rng) as f32);
    }

    let train_data = Data::new(gen_train, Phenotypes::new(y_train.clone()));
    let test_data = Data::new(gen_test, Phenotypes::new(y_test.clone()));

    PhenStats::new(
        (&y_test.iter().map(|e| *e as f64).collect::<Vec<f64>>()).mean() as f32,
        (&y_test.iter().map(|e| *e as f64).collect::<Vec<f64>>()).variance() as f32,
        test_residual_variance as f32,
        net.mse(&test_data),
    )
    .to_file(&path.join("test_phen_stats.json"));

    PhenStats::new(
        (&y_train.iter().map(|e| *e as f64).collect::<Vec<f64>>()).mean() as f32,
        (&y_train.iter().map(|e| *e as f64).collect::<Vec<f64>>()).variance() as f32,
        train_residual_variance as f32,
        net.mse(&train_data),
    )
    .to_file(&path.join("train_phen_stats.json"));

    train_data.to_file(&train_path);
    test_data.to_file(&test_path);

    if args.json_data {
        train_data.to_json(&path.join("train.json"));
        test_data.to_json(&path.join("test.json"));
    }

    args.to_file(&args_path);
}

fn train_new(args: TrainNewArgs) {
    if args.debug_prints {
        simple_logger::init_with_level(log::Level::Debug).unwrap();
    } else {
        simple_logger::init_with_level(log::Level::Info).unwrap();
    }

    info!("Loading data");
    let mut train_data = Data::from_file(&Path::new(&args.indir).join("train.bin"));
    let mut test_data = Data::from_file(&Path::new(&args.indir).join("test.bin"));

    if args.standardize {
        info!("Standardizing data");
        train_data.standardize_x();
        test_data.standardize_x();
    }

    let outdir = format!(
        "{}_w{}_d{}_cl{}_il{}_{}_k{}_s{}",
        args.model_type,
        args.hidden_layer_width,
        args.branch_depth,
        args.chain_length,
        args.integration_length,
        args.step_size_mode,
        args.prior_shape,
        args.prior_scale,
    );

    let mcmc_cfg = MCMCCfg {
        hmc_step_size_factor: args.step_size.clone(),
        hmc_max_hamiltonian_error: args.max_hamiltonian_error.clone(),
        hmc_integration_length: args.integration_length.clone(),
        hmc_step_size_mode: args.step_size_mode.clone(),
        chain_length: args.chain_length.clone(),
        outpath: outdir,
        trace: args.trace.clone(),
        trajectories: args.trajectories.clone(),
        num_grad_traj: args.num_grad_traj.clone(),
        num_grad: args.num_grad,
    };
    mcmc_cfg.create_out();

    args.to_file(&mcmc_cfg.args_path());

    let report_cfg = ReportCfg::new(args.report_interval, Some(&test_data));

    info!("Building net");

    match args.model_type {
        ModelType::Base => {
            let mut net_cfg = BlockNetCfg::<BaseBranch>::new()
                .with_depth(args.branch_depth)
                .with_precision_prior(args.prior_shape, args.prior_scale);

            for bix in 0..train_data.num_branches() {
                net_cfg.add_branch(
                    train_data.num_markers_in_branch(bix),
                    args.hidden_layer_width,
                );
            }
            let mut net = net_cfg.build_net();

            for bix in 0..net.num_branches() {
                if net.num_branch_params(bix) > train_data.num_individuals() {
                    warn!(
                        "Num params > num individuals in branch {} (with {} params, {} individuals)",
                        bix, net.num_branch_params(bix), train_data.num_individuals());
                }
            }
            net.write_meta(&mcmc_cfg);
            info!("Training net");
            net.train(&train_data, &mcmc_cfg, true, Some(report_cfg));
        }
        ModelType::ARD => {
            let mut net_cfg = BlockNetCfg::<ArdBranch>::new()
                .with_depth(args.branch_depth)
                .with_precision_prior(args.prior_shape, args.prior_scale);

            for bix in 0..train_data.num_branches() {
                net_cfg.add_branch(
                    train_data.num_markers_in_branch(bix),
                    args.hidden_layer_width,
                );
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

            for bix in 0..train_data.num_branches() {
                net_cfg.add_branch(
                    train_data.num_markers_in_branch(bix),
                    args.hidden_layer_width,
                );
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

fn train(args: TrainArgs) {
    if args.debug_prints {
        simple_logger::init_with_level(log::Level::Debug).unwrap();
    } else {
        simple_logger::init_with_level(log::Level::Info).unwrap();
    }

    info!("Loading data");
    let mut train_data = Data::from_file(&Path::new(&args.indir).join("train.bin"));
    let mut test_data = Data::from_file(&Path::new(&args.indir).join("test.bin"));

    if args.standardize {
        info!("Standardizing data");
        train_data.standardize_x();
        test_data.standardize_x();
    }

    let model_path = Path::new(&args.model_file);

    let outdir = format!(
        "{}_cl{}_il{}_{}",
        model_path.file_stem().unwrap().to_string_lossy(),
        args.chain_length,
        args.integration_length,
        args.step_size_mode,
    );

    let mcmc_cfg = MCMCCfg {
        hmc_step_size_factor: args.step_size.clone(),
        hmc_max_hamiltonian_error: args.max_hamiltonian_error.clone(),
        hmc_integration_length: args.integration_length.clone(),
        hmc_step_size_mode: args.step_size_mode.clone(),
        chain_length: args.chain_length.clone(),
        outpath: outdir,
        trace: args.trace.clone(),
        trajectories: args.trajectories.clone(),
        num_grad_traj: args.num_grad_traj.clone(),
        num_grad: args.num_grad,
    };
    mcmc_cfg.create_out();

    args.to_file(&mcmc_cfg.args_path());

    let report_cfg = ReportCfg::new(args.report_interval, Some(&test_data));

    info!("Loading net");

    match args.model_type {
        ModelType::Base => {
            let mut net = Net::<BaseBranch>::from_file(&model_path);
            net.write_meta(&mcmc_cfg);
            info!("Training net");
            net.train(&train_data, &mcmc_cfg, true, Some(report_cfg));
        }
        ModelType::ARD => {
            let mut net = Net::<ArdBranch>::from_file(&model_path);
            net.write_meta(&mcmc_cfg);
            info!("Training net");
            net.train(&train_data, &mcmc_cfg, true, Some(report_cfg));
        }
        ModelType::StdNormal => {
            let mut net = Net::<StdNormalBranch>::from_file(&model_path);
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
