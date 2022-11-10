mod cli;

use clap::Parser;
use cli::cli::{
    Cli, GroupCenteredArgs, PredictArgs, SimulateXYArgs, SimulateYArgs, SubCmd, TrainArgs,
    TrainNewArgs,
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
    data::{Data, Genotypes, GenotypesBuilder, PhenStats, Phenotypes},
    mcmc_cfg::MCMCCfg,
    net::{ModelType, Net},
    train_stats::ReportCfg,
};
use serde_json::to_writer;
use statrs::statistics::Statistics;
use std::{
    fs::{read_dir, File},
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
        SubCmd::Predict(args) => predict(args),
    }
}

fn predict(args: PredictArgs) {
    let mut genotypes = Genotypes::from_file(&Path::new(&args.input_data));
    if args.standardize {
        genotypes.standardize();
    }
    // get model type
    let parent_path = Path::new(&args.model_path)
        .parent()
        .unwrap()
        .join("args.json");
    let train_args = TrainNewArgs::from_file(&parent_path);
    let model_type = train_args.model_type;
    // stdout writer in csv format
    let mut wtr = csv::Writer::from_writer(std::io::stdout());

    // load models and predict
    let mut model_files = read_dir(Path::new(&args.model_path))
        .expect("Failed to parse model dir")
        .map(|res| res.map(|e| e.path()))
        .filter_map(|e| e.ok())
        .filter(|e| e.is_file())
        .collect::<Vec<PathBuf>>();
    model_files.sort();

    for path in model_files {
        let prediction = match model_type {
            ModelType::ARD => Net::<ArdBranch>::from_file(&path).predict(&genotypes),
            ModelType::Base => Net::<BaseBranch>::from_file(&path).predict(&genotypes),
            ModelType::StdNormal => Net::<StdNormalBranch>::from_file(&path).predict(&genotypes),
        };
        wtr.write_record(prediction.iter().map(|e| e.to_string()))
            .unwrap();
    }
    wtr.flush().expect("Failed to flush csv writer");
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
        burn_in: args.burn_in,
        outpath: outdir,
        trace: args.trace.clone(),
        trajectories: args.trajectories.clone(),
        num_grad_traj: args.num_grad_traj.clone(),
        num_grad: args.num_grad,
        gradient_descent: args.gradient_descent,
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
            if let Some(p) = args.error_precision {
                net.set_error_precision(p);
            }

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
            if let Some(p) = args.error_precision {
                net.set_error_precision(p);
            }

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
        ModelType::StdNormal => {
            let mut net_cfg = BlockNetCfg::<StdNormalBranch>::new().with_depth(args.branch_depth);

            for bix in 0..train_data.num_branches() {
                net_cfg.add_branch(
                    train_data.num_markers_in_branch(bix),
                    args.hidden_layer_width,
                );
            }
            let mut net = net_cfg.build_net();
            if let Some(p) = args.error_precision {
                net.set_error_precision(p);
            }

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
        burn_in: args.burn_in,
        outpath: outdir,
        trace: args.trace.clone(),
        trajectories: args.trajectories.clone(),
        num_grad_traj: args.num_grad_traj.clone(),
        num_grad: args.num_grad,
        gradient_descent: args.gradient_descent,
    };
    mcmc_cfg.create_out();

    args.to_file(&mcmc_cfg.args_path());

    let report_cfg = ReportCfg::new(args.report_interval, Some(&test_data));

    info!("Loading net");

    match args.model_type {
        ModelType::Base => {
            let mut net = Net::<BaseBranch>::from_file(&model_path);
            if let Some(p) = args.error_precision {
                net.set_error_precision(p);
            }
            net.write_meta(&mcmc_cfg);
            info!("Training net");
            net.train(&train_data, &mcmc_cfg, true, Some(report_cfg));
        }
        ModelType::ARD => {
            let mut net = Net::<ArdBranch>::from_file(&model_path);
            if let Some(p) = args.error_precision {
                net.set_error_precision(p);
            }
            net.write_meta(&mcmc_cfg);
            info!("Training net");
            net.train(&train_data, &mcmc_cfg, true, Some(report_cfg));
        }
        ModelType::StdNormal => {
            let mut net = Net::<StdNormalBranch>::from_file(&model_path);
            if let Some(p) = args.error_precision {
                net.set_error_precision(p);
            }
            net.write_meta(&mcmc_cfg);
            info!("Training net");
            net.train(&train_data, &mcmc_cfg, true, Some(report_cfg));
        }
    };
}
