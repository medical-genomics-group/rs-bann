mod cli;

use clap::Parser;
use cli::cli::{
    ActivationArgs, BranchR2Args, Cli, GradientsArgs, GroupByGenesArgs, GroupCenteredArgs,
    MCMCArgs, PredictArgs, SimulateXYArgs, SimulateYArgs, SubCmd, TrainIOArgs, TrainNewModelArgs,
    TrainOldModelArgs,
};
use log::{debug, info, warn};
use rand::thread_rng;
use rand_distr::{Distribution, Normal, Uniform};
use rs_bann::data::genotypes::CompressedGenotypes;
use rs_bann::data::{
    data::Data, genotypes::GroupedGenotypes, phen_stats::PhenStats, phenotypes::Phenotypes,
};
use rs_bann::group::uniform::UniformGrouping;
use rs_bann::group::{
    centered::CorrGraph, external::ExternalGrouping, gene::GeneGrouping, grouping::MarkerGrouping,
};
use rs_bann::io::bed::BedVM;
use rs_bann::linear_model::LinearModelBuilder;
use rs_bann::net::mcmc_cfg::MCMCCfgBuilder;
use rs_bann::net::{
    architectures::{BlockNetCfg, HiddenLayerWidthRule, SummaryLayerWidthRule},
    branch::{
        branch_sampler::BranchSampler, lasso_ard::LassoArdBranch, lasso_base::LassoBaseBranch,
        ridge_ard::RidgeArdBranch, ridge_base::RidgeBaseBranch, std_normal_branch::StdNormalBranch,
    },
    model_type::ModelType,
    net::Net,
    train_stats::ReportCfg,
};
use serde_json::to_writer;
use statrs::statistics::Statistics;
use std::process::exit;
use std::str::FromStr;
use std::{
    fs::{read_dir, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

fn main() {
    match Cli::parse().cmd {
        SubCmd::SimulateY(args) => match args.model_type {
            ModelType::LassoBase => simulate_y::<LassoBaseBranch>(args),
            ModelType::LassoARD => simulate_y::<LassoArdBranch>(args),
            ModelType::RidgeBase => simulate_y::<RidgeBaseBranch>(args),
            ModelType::RidgeARD => simulate_y::<RidgeArdBranch>(args),
            ModelType::StdNormal => simulate_y::<StdNormalBranch>(args),
            ModelType::Linear => simulate_y_linear(args),
        },
        SubCmd::SimulateXY(args) => match args.model_type {
            ModelType::LassoBase => simulate_xy::<LassoBaseBranch>(args),
            ModelType::LassoARD => simulate_xy::<LassoArdBranch>(args),
            ModelType::RidgeBase => simulate_xy::<RidgeBaseBranch>(args),
            ModelType::RidgeARD => simulate_xy::<RidgeArdBranch>(args),
            ModelType::StdNormal => simulate_xy::<StdNormalBranch>(args),
            ModelType::Linear => simulate_xy_linear(args),
        },
        SubCmd::TrainNew {
            input_args,
            mcmc_args,
            model_args,
        } => match model_args.model_type {
            ModelType::LassoBase => train_new::<LassoBaseBranch>(input_args, mcmc_args, model_args),
            ModelType::LassoARD => train_new::<LassoArdBranch>(input_args, mcmc_args, model_args),
            ModelType::RidgeBase => train_new::<RidgeBaseBranch>(input_args, mcmc_args, model_args),
            ModelType::RidgeARD => train_new::<RidgeArdBranch>(input_args, mcmc_args, model_args),
            ModelType::StdNormal => train_new::<StdNormalBranch>(input_args, mcmc_args, model_args),
            ModelType::Linear => {
                unimplemented!("Training linear models is currently not supported.")
            }
        },
        SubCmd::Train {
            input_args,
            mcmc_args,
            model_args,
        } => match model_args.model_type {
            ModelType::LassoBase => train::<LassoBaseBranch>(input_args, mcmc_args, model_args),
            ModelType::LassoARD => train::<LassoArdBranch>(input_args, mcmc_args, model_args),
            ModelType::RidgeBase => train::<RidgeBaseBranch>(input_args, mcmc_args, model_args),
            ModelType::RidgeARD => train::<RidgeArdBranch>(input_args, mcmc_args, model_args),
            ModelType::StdNormal => train::<StdNormalBranch>(input_args, mcmc_args, model_args),
            ModelType::Linear => {
                unimplemented!("Training linear models is currently not supported.")
            }
        },
        SubCmd::GroupByLD(args) => group_centered(args),
        SubCmd::Activations(args) => activations(args),
        SubCmd::Predict(args) => predict(args),
        SubCmd::BranchR2(args) => branch_r2(args),
        SubCmd::AvailableBackends => available_backends(),
        SubCmd::GroupByGenes(args) => group_by_genes(args),
        SubCmd::Gradients(args) => gradients(args),
    }
    exit(exitcode::OK);
}

fn group_by_genes(args: GroupByGenesArgs) {
    let bim_path = PathBuf::from(&args.bim);
    let gff_path = PathBuf::from(&args.gff);
    let mut outpath = Path::new(&args.outdir).join(bim_path.file_stem().unwrap());
    outpath.set_extension("gene_grouping");
    let grouping = GeneGrouping::from_gff(&gff_path, &bim_path, args.margin, args.min_group_size);
    grouping.to_file(&outpath);
    outpath.set_extension("gene_grouping_meta");
    grouping.meta_to_file(&outpath);
}

fn group_centered(args: GroupCenteredArgs) {
    let mut bim_path = PathBuf::from(&args.inpath);
    bim_path.set_extension("bim");
    let mut corr_path = PathBuf::from(&args.inpath);
    corr_path.set_extension("ld");
    let mut outpath = Path::new(&args.outdir).join(bim_path.file_stem().unwrap());
    outpath.set_extension("centered_grouping");
    CorrGraph::from_plink_ld(&corr_path, &bim_path)
        .centered_grouping(args.min_group_size)
        .to_file(&outpath);
}

fn available_backends() {
    println!("{:?}", arrayfire::get_available_backends());
}

fn branch_r2(args: BranchR2Args) {
    let bed = BedVM::from_file(Path::new(&args.bfile));
    let groups = ExternalGrouping::from_file(Path::new(&args.groups));
    let genotypes = CompressedGenotypes::new(bed, groups);
    let phenotypes =
        Phenotypes::from_file(Path::new(&args.phen)).expect("Failed to load phenotype input data");
    let data = Data::new(genotypes, phenotypes);
    // get model type
    let parent_path = Path::new(&args.model_path)
        .parent()
        .unwrap()
        .join("args.json");
    let train_args = TrainNewModelArgs::from_file(&parent_path);
    let model_type = train_args.model_type;
    // stdout writer in csv format
    let mut wtr = csv::Writer::from_writer(std::io::stdout());

    // load models and compute r2
    let mut model_files = read_dir(Path::new(&args.model_path))
        .expect("Failed to parse model dir")
        .map(|res| res.map(|e| e.path()))
        .filter_map(|e| e.ok())
        .filter(|e| e.is_file())
        .collect::<Vec<PathBuf>>();
    model_files.sort();

    for path in model_files {
        let r2s = match model_type {
            ModelType::RidgeARD => Net::<RidgeArdBranch>::from_file(&path).branch_r2s(&data),
            ModelType::LassoARD => Net::<LassoArdBranch>::from_file(&path).branch_r2s(&data),
            ModelType::RidgeBase => Net::<RidgeBaseBranch>::from_file(&path).branch_r2s(&data),
            ModelType::LassoBase => Net::<LassoBaseBranch>::from_file(&path).branch_r2s(&data),
            ModelType::StdNormal => Net::<StdNormalBranch>::from_file(&path).branch_r2s(&data),
            ModelType::Linear => unimplemented!("Linear models are currently not supported."),
        };
        wtr.write_record(r2s.iter().map(|e| e.to_string())).unwrap();
    }
    wtr.flush().expect("Failed to flush csv writer");
}

fn read_model_type_from_model_args(path: &Path) -> ModelType {
    let text = std::fs::read_to_string(path).unwrap();
    // Parse the string into a dynamically-typed JSON structure.
    let json = serde_json::from_str::<serde_json::Value>(&text).unwrap();
    ModelType::from_str(json["model_type"].as_str().unwrap()).unwrap()
}

fn activations(args: ActivationArgs) {
    let bed = BedVM::from_file(Path::new(&args.bfile));
    let groups = ExternalGrouping::from_file(Path::new(&args.groups));
    let genotypes = CompressedGenotypes::new(bed, groups);

    // get model type
    let args_path = Path::new(&args.model_path)
        .parent()
        .unwrap()
        .join("args.json");
    let model_type = read_model_type_from_model_args(&args_path);

    let outdir = Path::new(&args.model_path)
        .parent()
        .unwrap()
        .join("activations");
    if !Path::new(&outdir).exists() {
        std::fs::create_dir_all(&outdir).expect("Could not create output directory!");
    }

    let mut model_files = read_dir(Path::new(&args.model_path))
        .expect("Failed to parse model dir")
        .map(|res| res.map(|e| e.path()))
        .filter_map(|e| e.ok())
        .filter(|e| e.is_file())
        .collect::<Vec<PathBuf>>();
    model_files.sort();

    for path in model_files {
        let activations = match model_type {
            ModelType::RidgeARD => Net::<RidgeArdBranch>::from_file(&path).activations(&genotypes),
            ModelType::LassoARD => Net::<LassoArdBranch>::from_file(&path).activations(&genotypes),
            ModelType::RidgeBase => {
                Net::<RidgeBaseBranch>::from_file(&path).activations(&genotypes)
            }
            ModelType::LassoBase => {
                Net::<LassoBaseBranch>::from_file(&path).activations(&genotypes)
            }
            ModelType::StdNormal => {
                Net::<StdNormalBranch>::from_file(&path).activations(&genotypes)
            }
            ModelType::Linear => unimplemented!("Linear models are currently not supported."),
        };
        let outfile = outdir.join(format!(
            "{}.json",
            path.file_stem().unwrap().to_str().unwrap()
        ));
        activations.to_json(&outfile);
    }
}

fn gradients(args: GradientsArgs) {
    let bed = BedVM::from_file(Path::new(&args.bfile));
    let groups = ExternalGrouping::from_file(Path::new(&args.groups));
    let genotypes = CompressedGenotypes::new(bed, groups);
    let phenotypes =
        Phenotypes::from_file(Path::new(&args.phen)).expect("Failed to load phenotype input data");
    let data = Data::new(genotypes, phenotypes);

    // get model type
    let args_path = Path::new(&args.model_path)
        .parent()
        .unwrap()
        .join("args.json");
    let model_type = read_model_type_from_model_args(&args_path);

    let outdir = Path::new(&args.model_path)
        .parent()
        .unwrap()
        .join("gradients");

    if !Path::new(&outdir).exists() {
        std::fs::create_dir_all(&outdir).expect("Could not create output directory!");
    }

    let mut model_files = read_dir(Path::new(&args.model_path))
        .expect("Failed to parse model dir")
        .map(|res| res.map(|e| e.path()))
        .filter_map(|e| e.ok())
        .filter(|e| e.is_file())
        .collect::<Vec<PathBuf>>();
    model_files.sort();

    for path in model_files {
        let gradient = match model_type {
            ModelType::RidgeARD => Net::<RidgeArdBranch>::from_file(&path).gradient(&data),
            ModelType::LassoARD => Net::<LassoArdBranch>::from_file(&path).gradient(&data),
            ModelType::RidgeBase => Net::<RidgeBaseBranch>::from_file(&path).gradient(&data),
            ModelType::LassoBase => Net::<LassoBaseBranch>::from_file(&path).gradient(&data),
            ModelType::StdNormal => Net::<StdNormalBranch>::from_file(&path).gradient(&data),
            ModelType::Linear => unimplemented!("Linear models are currently not supported."),
        };
        let outfile = outdir.join(format!(
            "{}.json",
            path.file_stem().unwrap().to_str().unwrap()
        ));
        to_writer(File::create(outfile).unwrap(), &gradient)
            .expect("Failed to write activations to json");
    }
}

fn predict(args: PredictArgs) {
    let bed = BedVM::from_file(Path::new(&args.bfile));
    let groups = ExternalGrouping::from_file(Path::new(&args.groups));
    let genotypes = CompressedGenotypes::new(bed, groups);

    // get model type
    let args_path = Path::new(&args.model_path)
        .parent()
        .unwrap()
        .join("args.json");
    let model_type = read_model_type_from_model_args(&args_path);
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
            ModelType::RidgeARD => Net::<RidgeArdBranch>::from_file(&path).predict(&genotypes),
            ModelType::LassoARD => Net::<LassoArdBranch>::from_file(&path).predict(&genotypes),
            ModelType::RidgeBase => Net::<RidgeBaseBranch>::from_file(&path).predict(&genotypes),
            ModelType::LassoBase => Net::<LassoBaseBranch>::from_file(&path).predict(&genotypes),
            ModelType::StdNormal => Net::<StdNormalBranch>::from_file(&path).predict(&genotypes),
            ModelType::Linear => unimplemented!("Linear models are currently not supported."),
        };
        wtr.write_record(prediction.iter().map(|e| e.to_string()))
            .unwrap();
    }
    wtr.flush().expect("Failed to flush csv writer");
}

fn simulate_y<B>(args: SimulateYArgs)
where
    B: BranchSampler,
{
    if args.debug {
        simple_logger::init_with_level(log::Level::Debug).unwrap();
    } else {
        simple_logger::init_with_level(log::Level::Info).unwrap();
    }

    if !(args.heritability >= 0. && args.heritability <= 1.) {
        panic!("Heritability must be within [0, 1].");
    }

    let mut outdir = format!(
        "{}_{}_d{}_h{}",
        args.model_type, args.activation_function, args.depth, args.heritability
    );

    if let Some(n) = args.num_effective {
        outdir.push_str(&format!("_me{:?}", n));
    } else if let Some(p) = args.proportion_effective {
        outdir.push_str(&format!("_pe{:?}", p));
    }

    if let Some(v) = args.init_param_variance {
        outdir.push_str(&format!("_v{:?}", v));
    } else if let (Some(k), Some(s)) = (args.init_gamma_shape, args.init_gamma_scale) {
        outdir.push_str(&format!("_k{:?}", k));
        outdir.push_str(&format!("_s{:?}", s));
    }

    let path = set_replicate_ix(&args.outdir, &outdir);
    create_outdir(&path);

    let mut train_path = path.join("train");
    let mut test_path = path.join("test");
    let args_path = path.join("args.json");
    let params_path = path.join("model.params");
    let model_path = path.join("model.bin");

    info!("Loading genotype data");
    let gen_train = CompressedGenotypes::new(
        BedVM::from_file(Path::new(&args.bfile_train)),
        ExternalGrouping::from_file(Path::new(&args.groups)),
    );

    let gen_test = CompressedGenotypes::new(
        BedVM::from_file(Path::new(&args.bfile_test)),
        ExternalGrouping::from_file(Path::new(&args.groups)),
    );

    // TODO: depth should be set such that the largest branch still has n > p.
    // Although, that is for models I want to train. For simulation it doesn't
    // matter I guess.
    info!("Building model");
    let mut net_cfg = BlockNetCfg::<B>::new()
        .with_num_effective_markers(args.num_effective)
        .with_proportion_effective_markers(args.proportion_effective)
        .with_num_hidden_layers(args.depth)
        .with_hidden_layer_width_rule(HiddenLayerWidthRule::FractionOfInput(0.5))
        .with_summary_layer_width_rule(SummaryLayerWidthRule::LikeHiddenLayerWidth)
        .with_activation_function(args.activation_function);
    net_cfg = if let Some(v) = args.init_param_variance {
        net_cfg.with_init_param_variance(v)
    } else if let (Some(k), Some(s)) = (args.init_gamma_shape, args.init_gamma_scale) {
        net_cfg
            .with_init_gamma_params(k, s)
            .with_dense_precision_prior(k, s)
            .with_summary_precision_prior(k, s)
            // this is Gamma(1, 1) because at the moment the output variance
            // is hardcoded to 1. TODO: this should be configurable.
            .with_output_precision_prior(1., 1.)
    } else {
        net_cfg
    };

    // width is fixed to half the number of input nodes
    for &size in gen_test.num_markers_per_group() {
        net_cfg.add_branch(size);
    }
    let net = net_cfg.build_net();

    info!("Saving model");
    net.to_file(&model_path);

    info!("Saving model params");
    info!("Creating: {:?}", params_path);
    let mut net_params_file = BufWriter::new(File::create(params_path).unwrap());
    to_writer(&mut net_params_file, net.branch_cfgs()).unwrap();
    net_params_file.write_all(b"\n").unwrap();

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

    PhenStats::new(
        (&y_test.iter().map(|e| *e as f64).collect::<Vec<f64>>()).mean() as f32,
        (&y_test.iter().map(|e| *e as f64).collect::<Vec<f64>>()).variance() as f32,
        test_residual_variance as f32,
    )
    .to_file(&path.join("test_phen_stats.json"));

    PhenStats::new(
        (&y_train.iter().map(|e| *e as f64).collect::<Vec<f64>>()).mean() as f32,
        (&y_train.iter().map(|e| *e as f64).collect::<Vec<f64>>()).variance() as f32,
        train_residual_variance as f32,
    )
    .to_file(&path.join("train_phen_stats.json"));

    train_path.set_extension("phen");
    let phen_train = Phenotypes::new(y_train);
    phen_train.to_file(&train_path);

    test_path.set_extension("phen");
    let phen_test = Phenotypes::new(y_test);
    phen_test.to_file(&test_path);

    if args.json_data {
        phen_train.to_json(&path.join("phen_train.json"));
        phen_test.to_json(&path.join("phen_test.json"));
    }

    args.to_file(&args_path);
}

fn simulate_y_linear(args: SimulateYArgs) {
    if args.debug {
        simple_logger::init_with_level(log::Level::Debug).unwrap();
    } else {
        simple_logger::init_with_level(log::Level::Info).unwrap();
    }

    if !(args.heritability >= 0. && args.heritability <= 1.) {
        panic!("Heritability must be within [0, 1].");
    }

    let mut outdir = format!("{}_h{}", args.model_type, args.heritability);

    if let Some(n) = args.num_effective {
        outdir.push_str(&format!("_me{:?}", n));
    } else if let Some(p) = args.proportion_effective {
        outdir.push_str(&format!("_pe{:?}", p));
    }

    if let Some(v) = args.init_param_variance {
        outdir.push_str(&format!("_v{:?}", v));
    } else if let (Some(k), Some(s)) = (args.init_gamma_shape, args.init_gamma_scale) {
        outdir.push_str(&format!("_k{:?}", k));
        outdir.push_str(&format!("_s{:?}", s));
    }

    let path = set_replicate_ix(&args.outdir, &outdir);
    create_outdir(&path);
    let mut train_path = path.join("train");
    let mut test_path = path.join("test");
    let args_path = path.join("args.json");
    let params_path = path.join("model.params");
    let _model_path = path.join("model.bin");

    let mut rng = thread_rng();

    info!("Loading genotype data");
    let gen_train = CompressedGenotypes::new(
        BedVM::from_file(Path::new(&args.bfile_train)),
        ExternalGrouping::from_file(Path::new(&args.groups)),
    );

    let gen_test = CompressedGenotypes::new(
        BedVM::from_file(Path::new(&args.bfile_test)),
        ExternalGrouping::from_file(Path::new(&args.groups)),
    );

    info!("Building model");
    let lm = LinearModelBuilder::new(gen_test.num_markers_per_group())
        .with_num_effective_markers(args.num_effective)
        .with_proportion_effective_markers(args.proportion_effective)
        .with_random_effects(args.heritability)
        .build();

    if args.debug {
        debug!("sum of squared effects: {:?}", lm.sum_of_squares());
    }

    info!("Making phenotype data");
    // genetic values
    let g_train = lm.predict(&gen_train);
    let g_var_train = g_train
        .iter()
        .map(|e| *e as f64)
        .collect::<Vec<f64>>()
        .variance() as f32;
    let tot_var_train = g_var_train / args.heritability;
    let e_var_train = tot_var_train - g_var_train;
    info!("Genetic variance in train: {}", g_var_train);
    let mut y_train = g_train.clone();
    let g_test = lm.predict(&gen_test);
    let g_var_test = g_test
        .iter()
        .map(|e| *e as f64)
        .collect::<Vec<f64>>()
        .variance() as f32;
    let tot_var_test = g_var_test / args.heritability;
    let e_var_test = tot_var_test - g_var_test;
    info!("Genetic variance in test: {}", g_var_test);
    let mut y_test = g_test.clone();

    let std_e_train: f32 = e_var_train.sqrt();
    let std_e_test: f32 = e_var_test.sqrt();

    let std_e_norm_train = Normal::new(0.0, std_e_train).unwrap();
    (0..gen_train.num_individuals()).for_each(|i| y_train[i] += std_e_norm_train.sample(&mut rng));
    let std_e_norm_test = Normal::new(0.0, std_e_test).unwrap();
    (0..gen_test.num_individuals()).for_each(|i| y_test[i] += std_e_norm_test.sample(&mut rng));

    info!("Saving model params");
    info!("Creating: {:?}", &params_path);
    let mut net_params_file = BufWriter::new(File::create(&params_path).unwrap());
    to_writer(&mut net_params_file, &lm).unwrap();
    net_params_file.write_all(b"\n").unwrap();

    PhenStats::new(
        (&y_test.iter().map(|e| *e as f64).collect::<Vec<f64>>()).mean() as f32,
        (&y_test.iter().map(|e| *e as f64).collect::<Vec<f64>>()).variance() as f32,
        e_var_test,
    )
    .to_file(&path.join("test_phen_stats.json"));

    PhenStats::new(
        (&y_train.iter().map(|e| *e as f64).collect::<Vec<f64>>()).mean() as f32,
        (&y_train.iter().map(|e| *e as f64).collect::<Vec<f64>>()).variance() as f32,
        e_var_train,
    )
    .to_file(&path.join("train_phen_stats.json"));

    train_path.set_extension("phen");
    let phen_train = Phenotypes::new(y_train);
    phen_train.to_file(&train_path);

    test_path.set_extension("phen");
    let phen_test = Phenotypes::new(y_test);
    phen_test.to_file(&test_path);

    if args.json_data {
        Phenotypes::new(g_test).to_json(&path.join("genetic_values_test.json"));
        Phenotypes::new(g_train).to_json(&path.join("genetic_values_train.json"));
        phen_train.to_json(&path.join("phen_train.json"));
        phen_test.to_json(&path.join("phen_test.json"));
    }

    args.to_file(&args_path);
}

fn simulate_xy_linear(args: SimulateXYArgs) {
    simple_logger::init_with_level(log::Level::Info).unwrap();

    if !(args.heritability >= 0. && args.heritability <= 1.) {
        panic!("Heritability must be within [0, 1].");
    }

    let mut outdir = format!(
        "{}_b{}_wh{}_ws{}_d{}_m{}_n{}_h{}",
        args.model_type,
        args.num_branches,
        args.hidden_layer_width,
        args.summary_layer_width.unwrap_or(args.hidden_layer_width),
        args.branch_depth,
        args.num_markers_per_branch,
        args.num_individuals,
        args.heritability
    );

    if let Some(n) = args.num_effective {
        outdir.push_str(&format!("_me{:?}", n));
    } else if let Some(p) = args.proportion_effective {
        outdir.push_str(&format!("_pe{:?}", p));
    }

    if let Some(v) = args.init_param_variance {
        outdir.push_str(&format!("v_{:?}", v));
    } else if let (Some(k), Some(s)) = (args.init_gamma_shape, args.init_gamma_scale) {
        outdir.push_str(&format!("k_{:?}", k));
        outdir.push_str(&format!("s_{:?}", s));
    }

    let path = set_replicate_ix(&args.outdir, &outdir);
    create_outdir(&path);
    let train_path = path.join("train");
    let test_path = path.join("test");
    let args_path = path.join("args.json");
    let params_path = path.join("model.params");

    let mut rng = thread_rng();

    info!("Generating random marker data");
    let ud = Uniform::from(0.0..0.5);
    let mafs = (0..args.num_branches * args.num_markers_per_branch)
        .map(|_| ud.sample(&mut rng))
        .collect::<Vec<f32>>();

    let groups = UniformGrouping::new(args.num_branches, args.num_markers_per_branch);
    let num_markers = args.num_branches * args.num_markers_per_branch;
    let bed_train = BedVM::random(args.num_individuals, num_markers, Some(mafs.clone()), None);
    let bed_test = BedVM::random(args.num_individuals, num_markers, Some(mafs), None);
    let gen_train = CompressedGenotypes::new(bed_train, groups.clone());
    let gen_test = CompressedGenotypes::new(bed_test, groups);

    let lm = LinearModelBuilder::new(&vec![args.num_markers_per_branch; args.num_branches])
        .with_num_effective_markers(args.num_effective)
        .with_proportion_effective_markers(args.proportion_effective)
        .with_random_effects(args.heritability)
        .build();

    info!("Making phenotype data");
    // genetic values
    let g_train = lm.predict(&gen_train);
    let g_var_train = g_train
        .iter()
        .map(|e| *e as f64)
        .collect::<Vec<f64>>()
        .variance() as f32;
    let tot_var_train = g_var_train / args.heritability;
    let e_var_train = tot_var_train - g_var_train;
    info!("Genetic variance in train: {}", g_var_train);
    let mut y_train = g_train.clone();
    let g_test = lm.predict(&gen_test);
    let g_var_test = g_test
        .iter()
        .map(|e| *e as f64)
        .collect::<Vec<f64>>()
        .variance() as f32;
    let tot_var_test = g_var_test / args.heritability;
    let e_var_test = tot_var_test - g_var_test;
    info!("Genetic variance in test: {}", g_var_test);
    let mut y_test = g_test.clone();

    let std_e_train: f32 = e_var_train.sqrt();
    let std_e_test: f32 = e_var_test.sqrt();

    let std_e_norm_train = Normal::new(0.0, std_e_train).unwrap();
    (0..args.num_individuals).for_each(|i| y_train[i] += std_e_norm_train.sample(&mut rng));
    let std_e_norm_test = Normal::new(0.0, std_e_test).unwrap();
    (0..args.num_individuals).for_each(|i| y_test[i] += std_e_norm_test.sample(&mut rng));

    info!("Saving model params");
    info!("Creating: {:?}", &params_path);
    let mut net_params_file = BufWriter::new(File::create(&params_path).unwrap());
    to_writer(&mut net_params_file, &lm).unwrap();
    net_params_file.write_all(b"\n").unwrap();

    gen_train.to_file(&train_path);
    gen_test.to_file(&test_path);

    PhenStats::new(
        (&y_test.iter().map(|e| *e as f64).collect::<Vec<f64>>()).mean() as f32,
        (&y_test.iter().map(|e| *e as f64).collect::<Vec<f64>>()).variance() as f32,
        e_var_test,
    )
    .to_file(&path.join("test_phen_stats.json"));

    PhenStats::new(
        (&y_train.iter().map(|e| *e as f64).collect::<Vec<f64>>()).mean() as f32,
        (&y_train.iter().map(|e| *e as f64).collect::<Vec<f64>>()).variance() as f32,
        e_var_train,
    )
    .to_file(&path.join("train_phen_stats.json"));

    let phen_train = Phenotypes::new(y_train);
    phen_train.to_file(&train_path.with_extension("phen"));

    let phen_test = Phenotypes::new(y_test);
    phen_test.to_file(&test_path.with_extension("phen"));

    if args.json_data {
        Phenotypes::new(g_test).to_json(&path.join("genetic_values_test.json"));
        Phenotypes::new(g_train).to_json(&path.join("genetic_values_train.json"));
        phen_train.to_json(&path.join("phen_train.json"));
        phen_test.to_json(&path.join("phen_test.json"));
    }

    args.to_file(&args_path);
}

fn set_replicate_ix(parent_dir: &String, outdir: &String) -> PathBuf {
    let mut rep_ix = 1;
    loop {
        let mut od_cp = outdir.clone();
        od_cp.push_str(&format!("_rep{:?}", rep_ix));
        let p = Path::new(parent_dir).join(od_cp);
        if !p.exists() {
            return p;
        }
        rep_ix += 1;
    }
}

fn create_outdir(dir: &PathBuf) {
    std::fs::create_dir_all(&dir).expect("Could not create output directory!");
}

fn simulate_xy<B>(args: SimulateXYArgs)
where
    B: BranchSampler,
{
    simple_logger::init_with_level(log::Level::Debug).unwrap();

    if !(args.heritability >= 0. && args.heritability <= 1.) {
        panic!("Heritability must be within [0, 1].");
    }

    let mut outdir = format!(
        "{}_{}_b{}_wh{}_ws{}_d{}_m{}_n{}_h{}",
        args.model_type,
        args.activation_function,
        args.num_branches,
        args.hidden_layer_width,
        args.summary_layer_width.unwrap_or(args.hidden_layer_width),
        args.branch_depth,
        args.num_markers_per_branch,
        args.num_individuals,
        args.heritability
    );

    if let Some(n) = args.num_effective {
        outdir.push_str(&format!("_me{:?}", n));
    } else if let Some(p) = args.proportion_effective {
        outdir.push_str(&format!("_pe{:?}", p));
    }

    if let Some(v) = args.init_param_variance {
        outdir.push_str(&format!("v_{:?}", v));
    } else if let (Some(k), Some(s)) = (args.init_gamma_shape, args.init_gamma_scale) {
        outdir.push_str(&format!("k_{:?}", k));
        outdir.push_str(&format!("s_{:?}", s));
    }

    let path = set_replicate_ix(&args.outdir, &outdir);
    create_outdir(&path);

    let train_path = path.join("train");
    let test_path = path.join("test");
    let args_path = path.join("args.json");
    let params_path = path.join("model.params");
    let model_path = path.join("model.bin");

    let mut train_residual_variance: f64 = 0.0;
    let mut test_residual_variance: f64 = 0.0;

    loop {
        info!("Building model");
        let slwr = if let Some(width) = args.summary_layer_width {
            SummaryLayerWidthRule::Fixed(width)
        } else {
            SummaryLayerWidthRule::LikeHiddenLayerWidth
        };

        let mut net_cfg = BlockNetCfg::<B>::new()
            .with_num_hidden_layers(args.branch_depth)
            .with_num_effective_markers(args.num_effective)
            .with_proportion_effective_markers(args.proportion_effective)
            .with_hidden_layer_width_rule(HiddenLayerWidthRule::Fixed(args.hidden_layer_width))
            .with_summary_layer_width_rule(slwr)
            .with_activation_function(args.activation_function);

        net_cfg = if let (Some(k), Some(s)) = (args.init_gamma_shape, args.init_gamma_scale) {
            net_cfg
                .with_init_gamma_params(k, s)
                .with_dense_precision_prior(k, s)
                .with_summary_precision_prior(k, s)
                .with_output_precision_prior(1., 1.)
        } else if let Some(v) = args.init_param_variance {
            net_cfg.with_init_param_variance(v)
        } else {
            net_cfg
        };

        for _ in 0..args.num_branches {
            net_cfg.add_branch(args.num_markers_per_branch);
        }
        let net = net_cfg.build_net();

        let mut rng = thread_rng();

        info!("Generating random marker data");
        let ud = Uniform::from(0.0..0.5);
        let mafs = (0..args.num_branches * args.num_markers_per_branch)
            .map(|_| ud.sample(&mut rng))
            .collect::<Vec<f32>>();

        let groups = UniformGrouping::new(args.num_branches, args.num_markers_per_branch);
        let num_markers = args.num_branches * args.num_markers_per_branch;
        let bed_train = BedVM::random(args.num_individuals, num_markers, Some(mafs.clone()), None);
        let bed_test = BedVM::random(args.num_individuals, num_markers, Some(mafs), None);
        let gen_train = CompressedGenotypes::new(bed_train, groups.clone());
        let gen_test = CompressedGenotypes::new(bed_test, groups);

        info!("Making phenotype data");
        // genetic values
        let g_train = net.predict(&gen_train);
        let mut y_train = g_train.clone();
        let g_test = net.predict(&gen_test);
        let mut y_test = g_test.clone();

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

            // TODO: let user set minimal residual variance
            if train_residual_variance < 0.01 || test_residual_variance < 0.01 {
                debug!(
                    "Residual variances too small; train: {:?}, test: {:?}",
                    train_residual_variance, test_residual_variance
                );
                continue;
            }
        }

        info!("Saving model");
        net.to_file(&model_path);

        info!("Saving model params");
        info!("Creating: {:?}", &params_path);
        let mut net_params_file = BufWriter::new(File::create(&params_path).unwrap());
        to_writer(&mut net_params_file, net.branch_cfgs()).unwrap();
        net_params_file.write_all(b"\n").unwrap();

        gen_train.to_file(&train_path);
        gen_test.to_file(&test_path);

        PhenStats::new(
            (&y_test.iter().map(|e| *e as f64).collect::<Vec<f64>>()).mean() as f32,
            (&y_test.iter().map(|e| *e as f64).collect::<Vec<f64>>()).variance() as f32,
            test_residual_variance as f32,
        )
        .to_file(&path.join("test_phen_stats.json"));

        PhenStats::new(
            (&y_train.iter().map(|e| *e as f64).collect::<Vec<f64>>()).mean() as f32,
            (&y_train.iter().map(|e| *e as f64).collect::<Vec<f64>>()).variance() as f32,
            train_residual_variance as f32,
        )
        .to_file(&path.join("train_phen_stats.json"));

        let phen_train = Phenotypes::new(y_train);
        phen_train.to_file(&train_path.with_extension("phen"));

        let phen_test = Phenotypes::new(y_test);
        phen_test.to_file(&test_path.with_extension("phen"));

        if args.json_data {
            Phenotypes::new(g_test).to_json(&path.join("genetic_values_test.json"));
            Phenotypes::new(g_train).to_json(&path.join("genetic_values_train.json"));
            phen_train.to_json(&path.join("phen_train.json"));
            phen_test.to_json(&path.join("phen_test.json"));
        }

        args.to_file(&args_path);

        break;
    }
}

fn load_ungrouped_data(
    args: &TrainIOArgs,
) -> (
    Data<CompressedGenotypes<ExternalGrouping>>,
    Option<Data<CompressedGenotypes<ExternalGrouping>>>,
) {
    let gen_train = CompressedGenotypes::new(
        BedVM::from_file(Path::new(&args.bfile_train)),
        ExternalGrouping::from_file(Path::new(&args.groups)),
    );

    let train_phen = Phenotypes::from_file(Path::new(&args.p_train))
        .expect("Failed to load train.phen training phenotypes");

    let train_data = Data::new(gen_train, train_phen);

    let gen_test = args.bfile_test.as_ref().map(|bfile| {
        CompressedGenotypes::new(
            BedVM::from_file(Path::new(&bfile)),
            ExternalGrouping::from_file(Path::new(&args.groups)),
        )
    });

    debug!("Trying to load test phen at: {:?}", args.p_test);

    let test_phen = args
        .p_test
        .as_ref()
        .map(|pfile| Phenotypes::from_file(Path::new(&pfile)));

    let test_data = if let (Some(tg), Some(tp)) = (gen_test, test_phen) {
        // just checked that they are Some
        Some(Data::new(tg, tp.expect("Failed to load test phenotypes")))
    } else {
        info!("No complete test data provided, proceeding without");
        None
    };
    (train_data, test_data)
}

fn train_new<B>(input_args: TrainIOArgs, mcmc_args: MCMCArgs, model_args: TrainNewModelArgs)
where
    B: BranchSampler,
{
    if mcmc_args.debug_prints {
        simple_logger::init_with_level(log::Level::Debug).unwrap();
    } else {
        simple_logger::init_with_level(log::Level::Info).unwrap();
    }

    info!("Loading data.");
    let (train_data, test_data) = load_ungrouped_data(&input_args);

    let mut outdir = format!(
        "{}_{}_d{}_cl{}_il{}_{}_st{}_dpk{}_dps{}_spk{}_sps{}_opk{}_ops{}",
        model_args.model_type,
        model_args.activation_function,
        model_args.branch_depth,
        mcmc_args.chain_length,
        mcmc_args.integration_length,
        mcmc_args.step_size_mode,
        mcmc_args.step_size,
        model_args.dpk,
        model_args.dps,
        model_args.spk,
        model_args.sps,
        model_args.opk,
        model_args.ops,
    );

    if mcmc_args.joint_hmc {
        outdir.push_str("_joint");
    }

    if mcmc_args.gradient_descent {
        outdir.push_str("_gd");
    }

    if mcmc_args.gradient_descent_joint {
        outdir.push_str("_gdj");
    }

    if let Some(v) = mcmc_args.fixed_param_precision {
        outdir.push_str(&format!("_fp{}", v));
    }

    let hlwr = if let Some(width) = model_args.fixed_hidden_layer_width {
        outdir.push_str(&format!("_fhlw{}", width));
        HiddenLayerWidthRule::Fixed(width)
    } else {
        outdir.push_str(&format!("_rhlw{}", model_args.relative_hidden_layer_width));
        HiddenLayerWidthRule::FractionOfInput(model_args.relative_hidden_layer_width)
    };

    let slwr = if let Some(width) = model_args.fixed_summary_layer_width {
        outdir.push_str(&format!("_fslw{}", width));
        SummaryLayerWidthRule::Fixed(width)
    } else {
        outdir.push_str(&format!("_rslw{}", model_args.relative_summary_layer_width));
        SummaryLayerWidthRule::FractionOfHiddenLayerWidth(model_args.relative_summary_layer_width)
    };

    let model_path = set_replicate_ix(&input_args.outpath, &outdir);

    let mcmc_cfg = MCMCCfgBuilder::default()
        .with_hmc_step_size_factor(mcmc_args.step_size)
        .with_hmc_max_hamiltonian_error(mcmc_args.max_hamiltonian_error)
        .with_hmc_integration_length(mcmc_args.integration_length)
        .with_hmc_step_size_mode(mcmc_args.step_size_mode.clone())
        .with_chain_length(mcmc_args.chain_length)
        .with_burn_in(mcmc_args.burn_in)
        .with_outpath(model_path.into_os_string().into_string().unwrap())
        .with_trace(mcmc_args.trace)
        .with_trajectories(mcmc_args.trajectories)
        .with_num_grad_traj(mcmc_args.num_grad_traj)
        .with_num_grad(mcmc_args.num_grad)
        .with_gradient_descent(mcmc_args.gradient_descent)
        .with_gradient_descent_joint(mcmc_args.gradient_descent_joint)
        .with_joint_hmc(mcmc_args.joint_hmc)
        .with_fixed_param_precisions(mcmc_args.fixed_param_precision.is_some())
        .build();
    mcmc_cfg.create_out();

    model_args.to_file(&mcmc_cfg.args_path());

    let report_cfg = ReportCfg::new(mcmc_args.report_interval, test_data.as_ref());

    info!("Building net");
    let mut net_cfg = BlockNetCfg::<B>::new()
        .with_num_hidden_layers(model_args.branch_depth)
        .with_dense_precision_prior(model_args.dpk, model_args.dps)
        .with_summary_precision_prior(model_args.spk, model_args.sps)
        .with_output_precision_prior(model_args.opk, model_args.ops)
        .with_hidden_layer_width_rule(hlwr)
        .with_summary_layer_width_rule(slwr)
        .with_fixed_param_precision(mcmc_args.fixed_param_precision)
        .with_activation_function(model_args.activation_function);

    for bix in 0..train_data.num_branches() {
        net_cfg.add_branch(train_data.num_markers_in_branch(bix));
    }
    let mut net = net_cfg.build_net();
    // if let Some(p) = args.error_precision {
    //     net.set_error_precision(p);
    // }

    for bix in 0..net.num_branches() {
        if net.num_branch_params(bix) > train_data.num_individuals() {
            warn!(
                "Num params > num individuals in branch {} (with {} params, {} individuals)",
                bix,
                net.num_branch_params(bix),
                train_data.num_individuals()
            );
        }
    }
    net.write_hyperparams(&mcmc_cfg);

    info!("Training net");
    if net.num_branches() == 1 {
        info!("Using single branch training procedure");
        net.train_single_branch(&train_data, &mcmc_cfg, true, Some(report_cfg));
    } else {
        net.train(&train_data, &mcmc_cfg, true, Some(report_cfg));
    }
}

fn train<B>(input_args: TrainIOArgs, mcmc_args: MCMCArgs, model_args: TrainOldModelArgs)
where
    B: BranchSampler,
{
    if mcmc_args.debug_prints {
        simple_logger::init_with_level(log::Level::Debug).unwrap();
    } else {
        simple_logger::init_with_level(log::Level::Info).unwrap();
    }

    info!("Loading data");
    let (train_data, test_data) = load_ungrouped_data(&input_args);

    let model_path = Path::new(&model_args.model_file);
    if !model_path.is_file() {
        log::error!("Specified model: No such file found");
        exit(exitcode::NOINPUT);
    }
    assert!(model_path.is_file(),);

    let mut outdir = format!(
        "{}_cl{}_il{}_{}_st{}_dtheta{}_dlambda{}",
        model_path.file_stem().unwrap().to_string_lossy(),
        mcmc_args.chain_length,
        mcmc_args.integration_length,
        mcmc_args.step_size_mode,
        mcmc_args.step_size,
        model_args.perturb_params.unwrap_or(0.),
        model_args.perturb_precisions.unwrap_or(0.)
    );

    if mcmc_args.joint_hmc {
        outdir.push_str("_joint");
    }

    if mcmc_args.gradient_descent {
        outdir.push_str("_gd");
    }

    if mcmc_args.gradient_descent_joint {
        outdir.push_str("_gdj");
    }

    if mcmc_args.fixed_param_precision.is_some() {
        outdir.push_str("_fp");
    }

    let mcmc_cfg = MCMCCfgBuilder::default()
        .with_hmc_step_size_factor(mcmc_args.step_size)
        .with_hmc_max_hamiltonian_error(mcmc_args.max_hamiltonian_error)
        .with_hmc_integration_length(mcmc_args.integration_length)
        .with_hmc_step_size_mode(mcmc_args.step_size_mode.clone())
        .with_chain_length(mcmc_args.chain_length)
        .with_burn_in(mcmc_args.burn_in)
        .with_outpath(outdir)
        .with_trace(mcmc_args.trace)
        .with_trajectories(mcmc_args.trajectories)
        .with_num_grad_traj(mcmc_args.num_grad_traj)
        .with_num_grad(mcmc_args.num_grad)
        .with_gradient_descent(mcmc_args.gradient_descent)
        .with_gradient_descent_joint(mcmc_args.gradient_descent_joint)
        .with_joint_hmc(mcmc_args.joint_hmc)
        .with_fixed_param_precisions(mcmc_args.fixed_param_precision.is_some())
        .build();
    mcmc_cfg.create_out();

    model_args.to_file(&mcmc_cfg.args_path());

    let report_cfg = ReportCfg::new(mcmc_args.report_interval, test_data.as_ref());

    info!("Loading net");

    let mut net = Net::<B>::from_file(model_path);
    net.perturb(model_args.perturb_params, model_args.perturb_precisions);
    // if let Some(p) = model_args.error_precision {
    //     net.set_error_precision(p);
    // }
    net.write_hyperparams(&mcmc_cfg);
    info!("Training net");
    if net.num_branches() == 1 {
        info!("Using single branch training procedure");
        net.train_single_branch(&train_data, &mcmc_cfg, true, Some(report_cfg));
    } else {
        net.train(&train_data, &mcmc_cfg, true, Some(report_cfg));
    }
}
