mod cli;

use clap::Parser;
use cli::cli::{
    ActivationArgs, BranchR2Args, Cli, GroupByGenesArgs, GroupCenteredArgs, GroupMarkerDataArgs,
    PredictArgs, SimulateXYArgs, SimulateYArgs, SubCmd, TrainArgs, TrainNewArgs, TrainNewBedArgs,
};
use log::{debug, info, warn};
use rand::thread_rng;
use rand_distr::{Binomial, Distribution, Normal, Uniform};
use rs_bann::data::genotypes::CompressedGenotypes;
use rs_bann::data::{
    data::Data,
    genotypes::{Genotypes, GenotypesBuilder, GroupedGenotypes},
    phen_stats::PhenStats,
    phenotypes::Phenotypes,
};
use rs_bann::group::{
    centered::CorrGraph, external::ExternalGrouping, gene::GeneGrouping, grouping::MarkerGrouping,
};
use rs_bann::io::bed::BedVM;
use rs_bann::linear_model::LinearModelBuilder;
use rs_bann::net::{
    architectures::{BlockNetCfg, HiddenLayerWidthRule, SummaryLayerWidthRule},
    branch::{
        branch::Branch, lasso_ard::LassoArdBranch, lasso_base::LassoBaseBranch,
        ridge_ard::RidgeArdBranch, ridge_base::RidgeBaseBranch, std_normal_branch::StdNormalBranch,
    },
    mcmc_cfg::MCMCCfg,
    model_type::ModelType,
    net::Net,
    train_stats::ReportCfg,
};
use serde_json::to_writer;
use statrs::statistics::Statistics;
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
        SubCmd::TrainNew(args) => match args.model_type {
            ModelType::LassoBase => train_new::<LassoBaseBranch>(args),
            ModelType::LassoARD => train_new::<LassoArdBranch>(args),
            ModelType::RidgeBase => train_new::<RidgeBaseBranch>(args),
            ModelType::RidgeARD => train_new::<RidgeArdBranch>(args),
            ModelType::StdNormal => train_new::<StdNormalBranch>(args),
            ModelType::Linear => {
                unimplemented!("Training linear models is currently not supported.")
            }
        },
        SubCmd::TrainNewBed(args) => match args.model_type {
            ModelType::LassoBase => train_new_bed::<LassoBaseBranch>(args),
            ModelType::LassoARD => train_new_bed::<LassoArdBranch>(args),
            ModelType::RidgeBase => train_new_bed::<RidgeBaseBranch>(args),
            ModelType::RidgeARD => train_new_bed::<RidgeArdBranch>(args),
            ModelType::StdNormal => train_new_bed::<StdNormalBranch>(args),
            ModelType::Linear => {
                unimplemented!("Training linear models is currently not supported.")
            }
        },
        SubCmd::Train(args) => match args.model_type {
            ModelType::LassoBase => train::<LassoBaseBranch>(args),
            ModelType::LassoARD => train::<LassoArdBranch>(args),
            ModelType::RidgeBase => train::<RidgeBaseBranch>(args),
            ModelType::RidgeARD => train::<RidgeArdBranch>(args),
            ModelType::StdNormal => train::<StdNormalBranch>(args),
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
        SubCmd::GroupMarkerData(args) => group_marker_data(args),
    }
}

fn group_marker_data(args: GroupMarkerDataArgs) {
    info!("Loading groups from: {:?}", args.groups);
    // load groups
    let grouping_path = Path::new(&args.groups);
    let grouping = ExternalGrouping::from_file(grouping_path);

    let path = Path::new(&args.outdir);

    let mut train_bed_str = args.bfile.clone();
    train_bed_str.push_str("_train.bed");
    let train_bed_path = Path::new(&train_bed_str);
    let mut test_bed_str = args.bfile.clone();
    test_bed_str.push_str("_test.bed");
    let test_bed_path = Path::new(&test_bed_str);
    let mut train_path = path.join("train");
    let mut test_path = path.join("test");

    info!("Train bfile path: {:?}", train_bed_path);
    info!("Test bfile path: {:?}", test_bed_path);

    info!("Building grouped data objects");
    // load marker data from .bed, group, save
    let gen_train = GenotypesBuilder::new()
        .with_x_from_bed(train_bed_path, &grouping, args.min_group_size)
        .build()
        .unwrap();
    train_path.set_extension("gen");
    gen_train.to_file(&train_path);

    let gen_test = GenotypesBuilder::new()
        .with_x_from_bed(test_bed_path, &grouping, args.min_group_size)
        .build()
        .unwrap();
    test_path.set_extension("gen");
    gen_test.to_file(&test_path);
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
    let mut genotypes =
        Genotypes::from_file(Path::new(&args.gen)).expect("Failed to load genotype input data");
    let phenotypes =
        Phenotypes::from_file(Path::new(&args.phen)).expect("Failed to load phenotype input data");
    if args.standardize {
        genotypes.standardize();
    }
    let data = Data::new(genotypes, phenotypes);
    // get model type
    let parent_path = Path::new(&args.model_path)
        .parent()
        .unwrap()
        .join("args.json");
    let train_args = TrainNewArgs::from_file(&parent_path);
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
    let mut genotypes = Genotypes::from_file(Path::new(&args.input_data))
        .expect("Failed to load genotype input data");
    if args.standardize {
        genotypes.standardize();
    }

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

    // load models and predict
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
        let outfile = outdir.join(format!("{:?}.json", path.file_stem().unwrap()));
        activations.to_json(&outfile);
    }
}

fn predict(args: PredictArgs) {
    let mut genotypes = Genotypes::from_file(Path::new(&args.input_data))
        .expect("Failed to load genotype input data");
    if args.standardize {
        genotypes.standardize();
    }
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
    B: Branch,
{
    if args.debug {
        simple_logger::init_with_level(log::Level::Debug).unwrap();
    } else {
        simple_logger::init_with_level(log::Level::Info).unwrap();
    }

    if !(args.heritability >= 0. && args.heritability <= 1.) {
        panic!("Heritability must be within [0, 1].");
    }

    let mut path = Path::new(&args.outdir).join(format!(
        "{}_d{}_h{}_v{}_p{}",
        args.model_type,
        args.depth,
        args.heritability,
        args.init_param_variance,
        args.proportion_effective
    ));

    if let (Some(k), Some(s)) = (args.init_gamma_shape, args.init_gamma_scale) {
        path = Path::new(&args.outdir).join(format!(
            "{}_d{}_h{}_k{}_s{}_p{}",
            args.model_type, args.depth, args.heritability, k, s, args.proportion_effective
        ));
    }

    if !path.exists() {
        std::fs::create_dir_all(&path).expect("Could not create output directory!");
    }
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
    let mut net_cfg = if let (Some(k), Some(s)) = (args.init_gamma_shape, args.init_gamma_scale) {
        BlockNetCfg::<B>::new()
            .with_proportion_effective_markers(args.proportion_effective)
            .with_num_hidden_layers(args.depth)
            .with_init_gamma_params(k, s)
            .with_dense_precision_prior(k, s)
            .with_summary_precision_prior(k, s)
            // this is Gamma(1, 1) because at the moment the output variance
            // is hardcoded to 1. TODO: this should be configurable.
            .with_output_precision_prior(1., 1.)
            .with_hidden_layer_width_rule(HiddenLayerWidthRule::FractionOfInput(0.5))
            .with_summary_layer_width_rule(SummaryLayerWidthRule::LikeHiddenLayerWidth)
    } else {
        BlockNetCfg::<B>::new()
            .with_proportion_effective_markers(args.proportion_effective)
            .with_num_hidden_layers(args.depth)
            .with_init_param_variance(args.init_param_variance)
            .with_hidden_layer_width_rule(HiddenLayerWidthRule::FractionOfInput(0.5))
            .with_summary_layer_width_rule(SummaryLayerWidthRule::LikeHiddenLayerWidth)
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

    let mut path = Path::new(&args.outdir).join(format!(
        "{}_d{}_h{}_v{}_p{}",
        args.model_type,
        args.depth,
        args.heritability,
        args.init_param_variance,
        args.proportion_effective
    ));

    if let (Some(k), Some(s)) = (args.init_gamma_shape, args.init_gamma_scale) {
        path = Path::new(&args.outdir).join(format!(
            "{}_d{}_h{}_k{}_s{}_p{}",
            args.model_type, args.depth, args.heritability, k, s, args.proportion_effective
        ));
    }

    if !path.exists() {
        std::fs::create_dir_all(&path).expect("Could not create output directory!");
    }
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

    let mut path = Path::new(&args.outdir).join(format!(
        "{}_b{}_wh{}_ws{}_d{}_m{}_n{}_h{}_v{}_p{}",
        args.model_type,
        args.num_branches,
        args.hidden_layer_width,
        args.summary_layer_width.unwrap_or(args.hidden_layer_width),
        args.branch_depth,
        args.num_markers_per_branch,
        args.num_individuals,
        args.heritability,
        args.init_param_variance,
        args.proportion_effective
    ));

    if let (Some(k), Some(s)) = (args.init_gamma_shape, args.init_gamma_scale) {
        path = Path::new(&args.outdir).join(format!(
            "{}_b{}_wh{}_ws{}_d{}_m{}_n{}_h{}_k{}_s{}_p{}",
            args.model_type,
            args.num_branches,
            args.hidden_layer_width,
            args.summary_layer_width.unwrap_or(args.hidden_layer_width),
            args.branch_depth,
            args.num_markers_per_branch,
            args.num_individuals,
            args.heritability,
            k,
            s,
            args.proportion_effective
        ));
    }

    if !path.exists() {
        std::fs::create_dir_all(&path).expect("Could not create output directory!");
    }
    let mut train_path = path.join("train");
    let mut test_path = path.join("test");
    let args_path = path.join("args.json");
    let params_path = path.join("model.params");

    let mut rng = thread_rng();

    info!("Generating random marker data");
    let ud = Uniform::from(0.0..0.5);
    let mafs = (0..args.num_branches * args.num_markers_per_branch)
        .map(|_| ud.sample(&mut rng))
        .collect::<Vec<f32>>();

    let mut gen_train = GenotypesBuilder::new()
        .with_random_x(
            vec![args.num_markers_per_branch; args.num_branches],
            args.num_individuals,
            Some(mafs.clone()),
        )
        .build()
        .unwrap();
    gen_train.standardize();

    let mut gen_test = GenotypesBuilder::new()
        .with_random_x(
            vec![args.num_markers_per_branch; args.num_branches],
            args.num_individuals,
            Some(mafs),
        )
        .build()
        .unwrap();
    gen_test.standardize();

    let lm = LinearModelBuilder::new(&vec![args.num_markers_per_branch; args.num_branches])
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

    train_path.set_extension("gen");
    gen_train.to_file(&train_path);
    test_path.set_extension("gen");
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
        gen_train.to_json(&path.join("gen_train.json"));
        gen_test.to_json(&path.join("gen_test.json"));
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
        "{}_b{}_wh{}_ws{}_d{}_m{}_n{}_h{}_v{}_p{}",
        args.model_type,
        args.num_branches,
        args.hidden_layer_width,
        args.summary_layer_width.unwrap_or(args.hidden_layer_width),
        args.branch_depth,
        args.num_markers_per_branch,
        args.num_individuals,
        args.heritability,
        args.init_param_variance,
        args.proportion_effective
    ));

    if let (Some(k), Some(s)) = (args.init_gamma_shape, args.init_gamma_scale) {
        path = Path::new(&args.outdir).join(format!(
            "{}_b{}_wh{}_ws{}_d{}_m{}_n{}_h{}_k{}_s{}_p{}",
            args.model_type,
            args.num_branches,
            args.hidden_layer_width,
            args.summary_layer_width.unwrap_or(args.hidden_layer_width),
            args.branch_depth,
            args.num_markers_per_branch,
            args.num_individuals,
            args.heritability,
            k,
            s,
            args.proportion_effective
        ));
    }

    if !path.exists() {
        std::fs::create_dir_all(&path).expect("Could not create output directory!");
    }
    let mut train_path = path.join("train");
    let mut test_path = path.join("test");
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

        let mut net_cfg = if let (Some(k), Some(s)) = (args.init_gamma_shape, args.init_gamma_scale)
        {
            BlockNetCfg::<B>::new()
                .with_num_hidden_layers(args.branch_depth)
                .with_proportion_effective_markers(args.proportion_effective)
                .with_init_gamma_params(k, s)
                .with_dense_precision_prior(k, s)
                .with_summary_precision_prior(k, s)
                .with_output_precision_prior(1., 1.)
                .with_hidden_layer_width_rule(HiddenLayerWidthRule::Fixed(args.hidden_layer_width))
                .with_summary_layer_width_rule(slwr)
        } else {
            BlockNetCfg::<B>::new()
                .with_num_hidden_layers(args.branch_depth)
                .with_proportion_effective_markers(args.proportion_effective)
                .with_init_param_variance(args.init_param_variance)
                .with_hidden_layer_width_rule(HiddenLayerWidthRule::Fixed(args.hidden_layer_width))
                .with_summary_layer_width_rule(slwr)
        };
        for _ in 0..args.num_branches {
            net_cfg.add_branch(args.num_markers_per_branch);
        }
        let net = net_cfg.build_net();

        let mut rng = thread_rng();

        info!("Generating random marker data");
        let gt_per_branch = args.num_markers_per_branch * args.num_individuals;
        let mut x_train: Vec<Vec<f32>> = vec![vec![0.0; gt_per_branch]; args.num_branches];
        let mut x_test: Vec<Vec<f32>> = vec![vec![0.0; gt_per_branch]; args.num_branches];
        let mut x_means: Vec<Vec<f32>> =
            vec![vec![0.0; args.num_markers_per_branch]; args.num_branches];
        let mut x_stds: Vec<Vec<f32>> =
            vec![vec![0.0; args.num_markers_per_branch]; args.num_branches];
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

        let mut gen_train = GenotypesBuilder::new()
            .with_x(
                x_train,
                vec![args.num_markers_per_branch; args.num_branches],
                args.num_individuals,
            )
            .with_means(x_means.clone())
            .with_stds(x_stds.clone())
            .build()
            .unwrap();
        gen_train.standardize();

        let mut gen_test = GenotypesBuilder::new()
            .with_x(
                x_test,
                vec![args.num_markers_per_branch; args.num_branches],
                args.num_individuals,
            )
            .with_means(x_means)
            .with_stds(x_stds)
            .build()
            .unwrap();
        gen_test.standardize();

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

        train_path.set_extension("gen");
        gen_train.to_file(&train_path);
        test_path.set_extension("gen");
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
            gen_train.to_json(&path.join("gen_train.json"));
            gen_test.to_json(&path.join("gen_test.json"));
        }

        args.to_file(&args_path);

        break;
    }
}

fn load_ungrouped_data(
    args: &TrainNewBedArgs,
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

    let test_phen = args
        .p_test
        .as_ref()
        .map(|pfile| Phenotypes::from_file(Path::new(&pfile)));

    let test_data = if let (Some(tg), Some(tp)) = (gen_test, test_phen) {
        // just checked that they are Some
        Some(Data::new(tg, tp.unwrap()))
    } else {
        info!("No complete test data provided, proceeding without");
        None
    };
    (train_data, test_data)
}

fn load_grouped_data(indir: &str) -> (Data<Genotypes>, Option<Data<Genotypes>>) {
    let train_gen = Genotypes::from_file(&Path::new(indir).join("train.gen"))
        .expect("Failed to load train.gen training genotypes");
    let train_phen = Phenotypes::from_file(&Path::new(indir).join("train.phen"))
        .expect("Failed to load train.phen training phenotypes");
    let train_data = Data::new(train_gen, train_phen);
    let test_gen = Genotypes::from_file(&Path::new(indir).join("test.gen"));
    let test_phen = Phenotypes::from_file(&Path::new(indir).join("test.phen"));
    let test_data = if let (Ok(tg), Ok(tp)) = (test_gen, test_phen) {
        // just checked that they are Ok()
        Some(Data::new(tg, tp))
    } else {
        info!("No complete test data provided in bed format, proceeding without");
        None
    };
    (train_data, test_data)
}

fn train_new<B>(args: TrainNewArgs)
where
    B: Branch,
{
    if args.debug_prints {
        simple_logger::init_with_level(log::Level::Debug).unwrap();
    } else {
        simple_logger::init_with_level(log::Level::Info).unwrap();
    }

    info!("Loading pre-grouped data from input directory");
    let (train_data, test_data) = load_grouped_data(&args.indir);

    let mut outdir = format!(
        "{}_d{}_cl{}_il{}_{}_dpk{}_dps{}_spk{}_sps{}_opk{}_ops{}",
        args.model_type,
        args.branch_depth,
        args.chain_length,
        args.integration_length,
        args.step_size_mode,
        args.dpk,
        args.dps,
        args.spk,
        args.sps,
        args.opk,
        args.ops,
    );

    if args.joint_hmc {
        outdir.push_str("_joint");
    }

    let hlwr = if let Some(width) = args.fixed_hidden_layer_width {
        outdir.push_str(&format!("_fhlw{}", width));
        HiddenLayerWidthRule::Fixed(width)
    } else {
        outdir.push_str(&format!("_rhlw{}", args.relative_hidden_layer_width));
        HiddenLayerWidthRule::FractionOfInput(args.relative_hidden_layer_width)
    };

    let slwr = if let Some(width) = args.fixed_summary_layer_width {
        outdir.push_str(&format!("_fslw{}", width));
        SummaryLayerWidthRule::Fixed(width)
    } else {
        outdir.push_str(&format!("_rslw{}", args.relative_summary_layer_width));
        SummaryLayerWidthRule::FractionOfHiddenLayerWidth(args.relative_summary_layer_width)
    };

    let mcmc_cfg = MCMCCfg {
        hmc_step_size_factor: args.step_size,
        hmc_max_hamiltonian_error: args.max_hamiltonian_error,
        hmc_integration_length: args.integration_length,
        hmc_step_size_mode: args.step_size_mode.clone(),
        chain_length: args.chain_length,
        burn_in: args.burn_in,
        outpath: outdir,
        trace: args.trace,
        trajectories: args.trajectories,
        num_grad_traj: args.num_grad_traj,
        num_grad: args.num_grad,
        gradient_descent: args.gradient_descent,
        gradient_descent_joint: args.gradient_descent_joint,
        joint_hmc: args.joint_hmc,
    };
    mcmc_cfg.create_out();

    args.to_file(&mcmc_cfg.args_path());

    let report_cfg = ReportCfg::new(args.report_interval, test_data.as_ref());

    info!("Building net");

    let mut net_cfg = BlockNetCfg::<B>::new()
        .with_num_hidden_layers(args.branch_depth)
        .with_dense_precision_prior(args.dpk, args.dps)
        .with_summary_precision_prior(args.spk, args.sps)
        .with_output_precision_prior(args.opk, args.ops)
        .with_hidden_layer_width_rule(hlwr)
        .with_summary_layer_width_rule(slwr);

    for bix in 0..train_data.num_branches() {
        net_cfg.add_branch(train_data.num_markers_in_branch(bix));
    }
    let mut net = net_cfg.build_net();
    if let Some(p) = args.error_precision {
        net.set_error_precision(p);
    }

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
    net.train(&train_data, &mcmc_cfg, true, Some(report_cfg));
}

fn train_new_bed<B>(args: TrainNewBedArgs)
where
    B: Branch,
{
    if args.debug_prints {
        simple_logger::init_with_level(log::Level::Debug).unwrap();
    } else {
        simple_logger::init_with_level(log::Level::Info).unwrap();
    }

    info!("Loading data.");
    let (train_data, test_data) = load_ungrouped_data(&args);

    let mut outdir = format!(
        "{}_d{}_cl{}_il{}_{}_dpk{}_dps{}_spk{}_sps{}_opk{}_ops{}",
        args.model_type,
        args.branch_depth,
        args.chain_length,
        args.integration_length,
        args.step_size_mode,
        args.dpk,
        args.dps,
        args.spk,
        args.sps,
        args.opk,
        args.ops,
    );

    if args.joint_hmc {
        outdir.push_str("_joint");
    }

    let hlwr = if let Some(width) = args.fixed_hidden_layer_width {
        outdir.push_str(&format!("_fhlw{}", width));
        HiddenLayerWidthRule::Fixed(width)
    } else {
        outdir.push_str(&format!("_rhlw{}", args.relative_hidden_layer_width));
        HiddenLayerWidthRule::FractionOfInput(args.relative_hidden_layer_width)
    };

    let slwr = if let Some(width) = args.fixed_summary_layer_width {
        outdir.push_str(&format!("_fslw{}", width));
        SummaryLayerWidthRule::Fixed(width)
    } else {
        outdir.push_str(&format!("_rslw{}", args.relative_summary_layer_width));
        SummaryLayerWidthRule::FractionOfHiddenLayerWidth(args.relative_summary_layer_width)
    };

    let mcmc_cfg = MCMCCfg {
        hmc_step_size_factor: args.step_size,
        hmc_max_hamiltonian_error: args.max_hamiltonian_error,
        hmc_integration_length: args.integration_length,
        hmc_step_size_mode: args.step_size_mode.clone(),
        chain_length: args.chain_length,
        burn_in: args.burn_in,
        outpath: outdir,
        trace: args.trace,
        trajectories: args.trajectories,
        num_grad_traj: args.num_grad_traj,
        num_grad: args.num_grad,
        gradient_descent: args.gradient_descent,
        gradient_descent_joint: args.gradient_descent_joint,
        joint_hmc: args.joint_hmc,
    };
    mcmc_cfg.create_out();

    args.to_file(&mcmc_cfg.args_path());

    let report_cfg = ReportCfg::new(args.report_interval, test_data.as_ref());

    info!("Building net");

    let mut net_cfg = BlockNetCfg::<B>::new()
        .with_num_hidden_layers(args.branch_depth)
        .with_dense_precision_prior(args.dpk, args.dps)
        .with_summary_precision_prior(args.spk, args.sps)
        .with_output_precision_prior(args.opk, args.ops)
        .with_hidden_layer_width_rule(hlwr)
        .with_summary_layer_width_rule(slwr);

    for bix in 0..train_data.num_branches() {
        net_cfg.add_branch(train_data.num_markers_in_branch(bix));
    }
    let mut net = net_cfg.build_net();
    if let Some(p) = args.error_precision {
        net.set_error_precision(p);
    }

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
    net.train(&train_data, &mcmc_cfg, true, Some(report_cfg));
}

fn train<B>(args: TrainArgs)
where
    B: Branch,
{
    if args.debug_prints {
        simple_logger::init_with_level(log::Level::Debug).unwrap();
    } else {
        simple_logger::init_with_level(log::Level::Info).unwrap();
    }

    info!("Loading data");
    let (train_data, test_data) = load_grouped_data(&args.indir);

    let model_path = Path::new(&args.model_file);

    let mut outdir = format!(
        "{}_cl{}_il{}_{}",
        model_path.file_stem().unwrap().to_string_lossy(),
        args.chain_length,
        args.integration_length,
        args.step_size_mode,
    );

    if args.joint_hmc {
        outdir.push_str("_joint");
    }

    let mcmc_cfg = MCMCCfg {
        hmc_step_size_factor: args.step_size,
        hmc_max_hamiltonian_error: args.max_hamiltonian_error,
        hmc_integration_length: args.integration_length,
        hmc_step_size_mode: args.step_size_mode.clone(),
        chain_length: args.chain_length,
        burn_in: args.burn_in,
        outpath: outdir,
        trace: args.trace,
        trajectories: args.trajectories,
        num_grad_traj: args.num_grad_traj,
        num_grad: args.num_grad,
        gradient_descent: args.gradient_descent,
        gradient_descent_joint: args.gradient_descent_joint,
        joint_hmc: args.joint_hmc,
    };
    mcmc_cfg.create_out();

    args.to_file(&mcmc_cfg.args_path());

    let report_cfg = ReportCfg::new(args.report_interval, test_data.as_ref());

    info!("Loading net");

    let mut net = Net::<RidgeBaseBranch>::from_file(model_path);
    if let Some(p) = args.error_precision {
        net.set_error_precision(p);
    }
    net.write_hyperparams(&mcmc_cfg);
    info!("Training net");
    net.train(&train_data, &mcmc_cfg, true, Some(report_cfg));
}
