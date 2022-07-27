use arrayfire::{dim4, randn, Array};
use clap::Parser;
use log::info;
use ndarray::arr1;
use rs_bann::afnet::ArmBuilder;
use rs_bann::network::MarkerGroup;
use rs_bedvec::io::BedReader;

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

    /// hmc step size
    #[clap(short, long)]
    step_size: Option<f64>,

    /// enable debug prints
    #[clap(short, long)]
    debug_prints: bool,
}

fn main() {
    test_crate_af();
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
    let true_net = ArmBuilder::new()
        .with_num_markers(args.num_markers as usize)
        .add_hidden_layer(args.hidden_layer_width as usize)
        .add_layer_weights(&w0)
        .add_layer_biases(&b0)
        .add_summary_weights(&w1)
        .add_summary_bias(&b1)
        .add_output_weight(&w2)
        .verbose()
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

    let mut train_net = ArmBuilder::new()
        .with_num_markers(args.num_markers as usize)
        .add_hidden_layer(args.hidden_layer_width as usize)
        .with_initial_weights_value(1.)
        .with_initial_bias_value(1.)
        .verbose()
        .build();

    // train
    let mut accepted_samples: u64 = 0;

    for i in 0..args.chain_length {
        if train_net.hmc_step(
            &x_train,
            &y_train,
            args.integration_length,
            args.step_size,
            args.max_hamiltonian_error,
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
