use ndarray::arr1;
use rs_bann::network::MarkerGroup;
use rs_bedvec::io::BedReader;

fn main() {
    test_crate();
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

fn test_crate() {
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
    let mut prev_res = arr1(&[-0.587_430_3, 0.020_813_8, 0.346_810_51, 0.283_149_64]);
    let n_samples = 1000;
    let mut n_rejected = 0;
    for _i in 0..n_samples {
        let res = mg.sample_params(100);
        if res == prev_res {
            n_rejected += 1;
        }
        println!("{:?}", res);
        prev_res = res.clone();
        mg.set_params(&res);
    }
    mg.forget_marker_data();
    dbg!(n_rejected as f64 / n_samples as f64);
}
