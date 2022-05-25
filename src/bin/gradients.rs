use ndarray::arr1;
use rs_bann::network::MarkerGroup;
use rs_bedvec::io::BedReader;
use std::env;

fn main() {
    let param_vec: Vec<f32> = env::args().skip(1).map(|e| e.parse().unwrap()).collect();

    assert_eq!(
        param_vec.len(),
        4,
        "Exactly four parameter values need to be provided."
    );

    let reader = BedReader::new("resources/test/four_by_two.bed", 4, 2);
    let mut mg = MarkerGroup::new(
        arr1(&[-0.587_430_3, 0.020_813_8, 0.346_810_51, 0.283_149_64]),
        arr1(&[1., 1.]),
        1.,
        1.,
        reader,
        2,
    );
    mg.set_params(&arr1(&param_vec));
    dbg!(mg.param_vec());
    mg.load_marker_data();
    println!(
        "numerical gradient: {:?}",
        mg.numerical_log_density_gradient_two_point(&mg.param_vec())
    );
    println!(
        "analytical gradient: {:?}",
        mg.log_density_gradient(&mg.param_vec())
    );
    mg.forget_marker_data();
}
