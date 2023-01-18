//! Useful elementary array based functions that are missing in arrayfire.

use arrayfire::{dim4, Array, MatProp};

/// Create a scalar stored on device
pub(crate) fn af_scalar(val: f32) -> Array<f32> {
    Array::new(&vec![val], dim4!(1, 1, 1, 1))
}

pub(crate) fn l2_norm(arr: &Array<f32>) -> f32 {
    let mut sum_of_squares = vec![0.0];
    arrayfire::dot(
        &arrayfire::flat(arr),
        &arrayfire::flat(arr),
        MatProp::NONE,
        MatProp::NONE,
    )
    .host(&mut sum_of_squares);
    sum_of_squares[0]
}

pub(crate) fn l1_norm(arr: &Array<f32>) -> f32 {
    arrayfire::sum_all(&arrayfire::abs(arr)).0
}

pub(crate) fn sign(arr: &Array<f32>) -> Array<f32> {
    let neg = arrayfire::sign(arr);
    let pos = arrayfire::gt(arr, &0f32, false);
    let a_dims = *arr.dims().get();
    arrayfire::constant!(0f32; a_dims[0], a_dims[1], a_dims[2], a_dims[3]) - neg + pos
}

pub(crate) fn to_host(a: &Array<f32>) -> Vec<f32> {
    let mut buffer = Vec::<f32>::new();
    buffer.resize(a.elements(), 0.);
    a.host(&mut buffer);
    buffer
}

pub(crate) fn scalar_to_host(a: &Array<f32>) -> f32 {
    let mut host_data = vec![0.0];
    a.host(&mut host_data);
    host_data[0]
}