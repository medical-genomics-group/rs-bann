//! Useful elementary array based functions that are missing in arrayfire.

use arrayfire::{Array, MatProp};

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
