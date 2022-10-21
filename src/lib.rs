extern crate blas_src;
extern crate openblas_src;

pub mod net;

use arrayfire::Array;

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
