use arrayfire::{gt, pow, sigmoid, sign, tanh, Array};

pub trait HasActivationFunction {
    fn activation(&self, x: &Array<f32>) -> Array<f32>;
    fn d_activation(&self, x: &Array<f32>) -> Array<f32>;
}

pub(crate) trait ActivationFunction {
    fn f(x: &Array<f32>) -> Array<f32>;
    fn dfdx(x: &Array<f32>) -> Array<f32>;
}

pub(crate) struct Tanh;
impl ActivationFunction for Tanh {
    fn f(x: &Array<f32>) -> Array<f32> {
        tanh(x)
    }

    fn dfdx(x: &Array<f32>) -> Array<f32> {
        1 - pow(&Tanh::f(x), &2, false)
    }
}

pub(crate) struct ReLU;
impl ActivationFunction for ReLU {
    fn f(x: &Array<f32>) -> Array<f32> {
        x * gt(x, &0f32, false)
    }

    fn dfdx(x: &Array<f32>) -> Array<f32> {
        gt(x, &0f32, false) * 1f32
    }
}

pub(crate) struct LeakyReLU;
impl ActivationFunction for LeakyReLU {
    fn f(x: &Array<f32>) -> Array<f32> {
        x * gt(x, &0f32, false) + x * sign(x) * 0.01f32
    }

    fn dfdx(x: &Array<f32>) -> Array<f32> {
        gt(x, &0f32, false) * 1f32 + sign(x) * 0.01f32
    }
}

pub(crate) struct SiLU;
impl ActivationFunction for SiLU {
    fn f(x: &Array<f32>) -> Array<f32> {
        x * sigmoid(x)
    }

    fn dfdx(x: &Array<f32>) -> Array<f32> {
        let fx = SiLU::f(x);
        &fx + sigmoid(x) * (1 - &fx)
    }
}
