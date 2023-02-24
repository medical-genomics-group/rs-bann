use arrayfire::{gt, pow, sigmoid, sign, tanh, Array};
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};

#[derive(Serialize, Deserialize, Clone, Copy, clap::ValueEnum, Debug)]
pub enum ActivationFunction {
    Tanh,
    ReLU,
    LeakyReLU,
    SiLU,
}

impl Display for ActivationFunction {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
        // or, alternatively:
        // fmt::Debug::fmt(self, f)
    }
}

impl HasActivationFunction for ActivationFunction {
    fn h(&self, x: &Array<f32>) -> Array<f32> {
        match self {
            ActivationFunction::Tanh => tanh(x),
            ActivationFunction::ReLU => x * gt(x, &0f32, false),
            ActivationFunction::LeakyReLU => x * gt(x, &0f32, false) + x * sign(x) * 0.01f32,
            ActivationFunction::SiLU => x * sigmoid(x),
        }
    }

    fn dhdx(&self, x: &Array<f32>) -> Array<f32> {
        match self {
            ActivationFunction::Tanh => 1 - pow(&self.h(x), &2, false),
            ActivationFunction::ReLU => gt(x, &0f32, false) * 1f32,
            ActivationFunction::LeakyReLU => gt(x, &0f32, false) * 1f32 + sign(x) * 0.01f32,
            ActivationFunction::SiLU => {
                let fx = self.h(x);
                &fx + sigmoid(x) * (1 - &fx)
            }
        }
    }
}

macro_rules! has_activation_function {
    ($t:ident) => {
        impl HasActivationFunction for $t {
            fn h(&self, x: &Array<f32>) -> Array<f32> {
                self.activation_function.h(x)
            }

            fn dhdx(&self, x: &Array<f32>) -> Array<f32> {
                self.activation_function.dhdx(x)
            }
        }
    };
}
pub(crate) use has_activation_function;

pub trait HasActivationFunction {
    fn h(&self, x: &Array<f32>) -> Array<f32>;
    fn dhdx(&self, x: &Array<f32>) -> Array<f32>;
}
