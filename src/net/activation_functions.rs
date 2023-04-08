use arrayfire::{gt, pow, sigmoid, sign, tanh, Array};
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};

#[derive(Serialize, Deserialize, Clone, Copy, clap::ValueEnum, Debug)]
pub enum ActivationFunction {
    Tanh,
    ReLU,
    LeakyReLU,
    SiLU,
    Identity,
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
            ActivationFunction::Identity => 1 * x,
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
            ActivationFunction::Identity => ones_like(x),
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

use crate::af_helpers::ones_like;

pub trait HasActivationFunction {
    fn h(&self, x: &Array<f32>) -> Array<f32>;
    fn dhdx(&self, x: &Array<f32>) -> Array<f32>;
}

#[cfg(test)]
mod tests {
    use super::{ActivationFunction, HasActivationFunction};
    use crate::af_helpers::to_host;
    use arrayfire::{dim4, Array};

    #[test]
    fn af_sign() {
        let af = ActivationFunction::Identity;
        let a = Array::new(&[0f32, 2f32, -2f32], dim4![3, 1, 1, 1]);

        assert_eq!(to_host(&af.h(&a)), vec![0f32, 2f32, -2f32]);
        assert_eq!(to_host(&af.dhdx(&a)), vec![1f32, 1f32, 1f32]);
    }
}
