extern crate blas_src;
extern crate openblas_src;

// re-export for macro use
pub extern crate rand;

pub mod af_helpers;
pub mod arr_helpers;
pub mod data;
pub mod error;
pub mod group;
pub mod io;
pub mod linear_model;
pub mod net;
