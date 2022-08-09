use bincode::{deserialize_from, serialize_into};
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

#[derive(Serialize, Deserialize, PartialEq, Debug)]
pub struct Data {
    pub x: Vec<Vec<f64>>,
    pub y: Vec<f64>,
    pub x_means: Vec<Vec<f64>>,
    pub x_stds: Vec<Vec<f64>>,
}

impl Data {
    pub fn new(
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        x_means: Vec<Vec<f64>>,
        x_stds: Vec<Vec<f64>>,
    ) -> Self {
        Data {
            x,
            y,
            x_means,
            x_stds,
        }
    }

    pub fn from_file(path: &Path) -> Self {
        let mut r = BufReader::new(File::open(path).unwrap());
        deserialize_from(&mut r).unwrap()
    }

    pub fn to_file(&self, path: &Path) {
        let mut f = BufWriter::new(File::create(path).unwrap());
        serialize_into(&mut f, self).unwrap();
    }
}
