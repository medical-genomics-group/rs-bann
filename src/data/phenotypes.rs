use crate::error::Error;
use arrayfire::{dim4, Array};
use bincode::{deserialize_from, serialize_into};
use serde::{Deserialize, Serialize};
use serde_json::to_writer;
use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

#[derive(Serialize, Deserialize, PartialEq, Debug)]
pub struct Phenotypes {
    y: Vec<f32>,
}

impl Phenotypes {
    pub fn new(y: Vec<f32>) -> Self {
        Self { y }
    }

    pub fn zeros(num_individuals: usize) -> Self {
        Self {
            y: vec![0f32; num_individuals],
        }
    }

    pub fn from_file(path: &Path) -> Result<Self, Error> {
        let mut r = BufReader::new(File::open(path)?);
        Ok(deserialize_from(&mut r)?)
    }

    pub fn to_file(&self, path: &Path) {
        let mut f = BufWriter::new(File::create(path).unwrap());
        serialize_into(&mut f, self).unwrap();
    }

    pub fn to_json(&self, path: &Path) {
        to_writer(File::create(path).unwrap(), self).unwrap();
    }

    pub fn y_af(&self) -> Array<f32> {
        Array::new(&self.y, dim4!(self.y.len() as u64))
    }

    pub fn y(&self) -> &[f32] {
        &self.y
    }
}
