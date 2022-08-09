use bincode::{deserialize_from, serialize_into};
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufReader, BufWriter},
};

#[derive(Serialize, Deserialize, PartialEq, Debug)]
pub struct Data {
    pub x: Vec<Vec<f64>>,
    pub y: Vec<f64>,
}

impl Data {
    pub fn new(x: Vec<Vec<f64>>, y: Vec<f64>) -> Self {
        Data { x, y }
    }

    pub fn from_file(path: &str) -> Self {
        let mut r = BufReader::new(File::open(path).unwrap());
        deserialize_from(&mut r).unwrap()
    }

    pub fn to_file(&self, path: &str) {
        let mut f = BufWriter::new(File::create(path).unwrap());
        serialize_into(&mut f, self).unwrap();
    }
}
