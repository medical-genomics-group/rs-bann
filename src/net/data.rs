use bincode::{deserialize_from, serialize_into};
use log::info;
use serde::{Deserialize, Serialize};
use serde_json::to_writer_pretty;
use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

#[derive(Serialize, Deserialize)]
pub struct PhenStats {
    mean: f32,
    variance: f32,
    env_variance: f32,
    mse: f32,
}

impl PhenStats {
    pub fn new(mean: f32, variance: f32, env_variance: f32, mse: f32) -> Self {
        Self {
            mean,
            variance,
            env_variance,
            mse,
        }
    }

    pub fn to_file(&self, path: &Path) {
        info!("Creating: {:?}", path);
        to_writer_pretty(File::create(path).unwrap(), self).unwrap();
    }
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
pub struct Data {
    pub x: Vec<Vec<f32>>,
    pub y: Vec<f32>,
    pub x_means: Vec<Vec<f32>>,
    pub x_stds: Vec<Vec<f32>>,
    num_markers_per_branch: usize,
    num_individuals: usize,
    standardized: bool,
}

impl Data {
    pub fn new(
        x: Vec<Vec<f32>>,
        y: Vec<f32>,
        x_means: Vec<Vec<f32>>,
        x_stds: Vec<Vec<f32>>,
        num_markers_per_branch: usize,
        num_individuals: usize,
        standardized: bool,
    ) -> Self {
        Data {
            x,
            y,
            x_means,
            x_stds,
            num_markers_per_branch,
            num_individuals,
            standardized,
        }
    }

    pub fn from_file(path: &Path) -> Self {
        let mut r = BufReader::new(File::open(path).unwrap());
        deserialize_from(&mut r).unwrap()
    }

    pub fn to_file(&self, path: &Path) {
        info!("Creating: {:?}", path);
        let mut f = BufWriter::new(File::create(path).unwrap());
        serialize_into(&mut f, self).unwrap();
    }

    pub fn num_branches(&self) -> usize {
        self.x.len()
    }

    pub fn num_markers_per_branch(&self) -> usize {
        self.num_markers_per_branch
    }

    pub fn num_individuals(&self) -> usize {
        self.num_individuals
    }

    pub fn standardize(&mut self) {
        if !self.standardized {
            for branch_ix in 0..self.num_branches() {
                for marker_ix in 0..self.num_markers_per_branch {
                    (0..self.num_individuals).for_each(|i| {
                        let val = self.x[branch_ix][self.num_individuals * marker_ix + i];
                        self.x[branch_ix][self.num_individuals * marker_ix + i] = (val
                            - self.x_means[branch_ix][marker_ix])
                            / self.x_stds[branch_ix][marker_ix];
                    })
                }
            }
            self.standardized = true;
        }
    }
}
