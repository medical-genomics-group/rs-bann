use bincode::{deserialize_from, serialize_into};
use serde::{Deserialize, Serialize};
use serde_json::{to_writer, to_writer_pretty};

use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

use crate::group::grouping::MarkerGrouping;

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
    num_branches: usize,
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
        num_branches: usize,
        standardized: bool,
    ) -> Self {
        Data {
            x,
            y,
            x_means,
            x_stds,
            num_markers_per_branch,
            num_individuals,
            num_branches,
            standardized,
        }
    }

    // pub fn from_bed<G>(bed_path: &Path, grouping: &G) -> Self
    // where
    //     G: MarkerGrouping,
    // {
    //     // the bed reader output is row major, but I'd need to put it
    //     // in the array as col major I think?
    //     // .c option should make it 'row major' in their mind,
    //     // I have to try what the actual output format is.
    //     let mut bed = Bed::new(bed_path).unwrap();
    //     let val = ReadOptions::builder().f32().read(&mut bed).unwrap();
    // }

    pub fn from_file(path: &Path) -> Self {
        let mut r = BufReader::new(File::open(path).unwrap());
        deserialize_from(&mut r).unwrap()
    }

    pub fn to_file(&self, path: &Path) {
        let mut f = BufWriter::new(File::create(path).unwrap());
        serialize_into(&mut f, self).unwrap();
    }

    pub fn to_json(&self, path: &Path) {
        to_writer(File::create(path).unwrap(), self).unwrap();
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

#[cfg(test)]
mod tests {
    use bed_reader::{Bed, ReadOptions};
    use std::env;
    use std::path::Path;

    #[test]
    fn test() {
        let base_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let base_path = Path::new(&base_dir);
        let bed_path = base_path.join("resources/test/small.bed");

        let mut bed = Bed::new(&bed_path).unwrap();
        // c or f doesn't affect the shape of the output array, it is always the same 20 x 11 matrix.
        // it probably sets the memory layout for efficient access. I guess I would want f(), as I
        // want to extract columns from the final array.
        let val = ReadOptions::builder().f32().f().read(&mut bed).unwrap();
        let row_major_mat: Vec<f32> = vec![
            0., 1., 0., 0., 0., 0., 2., 1., 0., 0., 1., 0., 0., 0., 1., 0., 2., 0., 1., 0., 1., 1.,
            1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 1.,
            1., 0., 0., 0., 0., 1., 0., 1., 0., 1., 1., 0., 2., 0., 1., 0., 1., 0., 1., 2., 2., 0.,
            0., 0., 0., 1., 0., 2., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 2.,
            1., 1., 0., 1., 0., 1., 0., 1., 0., 1., 1., 0., 1., 0., 0., 0., 1., 1., 2., 1., 1., 1.,
            0., 0., 0., 0., 0., 2., 1., 2., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1.,
            0., 0., 1., 1., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 1., 2., 1., 0.,
            1., 0., 0., 0., 0., 2., 0., 2., 0., 1., 1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 1.,
            2., 1., 0., 1., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 1., 0., 1., 1., 1., 0., 0., 0.,
        ];
        assert_eq!(val.iter().cloned().collect::<Vec<f32>>(), row_major_mat);
    }
}
