use crate::error::Error;
use bed_reader::{Bed, ReadOptions};
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

pub struct DataBuilder {
    x: Option<Vec<Vec<f32>>>,
    y: Option<Vec<f32>>,
    x_means: Option<Vec<Vec<f32>>>,
    x_stds: Option<Vec<Vec<f32>>>,
    num_markers_per_branch: Option<Vec<usize>>,
    num_individuals: Option<usize>,
    num_branches: Option<usize>,
    standardized: Option<bool>,
}

impl DataBuilder {
    pub fn new() -> Self {
        DataBuilder {
            x: None,
            y: None,
            x_means: None,
            x_stds: None,
            num_markers_per_branch: None,
            num_individuals: None,
            num_branches: None,
            standardized: None,
        }
    }

    pub fn with_y(mut self, y: Vec<f32>) -> Self {
        self.y = Some(y);
        self
    }

    pub fn with_x_means(mut self, x_means: Vec<Vec<f32>>) -> Self {
        self.x_means = Some(x_means);
        self
    }

    pub fn with_x_stds(mut self, x_stds: Vec<Vec<f32>>) -> Self {
        self.x_stds = Some(x_stds);
        self
    }

    // TODO: compute stds and means in here, too
    pub fn with_x(
        mut self,
        x: Vec<Vec<f32>>,
        num_markers_per_branch: Vec<usize>,
        num_individuals: usize,
    ) -> Self {
        self.x = Some(x);
        self.num_branches = Some(num_markers_per_branch.len());
        self.num_markers_per_branch = Some(num_markers_per_branch);
        self.num_individuals = Some(num_individuals);
        self
    }

    pub fn with_x_from_bed<G>(mut self, bed_path: &Path, grouping: &G) -> Self
    where
        G: MarkerGrouping,
    {
        let mut bed = Bed::new(bed_path).unwrap();
        let mut x = Vec::new();
        let mut x_means = Vec::new();
        let mut x_stds = Vec::new();
        let mut num_markers_per_branch = Vec::new();
        let mut num_individuals = 0;
        let num_branches = grouping.num_groups();
        for gix in 0..grouping.num_groups() {
            // I might want to sort the grouping first. I think that might be more expensive than just
            // reading the data out of order.
            // + I have to save the grouping anyway, or create output reports with the original snp indexing
            // (If there will ever be single SNP PIP or sth like that)
            let val = ReadOptions::builder()
                .f32()
                .f()
                .sid_index(grouping.group(gix).unwrap())
                .read(&mut bed)
                .unwrap();
            // TODO: make sure that t().iter() actually is column-major iteration.
            num_individuals = val.nrows();
            x.push(val.t().iter().copied().collect::<Vec<f32>>());
            x_means.push(val.t().mean_axis(ndarray::Axis(0)).unwrap().to_vec());
            x_stds.push(val.t().std_axis(ndarray::Axis(0), 1f32).to_vec());
            num_markers_per_branch.push(grouping.group_size(gix).unwrap());
        }
        self.x = Some(x);
        self.x_means = Some(x_means);
        self.x_stds = Some(x_stds);
        self.num_markers_per_branch = Some(num_markers_per_branch);
        self.num_individuals = Some(num_individuals);
        self.num_branches = Some(num_branches);
        self.standardized = Some(false);
        self
    }

    pub fn build(mut self) -> Result<Data, Error> {
        if self.x.is_none() {
            return Err(Error::MissingX);
        }
        if self.y.is_none() {
            self.y = Some(vec![0f32; self.num_individuals.unwrap()]);
        }
        if self.standardized.is_none() {
            self.standardized = Some(true);
        }
        Ok(Data {
            x: self.x.unwrap(),
            y: self.y.unwrap(),
            num_markers_per_branch: self.num_markers_per_branch.unwrap(),
            num_individuals: self.num_individuals.unwrap(),
            num_branches: self.num_branches.unwrap(),
            x_means: self.x_means,
            x_stds: self.x_stds,
            standardized: self.standardized.unwrap(),
        })
    }
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
pub struct Data {
    pub x: Vec<Vec<f32>>,
    pub y: Vec<f32>,
    num_markers_per_branch: Vec<usize>,
    num_individuals: usize,
    num_branches: usize,
    pub x_means: Option<Vec<Vec<f32>>>,
    pub x_stds: Option<Vec<Vec<f32>>>,
    standardized: bool,
}

impl Data {
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

    pub fn num_markers_per_branch(&self) -> &Vec<usize> {
        &self.num_markers_per_branch
    }

    pub fn num_markers_in_branch(&self, ix: usize) -> usize {
        self.num_markers_per_branch[ix]
    }

    pub fn num_individuals(&self) -> usize {
        self.num_individuals
    }

    pub fn standardize(&mut self) {
        if !self.standardized {
            if self.x_means.is_some() && self.x_stds.is_some() {
                let x_means = self.x_means.as_ref().unwrap();
                let x_stds = self.x_stds.as_ref().unwrap();
                for branch_ix in 0..self.num_branches() {
                    for marker_ix in 0..self.num_markers_in_branch(branch_ix) {
                        (0..self.num_individuals).for_each(|i| {
                            let val = self.x[branch_ix][self.num_individuals * marker_ix + i];
                            self.x[branch_ix][self.num_individuals * marker_ix + i] = (val
                                - x_means[branch_ix][marker_ix])
                                / x_stds[branch_ix][marker_ix];
                        })
                    }
                }
            } else {
                unimplemented!(
                    "Standardization without precomputed means and stds is not implemented yet."
                );
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
