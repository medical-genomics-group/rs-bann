use crate::error::Error;
use crate::group::grouping::MarkerGrouping;

use arrayfire::{dim4, Array};
use bed_reader::{Bed as ExternBed, ReadOptions};
use bincode::{deserialize_from, serialize_into};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Binomial, Distribution, Uniform};
use serde::{Deserialize, Serialize};
use serde_json::{to_writer, to_writer_pretty};
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
}

impl PhenStats {
    pub fn new(mean: f32, variance: f32, env_variance: f32) -> Self {
        Self {
            mean,
            variance,
            env_variance,
        }
    }

    pub fn to_file(&self, path: &Path) {
        to_writer_pretty(File::create(path).unwrap(), self).unwrap();
    }
}

pub struct GenotypesBuilder {
    x: Option<Vec<Vec<f32>>>,
    num_individuals: Option<usize>,
    num_markers_per_branch: Option<Vec<usize>>,
    num_branches: Option<usize>,
    means: Option<Vec<Vec<f32>>>,
    stds: Option<Vec<Vec<f32>>>,
    standardized: Option<bool>,
    rng: ChaCha20Rng,
}

impl Default for GenotypesBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GenotypesBuilder {
    pub fn new() -> Self {
        Self {
            x: None,
            num_individuals: None,
            num_markers_per_branch: None,
            num_branches: None,
            means: None,
            stds: None,
            standardized: None,
            rng: ChaCha20Rng::from_entropy(),
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = ChaCha20Rng::seed_from_u64(seed);
        self
    }

    pub fn with_means(mut self, means: Vec<Vec<f32>>) -> Self {
        self.means = Some(means);
        self
    }

    pub fn with_stds(mut self, stds: Vec<Vec<f32>>) -> Self {
        self.stds = Some(stds);
        self
    }

    pub fn with_random_x(
        mut self,
        num_markers_per_branch: Vec<usize>,
        num_individuals: usize,
        mafs: Option<Vec<f32>>,
    ) -> Self {
        let num_branches = num_markers_per_branch.len();
        let mut x: Vec<Vec<f32>> = Vec::new();
        let mut x_means: Vec<Vec<f32>> = Vec::new();
        let mut x_stds: Vec<Vec<f32>> = Vec::new();
        for branch_ix in 0..num_branches {
            let nmpb = num_markers_per_branch[branch_ix];
            x.push(vec![0.0; nmpb * num_individuals]);
            x_means.push(vec![0.0; nmpb]);
            x_stds.push(vec![0.0; nmpb]);
            for marker_ix in 0..nmpb {
                loop {
                    let maf = if let Some(v) = &mafs {
                        assert!(
                            v[num_branches * branch_ix + marker_ix] != 0.0,
                            "maf of 0 it not allowed in simulation"
                        );
                        v[num_branches * branch_ix + marker_ix]
                    } else {
                        Uniform::from(0.01..0.5).sample(&mut self.rng)
                    };
                    let mut col_sum: usize = 0;
                    let binom = Binomial::new(2, maf as f64).unwrap();
                    (0..num_individuals).for_each(|i| {
                        x[branch_ix][marker_ix * num_individuals + i] =
                            binom.sample(&mut self.rng) as f32;
                        col_sum += x[branch_ix][marker_ix * num_individuals + i] as usize;
                    });
                    let sampled_maf = col_sum as f32 / (2.0 * num_individuals as f32);
                    let col_mean = 2.0 * sampled_maf;
                    let var: f32 = (0..num_individuals)
                        .map(|i| x[branch_ix][marker_ix * num_individuals + i] - col_mean)
                        .map(|e| e * e)
                        .sum::<f32>()
                        / num_individuals as f32;
                    x_means[branch_ix][marker_ix] = col_mean;
                    x_stds[branch_ix][marker_ix] = var.sqrt();
                    if var != 0.0 {
                        break;
                    }
                }
            }
        }
        self.x = Some(x);
        self.stds = Some(x_stds);
        self.means = Some(x_means);
        self.num_branches = Some(num_branches);
        self.num_markers_per_branch = Some(num_markers_per_branch);
        self.num_individuals = Some(num_individuals);
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

    pub fn with_x_from_bed<G>(
        mut self,
        bed_path: &Path,
        grouping: &G,
        min_group_size: usize,
    ) -> Self
    where
        G: MarkerGrouping,
    {
        let mut bed = ExternBed::new(bed_path).unwrap();
        let mut x = Vec::new();
        let mut x_means = Vec::new();
        let mut x_stds = Vec::new();
        let mut num_markers_per_branch = Vec::new();
        let mut num_individuals = 0;
        let num_branches = grouping.num_groups();
        for gix in 0..grouping.num_groups() {
            // safe to unwrap, because loop doesn't exceed num_groups
            if grouping.group_size(gix).unwrap() < min_group_size {
                continue;
            }
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
        self.means = Some(x_means);
        self.stds = Some(x_stds);
        self.num_markers_per_branch = Some(num_markers_per_branch);
        self.num_individuals = Some(num_individuals);
        self.num_branches = Some(num_branches);
        self.standardized = Some(false);
        self
    }

    pub fn build(mut self) -> Result<Genotypes, Error> {
        if self.x.is_none() {
            return Err(Error::MissingX);
        }
        if self.standardized.is_none() {
            self.standardized = Some(false);
        }
        Ok(Genotypes {
            x: self.x.unwrap(),
            num_markers_per_branch: self.num_markers_per_branch.unwrap(),
            num_individuals: self.num_individuals.unwrap(),
            num_branches: self.num_branches.unwrap(),
            means: self.means,
            stds: self.stds,
            standardized: self.standardized.unwrap(),
        })
    }
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
pub struct Genotypes {
    x: Vec<Vec<f32>>,
    num_individuals: usize,
    num_markers_per_branch: Vec<usize>,
    num_branches: usize,
    means: Option<Vec<Vec<f32>>>,
    stds: Option<Vec<Vec<f32>>>,
    standardized: bool,
}

impl Genotypes {
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

    pub fn x(&self) -> &Vec<Vec<f32>> {
        &self.x
    }

    pub fn stds(&self) -> &Option<Vec<Vec<f32>>> {
        &self.stds
    }

    pub fn means(&self) -> &Option<Vec<Vec<f32>>> {
        &self.means
    }

    pub fn num_individuals(&self) -> usize {
        self.num_individuals
    }

    pub fn num_markers_per_branch(&self) -> &Vec<usize> {
        &self.num_markers_per_branch
    }

    pub fn af_branch_data(&self, branch_ix: usize) -> Array<f32> {
        Array::new(
            &self.x[branch_ix],
            dim4!(
                self.num_individuals as u64,
                self.num_markers_per_branch[branch_ix] as u64
            ),
        )
    }

    pub fn standardize(&mut self) {
        if !self.standardized {
            if self.means.is_some() && self.stds.is_some() {
                let means = self.means.as_ref().unwrap();
                let stds = self.stds.as_ref().unwrap();
                for branch_ix in 0..self.num_branches {
                    for marker_ix in 0..self.num_markers_per_branch[branch_ix] {
                        (0..self.num_individuals).for_each(|i| {
                            let val = self.x[branch_ix][self.num_individuals * marker_ix + i];
                            self.x[branch_ix][self.num_individuals * marker_ix + i] =
                                (val - means[branch_ix][marker_ix]) / stds[branch_ix][marker_ix];
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

    pub fn x_branch_af(&self, branch_ix: usize) -> Array<f32> {
        Array::new(
            &self.x[branch_ix],
            dim4!(
                self.num_individuals as u64,
                self.num_markers_per_branch[branch_ix] as u64
            ),
        )
    }
}

// #[derive(Serialize, Deserialize, PartialEq, Debug)]
// pub struct GenotypesBed {
//     x: Vec<Vec<u8>>,
// }

// impl GenotypesBed {
//     pub fn from_file(path: &Path) -> Self {

//         let  = std::fs::read(path).expect("failed to read .bed file");
//     }
// }

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
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
pub struct Data {
    pub gen: Genotypes,
    pub phen: Phenotypes,
}

impl Data {
    pub fn new(gen: Genotypes, phen: Phenotypes) -> Self {
        Self { gen, phen }
    }

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
        self.gen.x.len()
    }

    pub fn num_markers_per_branch(&self) -> &Vec<usize> {
        &self.gen.num_markers_per_branch
    }

    pub fn num_markers_in_branch(&self, ix: usize) -> usize {
        self.gen.num_markers_per_branch[ix]
    }

    pub fn num_individuals(&self) -> usize {
        self.gen.num_individuals
    }

    pub fn standardize_x(&mut self) {
        self.gen.standardize();
    }

    pub fn x(&self) -> &Vec<Vec<f32>> {
        &self.gen.x
    }

    pub fn y(&self) -> &Vec<f32> {
        &self.phen.y
    }

    pub fn x_branch_af(&self, branch_ix: usize) -> Array<f32> {
        self.gen.x_branch_af(branch_ix)
    }

    pub fn y_af(&self) -> Array<f32> {
        self.phen.y_af()
    }
}

#[cfg(test)]
mod tests {
    // use crate::data::{Genotypes, GenotypesBuilder};
    use bed_reader::{Bed, ReadOptions};
    use std::env;
    use std::path::Path;

    // const SEED: u64 = 42;
    // const NB: usize = 1;
    // const NMPB: usize = 5;
    // const N: usize = 10;

    #[test]
    fn rust_bed_reader() {
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

    // fn make_test_gt() -> Genotypes {
    //     let mut gt = GenotypesBuilder::new()
    //         .with_seed(SEED)
    //         .with_random_x(vec![NMPB; NB], N, None)
    //         .build()
    //         .unwrap();
    //     gt.standardize();
    //     gt
    // }

    // #[test]
    // fn standardization() {
    //     let get = make_test_gt();

    // }
}
