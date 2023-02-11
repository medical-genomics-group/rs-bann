use crate::group::grouping::MarkerGrouping;
use crate::{error::Error, io::bed::BedVM};

use arrayfire::{dim4, Array};
use bed_reader::{Bed as ExternBed, ReadOptions};
use bincode::{deserialize_from, serialize_into};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Binomial, Distribution, Uniform};
use serde::{Deserialize, Serialize};
use serde_json::to_writer;
use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

pub trait GroupedGenotypes {
    fn num_individuals(&self) -> usize;
    fn num_markers_per_group(&self) -> &[usize];
    fn num_groups(&self) -> usize;
    fn x_group_af(&self, group_ix: usize) -> Array<f32>;
}

pub struct GenotypesBuilder {
    x: Option<Vec<Vec<f32>>>,
    num_individuals: Option<usize>,
    num_markers_per_group: Option<Vec<usize>>,
    num_groups: Option<usize>,
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
            num_markers_per_group: None,
            num_groups: None,
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
        num_markers_per_group: Vec<usize>,
        num_individuals: usize,
        mafs: Option<Vec<f32>>,
    ) -> Self {
        let num_groups = num_markers_per_group.len();
        let mut x: Vec<Vec<f32>> = Vec::new();
        let mut x_means: Vec<Vec<f32>> = Vec::new();
        let mut x_stds: Vec<Vec<f32>> = Vec::new();
        for branch_ix in 0..num_groups {
            let nmpb = num_markers_per_group[branch_ix];
            x.push(vec![0.0; nmpb * num_individuals]);
            x_means.push(vec![0.0; nmpb]);
            x_stds.push(vec![0.0; nmpb]);
            for marker_ix in 0..nmpb {
                loop {
                    let maf = if let Some(v) = &mafs {
                        assert!(
                            v[num_groups * branch_ix + marker_ix] != 0.0,
                            "maf of 0 it not allowed in simulation"
                        );
                        v[num_groups * branch_ix + marker_ix]
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
        self.num_groups = Some(num_groups);
        self.num_markers_per_group = Some(num_markers_per_group);
        self.num_individuals = Some(num_individuals);
        self
    }

    // TODO: compute stds and means in here, too
    pub fn with_x(
        mut self,
        x: Vec<Vec<f32>>,
        num_markers_per_group: Vec<usize>,
        num_individuals: usize,
    ) -> Self {
        self.x = Some(x);
        self.num_groups = Some(num_markers_per_group.len());
        self.num_markers_per_group = Some(num_markers_per_group);
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
        let mut num_markers_per_group = Vec::new();
        let mut num_individuals = 0;
        let num_groups = grouping.num_groups();
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
                .sid_index(
                    grouping
                        .group(gix)
                        .unwrap()
                        .iter()
                        .map(|v| *v as isize)
                        .collect::<Vec<isize>>(),
                )
                .read(&mut bed)
                .unwrap();
            // TODO: make sure that t().iter() actually is column-major iteration.
            num_individuals = val.nrows();
            x.push(val.t().iter().copied().collect::<Vec<f32>>());
            x_means.push(val.t().mean_axis(ndarray::Axis(0)).unwrap().to_vec());
            x_stds.push(val.t().std_axis(ndarray::Axis(0), 1f32).to_vec());
            num_markers_per_group.push(grouping.group_size(gix).unwrap());
        }
        self.x = Some(x);
        self.means = Some(x_means);
        self.stds = Some(x_stds);
        self.num_markers_per_group = Some(num_markers_per_group);
        self.num_individuals = Some(num_individuals);
        self.num_groups = Some(num_groups);
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
            num_markers_per_group: self.num_markers_per_group.unwrap(),
            num_individuals: self.num_individuals.unwrap(),
            num_groups: self.num_groups.unwrap(),
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
    num_markers_per_group: Vec<usize>,
    num_groups: usize,
    means: Option<Vec<Vec<f32>>>,
    stds: Option<Vec<Vec<f32>>>,
    standardized: bool,
}

impl GroupedGenotypes for Genotypes {
    fn num_groups(&self) -> usize {
        self.num_groups
    }

    fn num_individuals(&self) -> usize {
        self.num_individuals
    }

    fn num_markers_per_group(&self) -> &[usize] {
        &self.num_markers_per_group
    }

    fn x_group_af(&self, branch_ix: usize) -> Array<f32> {
        Array::new(
            &self.x[branch_ix],
            dim4!(
                self.num_individuals as u64,
                self.num_markers_per_group[branch_ix] as u64
            ),
        )
    }
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

    pub fn af_branch_data(&self, branch_ix: usize) -> Array<f32> {
        Array::new(
            &self.x[branch_ix],
            dim4!(
                self.num_individuals as u64,
                self.num_markers_per_group[branch_ix] as u64
            ),
        )
    }

    pub fn standardize(&mut self) {
        if !self.standardized {
            if self.means.is_some() && self.stds.is_some() {
                let means = self.means.as_ref().unwrap();
                let stds = self.stds.as_ref().unwrap();
                for branch_ix in 0..self.num_groups {
                    for marker_ix in 0..self.num_markers_per_group[branch_ix] {
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
}

/// Genotype information stored in the Plink .bed format.
pub struct CompressedGenotypes<T: MarkerGrouping> {
    bed: BedVM,
    groups: T,
}

impl<T: MarkerGrouping> CompressedGenotypes<T> {
    pub fn new(bed: BedVM, groups: T) -> Self {
        Self { bed, groups }
    }
}

impl<T: MarkerGrouping> GroupedGenotypes for CompressedGenotypes<T> {
    fn num_groups(&self) -> usize {
        self.groups.num_groups()
    }

    fn num_individuals(&self) -> usize {
        self.bed.num_individuals()
    }

    fn num_markers_per_group(&self) -> &[usize] {
        self.groups.group_sizes()
    }

    fn x_group_af(&self, group_ix: usize) -> Array<f32> {
        self.bed.get_submatrix_af_standardized(
            self.groups.group(group_ix).expect("Invalid group index"),
        )
    }
}
