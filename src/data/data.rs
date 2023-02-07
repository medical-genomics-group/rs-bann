use crate::data::genotypes::Genotypes;
use crate::data::phenotypes::Phenotypes;
use arrayfire::Array;
use bincode::{deserialize_from, serialize_into};
use serde::{Deserialize, Serialize};
use serde_json::to_writer;
use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

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
        self.gen.x().len()
    }

    pub fn num_markers_per_branch(&self) -> &Vec<usize> {
        &self.gen.num_markers_per_branch()
    }

    pub fn num_markers_in_branch(&self, ix: usize) -> usize {
        self.gen.num_markers_per_branch()[ix]
    }

    pub fn num_individuals(&self) -> usize {
        self.gen.num_individuals()
    }

    pub fn standardize_x(&mut self) {
        self.gen.standardize();
    }

    pub fn x(&self) -> &Vec<Vec<f32>> {
        &self.gen.x()
    }

    pub fn y(&self) -> &[f32] {
        self.phen.y()
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
