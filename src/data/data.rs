use crate::data::phenotypes::Phenotypes;
use arrayfire::Array;

use super::genotypes::GroupedGenotypes;

#[derive(PartialEq, Debug)]
pub struct Data<T>
where
    T: GroupedGenotypes,
{
    pub gen: T,
    pub phen: Phenotypes,
}

impl<T: GroupedGenotypes> Data<T> {
    pub fn new(gen: T, phen: Phenotypes) -> Self {
        Self { gen, phen }
    }

    pub fn num_branches(&self) -> usize {
        self.gen.num_groups()
    }

    pub fn num_markers_per_branch(&self) -> &[usize] {
        self.gen.num_markers_per_group()
    }

    pub fn num_markers_in_branch(&self, ix: usize) -> usize {
        self.gen.num_markers_per_group()[ix]
    }

    pub fn num_individuals(&self) -> usize {
        self.gen.num_individuals()
    }

    pub fn y(&self) -> &[f32] {
        self.phen.y()
    }

    /// Returns marker data belonging to a given branch / ix in a af array.
    pub fn x_branch_af(&self, branch_ix: usize) -> Array<f32> {
        self.gen.x_group_af(branch_ix)
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

        let mut bed = Bed::new(bed_path).unwrap();
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
