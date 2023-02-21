use crate::group::grouping::MarkerGrouping;
use crate::io::bed::BedVM;

use arrayfire::Array;
use std::path::Path;

pub trait GroupedGenotypes {
    fn num_individuals(&self) -> usize;
    fn num_markers_per_group(&self) -> &[usize];
    fn num_groups(&self) -> usize;
    fn x_group_af(&self, group_ix: usize) -> Array<f32>;
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

    pub fn to_file(&self, stem: &Path) {
        self.bed.to_file(stem);
        self.groups.to_file(stem);
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
