pub trait MarkerGrouping {
    fn num_groups(&self) -> usize;
    fn group(&self, ix: usize) -> Option<&Vec<usize>>;
}
