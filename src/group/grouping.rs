pub trait MarkerGrouping {
    fn num_groups(&self) -> usize;
    fn group(&self, ix: usize) -> Option<&Vec<isize>>;

    fn group_size(&self, ix: usize) -> Option<usize> {
        self.group(ix).map(|g| g.len())
    }
}
