pub trait MarkerGrouping {
    fn num_groups(&self) -> usize;
    fn group(&self, ix: usize) -> Option<&Vec<isize>>;

    fn group_size(&self, ix: usize) -> Option<usize> {
        self.group(ix).map(|g| g.len())
    }

    fn group_sizes(&self) -> Vec<usize> {
        (0..self.num_groups())
            .map(|i| self.group_size(i).unwrap())
            .collect()
    }
}
