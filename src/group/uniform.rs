use super::grouping::MarkerGrouping;

pub struct UniformGrouping {
    num_groups: usize,
    num_markers_per_group: usize,
    group_sizes: Vec<usize>,
    groups: Vec<Vec<usize>>,
}

impl UniformGrouping {
    pub fn new(num_groups: usize, num_markers_per_group: usize) -> Self {
        Self {
            num_groups,
            num_markers_per_group,
            group_sizes: vec![num_markers_per_group; num_groups],
            groups: (0..num_groups)
                .map(|gix| {
                    ((gix * num_markers_per_group)..(gix * (num_markers_per_group + 1)))
                        .collect::<Vec<usize>>()
                })
                .collect(),
        }
    }
}

impl MarkerGrouping for UniformGrouping {
    fn num_groups(&self) -> usize {
        self.num_groups
    }

    fn group(&self, ix: usize) -> Option<&Vec<usize>> {
        if ix < self.num_groups {
            Some(&self.groups[ix])
        } else {
            None
        }
    }

    fn group_sizes(&self) -> &[usize] {
        &self.group_sizes
    }
}
