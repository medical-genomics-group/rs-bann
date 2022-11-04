use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

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

    /// Writes the grouping to a two column tab separated file where the
    /// first column is the marker index and the second the group index.
    fn to_file(&self, path: &Path) {
        let mut writer =
            BufWriter::new(File::open(path).expect("Could not open file to save grouping"));
        for group_index in 0..self.num_groups() {
            // cannot panic, controlled range
            for marker_index in self.group(group_index).unwrap() {
                writer
                    .write(format!("{}\t{}\n", marker_index, group_index).as_bytes())
                    .unwrap();
            }
        }
        writer.flush().unwrap();
    }
}
