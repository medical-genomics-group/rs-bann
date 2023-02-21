use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

pub trait MarkerGrouping {
    fn num_groups(&self) -> usize;
    fn group(&self, ix: usize) -> Option<&Vec<usize>>;
    fn group_sizes(&self) -> &[usize];

    fn group_size(&self, ix: usize) -> Option<usize> {
        self.group(ix).map(|g| g.len())
    }

    /// Writes the grouping to a two column tab separated file where the
    /// first column is the marker index and the second the group index.
    fn to_file(&self, stem: &Path) {
        let path = stem.with_extension("groups");
        let mut writer =
            BufWriter::new(File::create(path).expect("Could not open file to save grouping"));
        for group_index in 0..self.num_groups() {
            // cannot panic, controlled range
            for marker_index in self.group(group_index).unwrap() {
                writer
                    .write_all(format!("{}\t{}\n", marker_index, group_index).as_bytes())
                    .unwrap();
            }
        }
        writer.flush().unwrap();
    }
}
