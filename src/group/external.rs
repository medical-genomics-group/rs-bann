use super::grouping::MarkerGrouping;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Mapping from branch id to marker ids
pub struct ExternalGrouping {
    groups: HashMap<usize, Vec<usize>>,
    group_sizes: Vec<usize>,
}

impl ExternalGrouping {
    /// This assumes a two column file with columns: marker_ix, group_ix
    pub fn from_file(file: &Path) -> Self {
        let mut res = ExternalGrouping {
            groups: HashMap::new(),
            group_sizes: Vec::new(),
        };

        let file = File::open(file).expect("Failed to open file with external grouping");
        let mut reader = BufReader::new(file);
        let mut buffer = String::new();
        let mut line_fields = [0, 0];

        while let Ok(bytes_read) = reader.read_line(&mut buffer) {
            if bytes_read == 0 {
                break;
            }
            // TODO: this will panic if the file has to many line entries
            // and will cause undesired behaviour if too few line entries
            buffer
                .split_whitespace()
                .map(|e| e.parse::<usize>().unwrap())
                .enumerate()
                .for_each(|(ix, e)| line_fields[ix] = e);

            res.groups
                .entry(line_fields[1] as usize)
                .or_default()
                .push(line_fields[0]);

            buffer.clear();
        }

        let mut group_sizes: Vec<usize> = vec![0; res.groups.len()];
        res.groups
            .iter()
            .for_each(|(k, v)| group_sizes[*k] = v.len());
        res.group_sizes = group_sizes;

        res
    }
}

impl MarkerGrouping for ExternalGrouping {
    fn num_groups(&self) -> usize {
        self.groups.len()
    }

    fn group(&self, ix: usize) -> Option<&Vec<usize>> {
        self.groups.get(&ix)
    }

    fn group_sizes(&self) -> &[usize] {
        &self.group_sizes
    }
}
