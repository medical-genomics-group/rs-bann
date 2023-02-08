use super::grouping::MarkerGrouping;
use crate::io::{
    bim::BimEntry,
    gff::{Feature, GFFEntry, GFFRead, GFFReader, GzGFFReader},
    indexed_read::IndexedReader,
};
use serde_json::to_writer_pretty;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::fs::File;
use std::path::Path;

enum RelativePosition {
    BimIsAhead,
    GFFIsAhead,
    Overlap,
}

/// Grouping of SNPs.
/// All SNPs within a gene or within a given distance to it are part of the same group.
pub struct GeneGrouping {
    pub groups: HashMap<usize, Vec<usize>>,
    pub meta: HashMap<usize, GFFEntry>,
    pub group_sizes: Vec<usize>,
}

impl MarkerGrouping for GeneGrouping {
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

impl GeneGrouping {
    pub fn from_gff(
        gff_file: &Path,
        bim_file: &Path,
        margin: usize,
        min_group_size: usize,
    ) -> Self {
        // TODO: check for correct extensions here (gff or gff3 or gff.gz or gff3.gz)
        let mut gff_reader = if gff_file
            .extension()
            .expect("gff file has unknown or no extension")
            == "gz"
        {
            Box::new(GzGFFReader::new(gff_file)) as Box<dyn GFFRead>
        } else {
            Box::new(GFFReader::new(gff_file)) as Box<dyn GFFRead>
        };
        let mut bim_reader = IndexedReader::new(bim_file);
        let mut bim_buffer: VecDeque<BimEntry> = VecDeque::new();
        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut meta: HashMap<usize, GFFEntry> = HashMap::new();
        let mut group_id = 0;

        while let Some(gff_entry) = gff_reader.next_entry() {
            if let Feature::Gene = gff_entry.feature {
                // remove bim entries that are before current window start
                while let Some(bim_entry) = bim_buffer.get(0) {
                    if let RelativePosition::GFFIsAhead =
                        GeneGrouping::relative_position(bim_entry, &gff_entry, margin)
                    {
                        // bim entry is out of range
                        bim_buffer.pop_front();
                    } else {
                        // rest of buffer cannot be behind
                        break;
                    }
                }
                // add buffer entries that are within range to group
                for bim_entry in &bim_buffer {
                    if let RelativePosition::Overlap =
                        GeneGrouping::relative_position(bim_entry, &gff_entry, margin)
                    {
                        groups.entry(group_id).or_default().push(bim_entry.ix);
                    }
                }
                // check next bim entries
                while let Some(bim_entry) = bim_reader.next_entry() {
                    match GeneGrouping::relative_position(&bim_entry, &gff_entry, margin) {
                        RelativePosition::BimIsAhead => {
                            // potentially needed for next feature
                            bim_buffer.push_back(bim_entry);
                            break;
                        }
                        RelativePosition::GFFIsAhead => {}
                        RelativePosition::Overlap => {
                            groups.entry(group_id).or_default().push(bim_entry.ix);
                            // needed for next feature
                            bim_buffer.push_back(bim_entry);
                        }
                    }
                }
                // only increment group_id if any snps were added
                if groups.contains_key(&group_id) {
                    meta.insert(group_id, gff_entry);
                    group_id += 1;
                }
            }
        }

        if min_group_size > 1 {
            groups.retain(|_, v| v.len() >= min_group_size);
        }

        let mut group_sizes: Vec<usize> = vec![0; groups.len()];
        groups.iter().for_each(|(k, v)| group_sizes[*k] = v.len());

        Self {
            groups,
            meta,
            group_sizes,
        }
    }

    fn relative_position(snp: &BimEntry, feature: &GFFEntry, margin: usize) -> RelativePosition {
        if snp.chromosome > feature.chromosome {
            RelativePosition::BimIsAhead
        } else if feature.chromosome > snp.chromosome {
            RelativePosition::GFFIsAhead
        } else if feature.chromosome == snp.chromosome {
            let window_start = if feature.start < margin {
                0
            } else {
                feature.start - margin
            };
            let window_end = feature.end + margin;
            if window_start > snp.position {
                RelativePosition::GFFIsAhead
            } else if snp.position > window_end {
                RelativePosition::BimIsAhead
            } else {
                RelativePosition::Overlap
            }
        } else {
            panic!("Invalid chromosome comparison!");
        }
    }

    pub fn meta_to_file(&self, path: &Path) {
        to_writer_pretty(File::create(path).unwrap(), &self.meta).unwrap();
    }
}
