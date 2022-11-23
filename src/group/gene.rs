use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

// TODO: Impl Reader trait with associated type for Entry
// Impl Entry trait with from_str method
struct GFFReader {
    last_entry_ix: usize,
    reader: BufReader<File>,
    buffer: String,
}

impl<'a> GFFReader {
    fn new(gff_path: &Path) -> Self {
        Self {
            last_entry_ix: 0,
            reader: BufReader::new(File::open(gff_path).unwrap()),
            buffer: String::new(),
        }
    }

    fn next_entry(&'a mut self) -> Option<GFFEntry<'a>> {
        if let Ok(bytes_read) = self.reader.read_line(&mut self.buffer) {
            if bytes_read > 0 {
                let res = Some(GFFEntry::from_str(&self.buffer, self.last_entry_ix));
                self.last_entry_ix += 1;
                return res;
            }
        }
        None
    }
}

struct BimEntry<'a> {
    ix: usize,
    chromosome: &'a str,
    id: &'a str,
    centimorgan: usize,
    position: usize,
    allele_1: &'a str,
    allele_2: &'a str,
}

impl<'a> BimEntry<'a> {
    fn from_str(s: &'a str, ix: usize) -> Self {
        let fields = s.split_whitespace().collect::<Vec<&str>>();
        Self {
            ix,
            chromosome: fields[0],
            id: fields[1],
            centimorgan: fields[2].parse().unwrap(),
            position: fields[3].parse().unwrap(),
            allele_1: fields[4],
            allele_2: fields[5],
        }
    }

    fn chr(&self) -> &str {
        self.chromosome
    }

    fn id(&self) -> &str {
        self.id
    }

    fn position(&self) -> usize {
        self.position
    }
}

struct BimReader {
    last_entry_ix: usize,
    reader: BufReader<File>,
    buffer: String,
}

impl<'a> BimReader {
    fn new(bim_path: &Path) -> Self {
        Self {
            last_entry_ix: 0,
            reader: BufReader::new(File::open(bim_path).unwrap()),
            buffer: String::new(),
        }
    }

    fn next_entry(&'a mut self) -> Option<BimEntry<'a>> {
        if let Ok(bytes_read) = self.reader.read_line(&mut self.buffer) {
            if bytes_read > 0 {
                let res = Some(BimEntry::from_str(&self.buffer, self.last_entry_ix));
                self.last_entry_ix += 1;
                return res;
            }
        }
        None
    }
}

/// Grouping of SNPs.
/// All SNPs within a gene or within a given distance to it are part of the same group.
pub struct GeneGrouping {
    pub groups: HashMap<usize, Vec<isize>>,
}

impl GeneGrouping {
    pub fn from_gff(_gff_file: &Path, _bim_file: &Path) -> Self {
        Self {
            groups: HashMap::new(),
        }
    }
}
