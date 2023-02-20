use super::bed::{BedBinaryFileset, PlinkBinaryFileset};
use crate::error::Error;
use crate::io::{bim::BimEntry, fam::FamEntry, indexed_read::IndexedReader};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug)]
pub struct BedDims {
    num_individuals: usize,
    num_markers: usize,
}

impl BedDims {
    pub fn from_dims_file(stem: &Path) -> Result<Self, Error> {
        let path = BedBinaryFileset::new(stem).dims();
        let file = fs::File::open(path)?;
        let mut buffer = BufReader::new(file);
        let mut first_line = String::new();
        let _ = buffer.read_line(&mut first_line);
        let fields = first_line.split_ascii_whitespace().collect::<Vec<&str>>();
        Ok(Self {
            num_individuals: fields[0].parse()?,
            num_markers: fields[1].parse()?,
        })
    }

    pub fn from_plink_fileset(stem: &Path) -> Result<Self, Error> {
        let bfiles = PlinkBinaryFileset::new(stem);
        Ok(Self {
            num_individuals: IndexedReader::<FamEntry>::num_lines(&bfiles.fam())?,
            num_markers: IndexedReader::<BimEntry>::num_lines(&bfiles.bim())?,
        })
    }

    pub fn num_individuals(&self) -> usize {
        self.num_individuals
    }
    pub fn num_markers(&self) -> usize {
        self.num_markers
    }
}
