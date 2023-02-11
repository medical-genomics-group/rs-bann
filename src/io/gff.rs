use crate::error::Error;
use crate::io::chromosome::Chromosome;
use flate2::read::GzDecoder;
use serde::Serialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::str::FromStr;

/// A genomic feature
#[derive(Serialize, Debug)]
pub enum Feature {
    Gene,
    Exon,
    Intron,
    Pseudogene,
    Transcipt,
    MIRNA,
    CDS,
    Silencer,
    LNCRNA,
    MRNA,
}

impl FromStr for Feature {
    type Err = Error;

    fn from_str(input: &str) -> Result<Feature, Self::Err> {
        match input {
            "gene" => Ok(Feature::Gene),
            "exon" => Ok(Feature::Exon),
            "intron" => Ok(Feature::Intron),
            "pseudogene" => Ok(Feature::Pseudogene),
            "transcipt" => Ok(Feature::Transcipt),
            "miRNA" => Ok(Feature::MIRNA),
            "CDS" => Ok(Feature::CDS),
            "silencer" => Ok(Feature::Silencer),
            "lnc_RNA" => Ok(Feature::LNCRNA),
            "mRNA" => Ok(Feature::MRNA),
            _ => Err(Error::GFFUnknownGenomicFeature),
        }
    }
}

#[derive(Serialize, Debug)]
pub struct GFFEntry {
    pub chromosome: Chromosome,
    pub source: String,
    pub feature: Feature,
    pub start: usize,
    pub end: usize,
    pub score: String,
    pub strand: String,
    pub frame: String,
    pub attribute: String,
}

impl FromStr for GFFEntry {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.starts_with('#') {
            return Err(Error::GFFCommentLine);
        }
        let fields = s.split_whitespace().collect::<Vec<&str>>();
        Ok(Self {
            chromosome: fields[0].parse()?,
            source: fields[1].to_owned(),
            feature: fields[2].parse()?,
            start: fields[3].parse().unwrap(),
            end: fields[4].parse().unwrap(),
            score: fields[5].to_owned(),
            strand: fields[6].to_owned(),
            frame: fields[7].to_owned(),
            attribute: fields[8].to_owned(),
        })
    }
}

pub trait GFFRead {
    fn next_entry(&mut self) -> Option<GFFEntry>;
}

pub struct GzGFFReader {
    num_read: usize,
    reader: BufReader<GzDecoder<File>>,
    buffer: String,
}

impl GzGFFReader {
    pub fn new(gff_path: &Path) -> Self {
        Self {
            num_read: 0,
            reader: BufReader::new(GzDecoder::new(File::open(gff_path).unwrap())),
            buffer: String::new(),
        }
    }
}

impl GFFRead for GzGFFReader {
    fn next_entry(&mut self) -> Option<GFFEntry> {
        self.buffer.clear();
        if let Ok(bytes_read) = self.reader.read_line(&mut self.buffer) {
            if bytes_read > 0 {
                if let Ok(entry) = self.buffer.parse::<GFFEntry>() {
                    self.num_read += 1;
                    return Some(entry);
                }
                return self.next_entry();
            }
        }
        None
    }
}

pub struct GFFReader {
    // number of entries that have been successfully read
    num_read: usize,
    reader: BufReader<File>,
    buffer: String,
}

impl GFFReader {
    pub fn new(gff_path: &Path) -> Self {
        Self {
            num_read: 0,
            reader: BufReader::new(File::open(gff_path).unwrap()),
            buffer: String::new(),
        }
    }
}

impl GFFRead for GFFReader {
    fn next_entry(&mut self) -> Option<GFFEntry> {
        self.buffer.clear();
        if let Ok(bytes_read) = self.reader.read_line(&mut self.buffer) {
            if bytes_read > 0 {
                if let Ok(entry) = self.buffer.parse::<GFFEntry>() {
                    self.num_read += 1;
                    return Some(entry);
                }
                return self.next_entry();
            }
        }
        None
    }
}
