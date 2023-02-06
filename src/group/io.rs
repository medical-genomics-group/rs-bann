//! Parsing functionality for common file types
use flate2::read::GzDecoder;
use serde::Serialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::marker::PhantomData;
use std::path::Path;
use std::str::FromStr;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Line starts with '#'")]
    CommentLine,
    #[error("Unknown feature found in gff")]
    UnknownGenomicFeature,
    #[error("Unknown chromosome")]
    UnknownChromosome,
    #[error("Unknown sex code")]
    UnknownSexCode,
}

// A human chromosome.
#[derive(PartialEq, PartialOrd, Eq, Hash, Copy, Clone, Debug, Serialize)]
pub enum Chromosome {
    One,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Ten,
    Eleven,
    Twelve,
    Thirteen,
    Fourteen,
    Fifteen,
    Sixteen,
    Seventeen,
    Eighteen,
    Nineteen,
    Twenty,
    TwentyOne,
    TwentyTwo,
    X,
    Y,
}

impl FromStr for Chromosome {
    type Err = Error;

    fn from_str(input: &str) -> Result<Chromosome, Self::Err> {
        match input {
            "1" => Ok(Chromosome::One),
            "2" => Ok(Chromosome::Two),
            "3" => Ok(Chromosome::Three),
            "4" => Ok(Chromosome::Four),
            "5" => Ok(Chromosome::Five),
            "6" => Ok(Chromosome::Six),
            "7" => Ok(Chromosome::Seven),
            "8" => Ok(Chromosome::Eight),
            "9" => Ok(Chromosome::Nine),
            "10" => Ok(Chromosome::Ten),
            "11" => Ok(Chromosome::Eleven),
            "12" => Ok(Chromosome::Twelve),
            "13" => Ok(Chromosome::Thirteen),
            "14" => Ok(Chromosome::Fourteen),
            "15" => Ok(Chromosome::Fifteen),
            "16" => Ok(Chromosome::Sixteen),
            "17" => Ok(Chromosome::Seventeen),
            "18" => Ok(Chromosome::Eighteen),
            "19" => Ok(Chromosome::Nineteen),
            "20" => Ok(Chromosome::Twenty),
            "21" => Ok(Chromosome::TwentyOne),
            "22" => Ok(Chromosome::TwentyTwo),
            "X" => Ok(Chromosome::X),
            "Y" => Ok(Chromosome::Y),
            _ => Err(Error::UnknownChromosome),
        }
    }
}

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
            _ => Err(Error::UnknownGenomicFeature),
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
            return Err(Error::CommentLine);
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

#[derive(Debug)]
pub struct BimEntry {
    pub ix: usize,
    pub chromosome: Chromosome,
    pub id: String,
    pub centimorgan: usize,
    pub position: usize,
    pub allele_1: String,
    pub allele_2: String,
}

impl IndexedEntry for BimEntry {
    fn from_str(s: &str, ix: usize) -> Self {
        let fields = s.split_whitespace().collect::<Vec<&str>>();
        Self {
            ix,
            chromosome: fields[0].parse().unwrap(),
            id: fields[1].to_owned(),
            centimorgan: fields[2]
                .parse()
                .expect("Failed to convert 3rd col entry in .bim to int"),
            position: fields[3]
                .parse()
                .expect("Failed to convert 4th col entry in .bim to int"),
            allele_1: fields[4].to_owned(),
            allele_2: fields[5].to_owned(),
        }
    }
}

impl BimEntry {
    pub fn chr(&self) -> &Chromosome {
        &self.chromosome
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn position(&self) -> usize {
        self.position
    }
}

enum FamPhenValue {
    Control,
    Case,
    Other,
}

impl FromStr for FamPhenValue {
    type Err = Error;

    fn from_str(input: &str) -> Result<FamPhenValue, Self::Err> {
        match input {
            "1" => Ok(FamPhenValue::Control),
            "2" => Ok(FamPhenValue::Case),
            _ => Ok(FamPhenValue::Other),
        }
    }
}

enum FamSex {
    Female,
    Male,
    Unknown,
}

impl FromStr for FamSex {
    type Err = Error;

    fn from_str(input: &str) -> Result<FamSex, Self::Err> {
        match input {
            "0" => Ok(FamSex::Unknown),
            "1" => Ok(FamSex::Male),
            "2" => Ok(FamSex::Female),
            _ => Err(Error::UnknownSexCode),
        }
    }
}

/// Entry of a .fam file.
///
/// According to the Plink 1.9 specs:
///
/// A text file with no header line, and one line per sample with the following six fields:
/// Family ID ('FID')
/// Within-family ID ('IID'; cannot be '0')
/// Within-family ID of father ('0' if father isn't in dataset)
/// Within-family ID of mother ('0' if mother isn't in dataset)
/// Sex code ('1' = male, '2' = female, '0' = unknown)
/// Phenotype value ('1' = control, '2' = case, '-9'/'0'/non-numeric = missing data if case/control)///
pub struct FamEntry {
    ix: usize,
    fid: usize,
    iid: usize,
    father_iid: usize,
    mother_iid: usize,
    sex: FamSex,
    phenotype_value: FamPhenValue,
}

impl IndexedEntry for FamEntry {
    fn from_str(s: &str, ix: usize) -> Self {
        let fields = s.split_whitespace().collect::<Vec<&str>>();
        Self {
            ix,
            fid: fields[0].parse().unwrap(),
            iid: fields[1]
                .parse()
                .expect("Failed to convert 2nd col entry in .fam to int"),
            father_iid: fields[2]
                .parse()
                .expect("Failed to convert 3rd col entry in .fam to int"),
            mother_iid: fields[3]
                .parse()
                .expect("Failed to convert 4th col entry in .fam to int"),
            sex: fields[4]
                .parse()
                .expect("Failed to convert 5th col entry in .fam to sex"),
            phenotype_value: fields[5]
                .parse()
                .expect("Failed to convert 6th col entry in .fam to phenotype value"),
        }
    }
}

pub trait IndexedEntry {
    fn from_str(s: &str, ix: usize) -> Self;
}

pub struct IndexedReader<T: IndexedEntry> {
    num_read: usize,
    reader: BufReader<File>,
    buffer: String,
    _phantom: PhantomData<T>,
}

impl<T: IndexedEntry> IndexedReader<T> {
    pub fn num_lines(fam_path: &Path) -> usize {
        let mut reader = IndexedReader::<T>::new(fam_path);
        while let Some(_) = reader.next_entry() {}
        reader.num_read
    }

    pub fn new(fam_path: &Path) -> Self {
        Self {
            num_read: 0,
            reader: BufReader::new(File::open(fam_path).unwrap()),
            buffer: String::new(),
            _phantom: PhantomData,
        }
    }

    pub fn next_entry(&mut self) -> Option<T> {
        self.buffer.clear();
        if let Ok(bytes_read) = self.reader.read_line(&mut self.buffer) {
            if bytes_read > 0 {
                self.num_read += 1;
                return Some(T::from_str(&self.buffer, self.last_entry_ix()));
            }
        }
        None
    }

    pub fn last_entry_ix(&self) -> usize {
        self.num_read - 1
    }
}
