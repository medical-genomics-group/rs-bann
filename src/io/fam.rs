use crate::error::Error;
use crate::io::indexed_read::IndexedEntry;
use std::str::FromStr;

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
            _ => Err(Error::FamUnknownSexCode),
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
