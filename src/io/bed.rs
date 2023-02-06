use crate::error::Error;
use crate::io::{
    bed_lookup_tables::BED_LOOKUP_GENOTYPE, bim::BimEntry, fam::FamEntry,
    indexed_read::IndexedReader,
};
use arrayfire::{dim4, Array};
use std::io::{Read, Seek};
use std::path::PathBuf;

const BED_SIGNATURE_LENGTH: usize = 3;

enum BedSignature {
    SampleMajor,
    VariantMajor,
}

impl BedSignature {
    fn from_bed_file(path: &PathBuf) -> Result<Self, Error> {
        let bed_file = std::fs::File::open(path).expect("Failed to open .bed file!");
        let bed_sig_bytes: Vec<u8> = bed_file
            .bytes()
            .take(BED_SIGNATURE_LENGTH)
            .map(|r| r.expect("Failed to read byte from bed!"))
            .collect();
        Self::from_bytes(&bed_sig_bytes)
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if bytes[0] != 0x6c {
            return Err(Error::BedFalseFirstByte);
        }
        if bytes[1] != 0x1b {
            return Err(Error::BedFalseSecondByte);
        }
        match bytes[2] {
            0x00 => Ok(Self::SampleMajor),
            0x01 => Ok(Self::VariantMajor),
            _ => Err(Error::BedFalseThirdByte),
        }
    }
}

/// Variant Major (i.e. column major) bed file in memory.
struct BedVM {
    signature: BedSignature,
    /// .bed data without signature
    data: Vec<u8>,
    num_individuals: usize,
    num_markers: usize,
    num_bytes_per_col: usize,
    // pairs of bits without info in the last byte per col
    padding: usize,
}

impl BedVM {
    /// Reads .bed file from disc.
    /// Determines number of markers and individuals from .bim and .fam files with the same filestem as the .bed.
    /// Checks if .bed signature is valid.
    fn from_file(stem: &PathBuf) -> Self {
        let mut bed_path = stem.clone();
        bed_path.set_extension("bed");
        let signature = BedSignature::from_bed_file(&bed_path).expect("Unexpected .bed signature");
        if let BedSignature::SampleMajor = signature {
            panic!("SampleMajor .bed formats are not supported at the moment. Try converting to VariantMajor format.")
        }

        let mut bed_file = std::fs::File::open(&bed_path).expect("Failed to open .bed file");
        bed_file
            .seek(std::io::SeekFrom::Start(
                BED_SIGNATURE_LENGTH.try_into().unwrap(),
            ))
            .unwrap();
        let mut data = Vec::new();
        bed_file
            .read_to_end(&mut data)
            .expect("Error while reading .bed file");

        let mut bim_path = stem.clone();
        bim_path.set_extension("bim");
        let num_markers = IndexedReader::<BimEntry>::num_lines(&bim_path);
        let mut fam_path = stem.clone();
        fam_path.set_extension("fam");
        let num_individuals = IndexedReader::<FamEntry>::num_lines(&fam_path);

        let mut num_bytes_per_col = num_individuals / 4;
        let padding = num_individuals % 4;
        if padding != 0 {
            num_bytes_per_col += 1;
        }

        Self {
            signature,
            data,
            num_individuals,
            num_markers,
            num_bytes_per_col,
            padding,
        }
    }

    /// Assumes VariantMajor format and no missing data.
    fn get_cols_af(&self, col_ixs: &[usize]) -> Vec<Array<f32>> {
        let mut res = Vec::new();
        for col_ix in col_ixs {
            let start_ix = col_ix * self.num_individuals;
            let end_ix = start_ix + self.num_individuals;
            let mut vals = Vec::with_capacity(self.num_bytes_per_col * 4);
            let col_data = &self.data[start_ix..end_ix];
            for b in col_data.iter() {
                let lu_ix = 4 * (*b as usize);
                vals.extend_from_slice(&BED_LOOKUP_GENOTYPE[lu_ix..lu_ix + 4]);
            }
            vals.truncate(self.num_individuals);
            res.push(Array::new(
                &vals,
                dim4!(self.num_individuals.try_into().unwrap()),
            ));
        }
        res
    }
}
