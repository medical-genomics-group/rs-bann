use crate::error::Error;
use crate::io::{
    bed_lookup_tables::BED_LOOKUP_GENOTYPE, bim::BimEntry, fam::FamEntry,
    indexed_read::IndexedReader,
};
use arrayfire::{dim4, Array};
use log::warn;
use rand_distr::num_traits::Pow;
use std::io::{Read, Seek};
use std::path::{Path, PathBuf};

const BED_SIGNATURE_LENGTH: usize = 3;
const BED_VM_SIGNATURE: [u8; 3] = [0x6c, 0x1b, 0x01];
const BED_SM_SIGNATURE: [u8; 3] = [0x6c, 0x1b, 0x00];

/// Paths of a set of .bed, .bim, .fam files
pub struct PlinkBinaryFileset {
    stem: PathBuf,
}

impl PlinkBinaryFileset {
    pub fn new(path: &Path) -> Self {
        Self {
            stem: path.to_owned(),
        }
    }

    pub fn bed(&self) -> PathBuf {
        self.stem_with_extension("bed")
    }

    pub fn bim(&self) -> PathBuf {
        self.stem_with_extension("bim")
    }

    pub fn fam(&self) -> PathBuf {
        self.stem_with_extension("fam")
    }

    fn stem_with_extension(&self, ext: &str) -> PathBuf {
        let mut path = self.stem.clone();
        path.set_extension(ext);
        path
    }
}

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
        if bytes[0] != BED_VM_SIGNATURE[0] {
            return Err(Error::BedFalseFirstByte);
        }
        if bytes[1] != BED_VM_SIGNATURE[1] {
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
///
/// This struct does not handle NAs correctly. NAs should be imputed / removed beforehand.
pub struct BedVM {
    /// .bed data without signature
    data: Vec<u8>,
    col_means: Vec<f32>,
    col_stds: Vec<f32>,
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
    pub fn from_file(stem: &Path) -> Self {
        let bfiles = PlinkBinaryFileset::new(stem);
        let signature =
            BedSignature::from_bed_file(&bfiles.bed()).expect("Unexpected .bed signature");
        if let BedSignature::SampleMajor = signature {
            panic!("SampleMajor .bed formats are not supported at the moment. Try converting to VariantMajor format.")
        }

        let mut bed_file = std::fs::File::open(bfiles.bed()).expect("Failed to open .bed file");
        bed_file
            .seek(std::io::SeekFrom::Start(
                BED_SIGNATURE_LENGTH.try_into().unwrap(),
            ))
            .unwrap();
        let mut data = Vec::new();
        bed_file
            .read_to_end(&mut data)
            .expect("Error while reading .bed file");

        let num_markers = IndexedReader::<BimEntry>::num_lines(&bfiles.bim());
        let num_individuals = IndexedReader::<FamEntry>::num_lines(&bfiles.fam());

        let mut num_bytes_per_col = num_individuals / 4;
        let padding = num_individuals % 4;
        if padding != 0 {
            num_bytes_per_col += 1;
        }

        let mut res = Self {
            data,
            col_means: vec![0.0; num_markers],
            col_stds: vec![0.0; num_markers],
            num_individuals,
            num_markers,
            num_bytes_per_col,
            padding,
        };

        for col_ix in 0..num_markers {
            let vals = &res.get_cols(&[col_ix])[0];
            let mean: f32 = vals.iter().sum::<f32>() / num_individuals as f32;
            let std: f32 = (vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>()
                / num_individuals as f32)
                .sqrt();
            res.col_means[col_ix] = mean;
            res.col_stds[col_ix] = std;
            if std == 0.0 {
                warn!("No variation in marker {:?}; This might lead to division by zero if accessing standardized marker data", col_ix);
            }
        }

        res
    }

    /// Decompress and load columns onto device.
    pub fn get_cols_af(&self, col_ixs: &[usize]) -> Vec<Array<f32>> {
        let mut res = Vec::new();
        for col_ix in col_ixs {
            let start_ix = col_ix * self.num_bytes_per_col;
            let end_ix = start_ix + self.num_bytes_per_col;
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

    pub fn get_cols(&self, col_ixs: &[usize]) -> Vec<Vec<f32>> {
        let mut res = Vec::new();
        for col_ix in col_ixs {
            let start_ix = col_ix * self.num_bytes_per_col;
            let end_ix = start_ix + self.num_bytes_per_col;
            let mut vals = Vec::with_capacity(self.num_bytes_per_col * 4);
            let col_data = &self.data[start_ix..end_ix];
            for b in col_data.iter() {
                let lu_ix = 4 * (*b as usize);
                vals.extend_from_slice(&BED_LOOKUP_GENOTYPE[lu_ix..lu_ix + 4]);
            }
            vals.truncate(self.num_individuals);
            res.push(vals);
        }
        res
    }

    pub fn get_cols_af_standardized(&self, col_ixs: &[usize]) -> Vec<Array<f32>> {
        let mut res = Vec::new();
        for col_ix in col_ixs {
            let start_ix = col_ix * self.num_bytes_per_col;
            let end_ix = start_ix + self.num_bytes_per_col;
            let mut vals = Vec::with_capacity(self.num_bytes_per_col * 4);
            let col_data = &self.data[start_ix..end_ix];
            for b in col_data.iter() {
                let lu_ix = 4 * (*b as usize);
                vals.extend_from_slice(&BED_LOOKUP_GENOTYPE[lu_ix..lu_ix + 4]);
            }
            vals.truncate(self.num_individuals);
            res.push(
                (Array::new(&vals, dim4!(self.num_individuals.try_into().unwrap()))
                    - self.col_means[*col_ix])
                    / self.col_stds[*col_ix],
            );
        }
        res
    }

    pub fn get_submatrix_af_standardized(&self, col_ixs: &[usize]) -> Array<f32> {
        let mut all_vals: Vec<f32> = Vec::new();
        let mut col_means: Vec<f32> = Vec::new();
        let mut col_stds: Vec<f32> = Vec::new();
        for col_ix in col_ixs {
            let start_ix = col_ix * self.num_bytes_per_col;
            let end_ix = start_ix + self.num_bytes_per_col;
            let mut vals = Vec::with_capacity(self.num_bytes_per_col * 4);
            let col_data = &self.data[start_ix..end_ix];
            for b in col_data.iter() {
                let lu_ix = 4 * (*b as usize);
                vals.extend_from_slice(&BED_LOOKUP_GENOTYPE[lu_ix..lu_ix + 4]);
            }
            vals.truncate(self.num_individuals);
            all_vals.append(&mut vals);
            col_means.push(self.col_means[*col_ix]);
            col_stds.push(self.col_stds[*col_ix]);
        }
        let tiling_dims = dim4!(self.num_individuals as u64);
        let means = arrayfire::tile(
            &Array::new(&col_means, dim4!(1, col_ixs.len() as u64)),
            tiling_dims,
        );
        let stds = arrayfire::tile(
            &Array::new(&col_stds, dim4!(1, col_ixs.len() as u64)),
            tiling_dims,
        );
        let subm_dims = dim4!(self.num_individuals as u64, col_ixs.len() as u64);
        let raw_arr = Array::new(&all_vals, subm_dims);
        (raw_arr - means) / stds
    }

    /// Decompresses data to f32.
    /// Removes padding from columns.
    pub fn data_f32(&self) -> Vec<f32> {
        let mut res = Vec::with_capacity(self.num_individuals * self.num_markers);
        for (ix, byte) in self.data.iter().enumerate() {
            let lu_ix = 4 * (*byte as usize);
            res.extend_from_slice(&BED_LOOKUP_GENOTYPE[lu_ix..lu_ix + 4]);
            res.truncate(self.num_individuals * (ix + 1));
        }
        res
    }

    pub fn num_individuals(&self) -> usize {
        self.num_individuals
    }

    pub fn num_markers(&self) -> usize {
        self.num_markers
    }
}

pub(crate) struct BedVMRandom {
    /// .bed data without signature
    data: Vec<u8>,
    col_means: Vec<f32>,
    col_stds: Vec<f32>,
    num_individuals: usize,
    num_markers: usize,
    num_bytes_per_col: usize,
}

// impl BedVMRandom {
//     pub fn new(num_individuals: usize, num_markers: usize, mafs: Option<f32>) {
//         unimplemented!()
//     }
// }

// There is no handing of NA here!
const BED_VALUE_MAPPING: [u8; 3] = [0x11, 0x10, 0x00];

fn chunkf32_to_byte(v: &[f32]) -> u8 {
    let mut res: u8 = 0;
    for (i, val) in v.iter().rev().enumerate() {
        assert!(
            i < 4,
            "Slice with more than four values in bed byte conversion!",
        );
        res <<= 2;
        res += BED_VALUE_MAPPING[*val as usize];
    }
    res
}

fn vecf32_to_bed(v: &[f32], bed_data: &mut Vec<u8>) {
    v.chunks(4)
        .map(chunkf32_to_byte)
        .for_each(|b| bed_data.push(b));
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::af_helpers::to_host;

    use super::BedVM;

    fn make_test_bed_vm() -> BedVM {
        let base_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let base_path = Path::new(&base_dir);
        let bed_path = base_path.join("resources/test/small");
        BedVM::from_file(&bed_path)
    }

    #[test]
    fn bed_vm_from_file() {
        let bed_vm = make_test_bed_vm();
        assert_eq!(bed_vm.num_individuals, 20);
        assert_eq!(bed_vm.num_markers, 11);
        let col_major_mat: Vec<f32> = vec![
            0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 2., 0., 1., 0.,
            1., 0., 0., 2., 0., 0., 1., 1., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1.,
            1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 1., 1., 1., 2., 0., 1., 1.,
            1., 1., 2., 0., 0., 1., 2., 1., 0., 1., 2., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
            1., 0., 0., 0., 0., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 0., 1., 0., 1., 2., 2., 1.,
            1., 1., 2., 1., 1., 1., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 1., 0., 0., 0., 2.,
            0., 0., 0., 0., 0., 1., 0., 1., 1., 2., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 0., 1., 1., 0., 1., 1., 0., 1., 0., 0., 2., 1., 1., 1., 1., 0., 0., 1., 1., 0., 0.,
        ];
        assert_eq!(bed_vm.data_f32(), col_major_mat);
    }

    #[test]
    fn bed_vm_get_cols() {
        let bed_vm = make_test_bed_vm();
        let cols = vec![0, 5];
        let exp = [
            [
                0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1.0, 0.0, 2.0, 0.0,
            ],
            [
                0.0, 2.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 1.0,
                2.0, 1.0, 0.0, 1.0,
            ],
        ];
        assert_eq!(bed_vm.get_cols(&cols), exp);
    }

    #[test]
    fn bed_vm_col_means() {
        let bed_vm = make_test_bed_vm();
        let exp = [0.35, 0.5, 0.05, 0.35, 0., 0.9, 0.45, 1., 0.25, 0.7, 0.65];
        assert_eq!(bed_vm.col_means, exp);
    }

    #[test]
    fn bed_vm_col_stds() {
        let bed_vm = make_test_bed_vm();
        let exp = [
            0.5722761, 0.591608, 0.21794495, 0.47696957, 0.0, 0.70000005, 0.58949125, 0.5477226,
            0.622495, 0.55677646, 0.5722762,
        ];
        assert_eq!(bed_vm.col_stds, exp);
    }

    #[test]
    fn bed_get_submatrix_af_standardized() {
        let bed_vm = make_test_bed_vm();
        let subm = bed_vm.get_submatrix_af_standardized(&[0, 5]);
        let exp = [
            -0.6115929, -0.6115929, 1.1358153, -0.6115929, 1.1358153, -0.6115929, -0.6115929,
            1.1358153, -0.6115929, -0.6115929, 1.1358153, -0.6115929, -0.6115929, -0.6115929,
            -0.6115929, -0.6115929, 1.1358153, -0.6115929, 2.8832235, -0.6115929, -1.2857141,
            1.5714285, -1.2857141, 0.14285716, 0.14285716, 0.14285716, 1.5714285, -1.2857141,
            0.14285716, 0.14285716, 0.14285716, 0.14285716, 1.5714285, -1.2857141, -1.2857141,
            0.14285716, 1.5714285, 0.14285716, -1.2857141, 0.14285716,
        ];
        assert_eq!(to_host(&subm), exp);
    }
}
