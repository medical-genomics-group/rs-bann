use super::dims::BedDims;
use crate::error::Error;
use crate::io::{
    bed_lookup_tables::BED_LOOKUP_GENOTYPE, bim::BimEntry, fam::FamEntry,
    indexed_read::IndexedReader,
};
use arrayfire::{dim4, Array};
use log::warn;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Binomial, Distribution, Uniform};
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};

const BED_SIGNATURE_LENGTH: usize = 3;
const BED_VM_SIGNATURE: [u8; 3] = [0x6c, 0x1b, 0x01];
const BED_SM_SIGNATURE: [u8; 3] = [0x6c, 0x1b, 0x00];
// There is no handing of NA here!
const BED_VALUE_MAPPING: [u8; 3] = [0x03, 0x02, 0x00];

pub trait CommonStemFileset {
    fn stem(&self) -> &PathBuf;

    fn stem_with_extension(&self, ext: &str) -> PathBuf {
        let mut path = self.stem().clone();
        path.set_extension(ext);
        path
    }
}

pub trait BedContainingFileset: CommonStemFileset {
    fn bed(&self) -> PathBuf {
        self.stem_with_extension("bed")
    }
}

/// Paths of a set of .bed, .bim, .fam files
pub struct PlinkBinaryFileset {
    stem: PathBuf,
}

impl CommonStemFileset for PlinkBinaryFileset {
    fn stem(&self) -> &PathBuf {
        &self.stem
    }
}

impl BedContainingFileset for PlinkBinaryFileset {}

impl PlinkBinaryFileset {
    pub fn new(path: &Path) -> Self {
        Self {
            stem: path.to_owned(),
        }
    }

    pub fn bim(&self) -> PathBuf {
        self.stem_with_extension("bim")
    }

    pub fn fam(&self) -> PathBuf {
        self.stem_with_extension("fam")
    }
}

/// Paths of a set of .bed and corresponding .dims file
pub struct BedBinaryFileset {
    stem: PathBuf,
}

impl CommonStemFileset for BedBinaryFileset {
    fn stem(&self) -> &PathBuf {
        &self.stem
    }
}

impl BedContainingFileset for BedBinaryFileset {}

impl BedBinaryFileset {
    pub fn new(path: &Path) -> Self {
        Self {
            stem: path.to_owned(),
        }
    }

    pub fn dims(&self) -> PathBuf {
        self.stem_with_extension("dims")
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
#[derive(Debug, PartialEq)]
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
    pub fn random(
        num_individuals: usize,
        num_markers: usize,
        mafs: Option<Vec<f32>>,
        seed: Option<u64>,
    ) -> Self {
        let mut rng = if let Some(s) = seed {
            ChaCha20Rng::seed_from_u64(s)
        } else {
            ChaCha20Rng::from_entropy()
        };
        let mut res = Self {
            data: Vec::new(),
            col_means: Vec::new(),
            col_stds: Vec::new(),
            num_individuals,
            num_markers,
            num_bytes_per_col: (num_individuals + 3) / 4,
            padding: num_individuals % 4,
        };
        let uniform = Uniform::from(0.01..0.5);
        for mix in 0..num_markers {
            loop {
                let maf = if let Some(v) = &mafs {
                    v[mix]
                } else {
                    uniform.sample(&mut rng)
                };
                let binom = Binomial::new(2, maf as f64).unwrap();
                let mut col_sum: f32 = 0.;
                let mut col_vals = Vec::with_capacity(num_individuals);
                for _ in 0..num_individuals {
                    let val = binom.sample(&mut rng) as f32;
                    col_sum += val;
                    col_vals.push(val);
                }
                let col_mean = col_sum / num_individuals as f32;
                let col_std: f32 = (col_vals
                    .iter()
                    .map(|v| (v - col_mean) * (v - col_mean))
                    .sum::<f32>()
                    / num_individuals as f32)
                    .sqrt();
                if col_std != 0. {
                    res.col_means.push(col_mean);
                    res.col_stds.push(col_std);
                    vecf32_to_bed(&col_vals, &mut res.data);
                    break;
                }
            }
        }
        res
    }

    /// Reads .bed file from disc.
    /// Determines number of markers and individuals from .bim and .fam files with the same filestem as the .bed.
    /// Checks if .bed signature is valid.
    pub fn from_file(stem: &Path) -> Self {
        let bed_dims = BedDims::from_dims_file(stem).unwrap_or_else(|_| {
            BedDims::from_plink_fileset(stem).expect("Failed to load bed dims!")
        });

        let bed_file = PlinkBinaryFileset::new(stem).bed();
        let signature = BedSignature::from_bed_file(&bed_file).expect("Unexpected .bed signature");
        if let BedSignature::SampleMajor = signature {
            panic!("SampleMajor .bed formats are not supported at the moment. Try converting to VariantMajor format.")
        }

        let mut bed_file = std::fs::File::open(bed_file).expect("Failed to open .bed file");
        bed_file
            .seek(std::io::SeekFrom::Start(
                BED_SIGNATURE_LENGTH.try_into().unwrap(),
            ))
            .unwrap();
        let mut data = Vec::new();
        bed_file
            .read_to_end(&mut data)
            .expect("Error while reading .bed file");

        let mut num_bytes_per_col = bed_dims.num_individuals() / 4;
        let padding = bed_dims.num_individuals() % 4;
        if padding != 0 {
            num_bytes_per_col += 1;
        }

        let mut res = Self {
            data,
            col_means: vec![0.0; bed_dims.num_markers()],
            col_stds: vec![0.0; bed_dims.num_markers()],
            num_individuals: bed_dims.num_individuals(),
            num_markers: bed_dims.num_markers(),
            num_bytes_per_col,
            padding,
        };

        for col_ix in 0..bed_dims.num_markers() {
            let vals = &res.get_cols(&[col_ix])[0];
            let mean: f32 = vals.iter().sum::<f32>() / bed_dims.num_individuals() as f32;
            let std: f32 = (vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>()
                / bed_dims.num_individuals() as f32)
                .sqrt();
            res.col_means[col_ix] = mean;
            res.col_stds[col_ix] = std;
            if std == 0.0 {
                warn!("No variation in marker {:?}; This might lead to division by zero if accessing standardized marker data", col_ix);
            }
        }

        res
    }

    /// Write bed data and dims
    pub fn to_file(&self, stem: &Path) {
        let bedfile =
            std::fs::File::create(stem.with_extension("bed")).expect("Unable to create bed file");
        let mut bedwriter = std::io::BufWriter::new(bedfile);
        bedwriter
            .write_all(&BED_VM_SIGNATURE)
            .expect("Unable to write bed signature");
        bedwriter
            .write_all(&self.data)
            .expect("Unable to write bed data");
        let dimsfile =
            std::fs::File::create(stem.with_extension("dims")).expect("Unable to create dims file");
        let mut dimswriter = std::io::BufWriter::new(dimsfile);
        dimswriter
            .write_all(format!("{}\t{}", self.num_individuals, self.num_markers).as_bytes())
            .expect("Unable to write dims data");
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
    fn chunkf32_to_byte() {
        assert_eq!(super::chunkf32_to_byte(&[1., 0., 1., 1.]), 174);
    }

    #[test]
    fn bed_vm_random_dump_and_load() {
        let num_individuals = 100;
        let num_markers = 20;
        let seed = 42;
        let bed_vm = BedVM::random(num_individuals, num_markers, None, Some(seed));
        let base_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let base_path = Path::new(&base_dir);
        let stem = base_path.join("resources/test/random");
        bed_vm.to_file(&stem);
        assert_eq!(bed_vm, BedVM::from_file(&stem));
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
