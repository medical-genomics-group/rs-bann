#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("No genotype data found.")]
    MissingX,
}
