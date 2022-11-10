use std::io;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("No genotype data found.")]
    MissingX,
    #[error("Failed to open source file")]
    FromFileOpeningError(#[from] io::Error),
    #[error("Failed to deserialize from file")]
    FromFileDeserializeError(#[from] bincode::Error),
}
