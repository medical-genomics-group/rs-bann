use std::io;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("No genotype data found.")]
    MissingX,
    #[error("Failed to open source file")]
    FromFileOpeningError(#[from] io::Error),
    #[error("Failed to deserialize from file")]
    FromFileDeserializeError(#[from] bincode::Error),
    #[error("Line starts with '#'")]
    GFFCommentLine,
    #[error("Unknown feature found in gff")]
    GFFUnknownGenomicFeature,
    #[error("Unknown chromosome")]
    UnknownChromosome,
    #[error("Unknown sex code")]
    FamUnknownSexCode,
    #[error("False first byte in .bed; expected 0x6c")]
    BedFalseFirstByte,
    #[error("False second byte in .bed; expected 0x1b")]
    BedFalseSecondByte,
    #[error("False third byte in .bed; expected 0x00 or 0x01")]
    BedFalseThirdByte,
}
