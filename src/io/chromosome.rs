use crate::error::Error;
use serde::Serialize;
use std::str::FromStr;

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
