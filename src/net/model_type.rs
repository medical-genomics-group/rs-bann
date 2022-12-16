use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use strum_macros::EnumString;

#[derive(clap::ValueEnum, Clone, Debug, Serialize, Deserialize, EnumString)]
pub enum ModelType {
    ARD,
    Base,
    StdNormal,
}

impl Display for ModelType {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
        // or, alternatively:
        // fmt::Debug::fmt(self, f)
    }
}
