use serde::{Deserialize, Serialize};
use serde_json::to_writer_pretty;
use std::{fs::File, path::Path};

#[derive(Serialize, Deserialize)]
pub struct PhenStats {
    mean: f32,
    variance: f32,
    env_variance: f32,
}

impl PhenStats {
    pub fn new(mean: f32, variance: f32, env_variance: f32) -> Self {
        Self {
            mean,
            variance,
            env_variance,
        }
    }

    pub fn to_file(&self, path: &Path) {
        to_writer_pretty(File::create(path).unwrap(), self).unwrap();
    }
}
