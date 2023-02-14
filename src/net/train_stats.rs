use super::branch::branch::HMCStepResult;
use crate::data::data::Data;
use crate::data::genotypes::GroupedGenotypes;
use serde::{Deserialize, Serialize};
use serde_json::to_writer;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

pub struct ReportCfg<'data, T: GroupedGenotypes> {
    pub(crate) interval: usize,
    pub(crate) test_data: Option<&'data Data<T>>,
}

impl<'data, T: GroupedGenotypes> ReportCfg<'data, T> {
    pub fn new(interval: usize, test_data: Option<&'data Data<T>>) -> Self {
        Self {
            interval,
            test_data,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct TrainingStats {
    pub(crate) num_samples: usize,
    pub(crate) num_accepted: usize,
    pub(crate) num_early_rejected: usize,
    pub(crate) mse_train: Vec<f32>,
    pub(crate) mse_test: Option<Vec<f32>>,
    pub(crate) lpd: Vec<f32>,
}

impl TrainingStats {
    pub(crate) fn new() -> Self {
        Self {
            num_samples: 0,
            num_accepted: 0,
            num_early_rejected: 0,
            mse_train: Vec::new(),
            mse_test: None,
            lpd: Vec::new(),
        }
    }

    pub(crate) fn add_hmc_step_result(&mut self, res: &HMCStepResult) {
        self.num_samples += 1;
        match res {
            HMCStepResult::Accepted(_) => self.num_accepted += 1,
            HMCStepResult::RejectedEarly => self.num_early_rejected += 1,
            HMCStepResult::Rejected => {}
        }
    }

    pub(crate) fn add_lpd(&mut self, lpd: f32) {
        self.lpd.push(lpd)
    }

    pub(crate) fn add_mse_test(&mut self, mse: f32) {
        if self.mse_test.is_none() {
            self.mse_test = Some(Vec::new());
        }
        self.mse_test.as_mut().unwrap().push(mse);
    }

    pub(crate) fn add_mse_train(&mut self, mse: f32) {
        self.mse_train.push(mse);
    }

    pub(crate) fn acceptance_rate(&self) -> f32 {
        self.num_accepted as f32 / self.num_samples as f32
    }

    pub(crate) fn early_rejection_rate(&self) -> f32 {
        self.num_early_rejected as f32 / self.num_samples as f32
    }

    pub(crate) fn end_rejection_rate(&self) -> f32 {
        (self.num_samples - self.num_early_rejected - self.num_accepted) as f32
            / self.num_samples as f32
    }

    pub(crate) fn to_file(&self, outdir: &String) {
        let outpath = Path::new(outdir).join("training_stats");
        let w = BufWriter::new(File::create(outpath).unwrap());
        to_writer(w, &self).unwrap();
    }
}
