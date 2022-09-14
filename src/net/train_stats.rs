use super::branch::branch::HMCStepResult;
use super::data::Data;
use serde::Serialize;
use serde_json::to_writer;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

pub struct ReportCfg<'data> {
    pub(crate) interval: usize,
    pub(crate) test_data: Option<&'data Data>,
}

impl<'data> ReportCfg<'data> {
    pub fn new(interval: usize, test_data: Option<&'data Data>) -> Self {
        Self {
            interval,
            test_data,
        }
    }
}

#[derive(Clone, Serialize)]
pub(crate) struct TrainingStats {
    pub(crate) num_samples: usize,
    pub(crate) num_accepted: usize,
    pub(crate) num_early_rejected: usize,
    pub(crate) rss_train: Vec<f64>,
    pub(crate) rss_test: Option<Vec<f64>>,
}

impl TrainingStats {
    pub(crate) fn new() -> Self {
        Self {
            num_samples: 0,
            num_accepted: 0,
            num_early_rejected: 0,
            rss_train: Vec::new(),
            rss_test: None,
        }
    }

    pub(crate) fn add_hmc_step_result(&mut self, res: HMCStepResult) {
        self.num_samples += 1;
        match res {
            HMCStepResult::Accepted => self.num_accepted += 1,
            HMCStepResult::RejectedEarly => self.num_early_rejected += 1,
            HMCStepResult::Rejected => {}
        }
    }

    pub(crate) fn add_rss_test(&mut self, rss: f64) {
        if self.rss_test.is_none() {
            self.rss_test = Some(Vec::new());
        }
        self.rss_test.as_mut().unwrap().push(rss);
    }

    pub(crate) fn add_rss_train(&mut self, rss: f64) {
        self.rss_train.push(rss);
    }

    pub(crate) fn acceptance_rate(&self) -> f64 {
        self.num_accepted as f64 / self.num_samples as f64
    }

    pub(crate) fn early_rejection_rate(&self) -> f64 {
        self.num_early_rejected as f64 / self.num_samples as f64
    }

    pub(crate) fn end_rejection_rate(&self) -> f64 {
        (self.num_samples - self.num_early_rejected - self.num_accepted) as f64
            / self.num_samples as f64
    }

    pub(crate) fn to_file(&self, outdir: &String) {
        let outpath = Path::new(outdir).join("training_stats");
        let w = BufWriter::new(File::create(outpath).unwrap());
        to_writer(w, &self).unwrap();
    }
}
