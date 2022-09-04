use super::branch::branch::HMCStepResult;
use super::data::Data;

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

pub(crate) struct TrainingStats {
    pub(crate) num_samples: usize,
    pub(crate) num_accepted: usize,
    pub(crate) num_early_rejected: usize,
}

impl TrainingStats {
    pub(crate) fn new() -> Self {
        Self {
            num_samples: 0,
            num_accepted: 0,
            num_early_rejected: 0,
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
}
