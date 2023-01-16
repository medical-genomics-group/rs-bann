use crate::data::Genotypes;
use crate::to_host;
use arrayfire::{dim4, matmul, Array, MatProp};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Bernoulli, Distribution, Normal};
use serde::Serialize;

pub struct LinearModelBuilder {
    num_branches: usize,
    num_markers_per_branch: usize,
    proportion_effective_markers: f32,
    effects: Option<Vec<Vec<f32>>>,
    rng: ChaCha20Rng,
}

impl LinearModelBuilder {
    pub fn new(num_branches: usize, num_markers_per_branch: usize) -> Self {
        Self {
            num_branches,
            num_markers_per_branch,
            proportion_effective_markers: 1.0,
            effects: None,
            rng: ChaCha20Rng::from_entropy(),
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = ChaCha20Rng::seed_from_u64(seed);
        self
    }

    pub fn with_proportion_effective_markers(
        &mut self,
        proportion_effective_markers: f32,
    ) -> &mut Self {
        self.proportion_effective_markers = proportion_effective_markers;
        self
    }

    pub fn with_random_effects(&mut self, heritability: f32) -> &mut Self {
        let m = self.num_markers_per_branch * self.num_branches;
        let inclusion_dist = Bernoulli::new(self.proportion_effective_markers as f64).unwrap();
        let included: Vec<bool> = (0..m)
            .map(|_| inclusion_dist.sample(&mut self.rng))
            .collect();
        let m_incl = included.iter().filter(|b| **b).count();
        let beta_std = (heritability / m_incl as f32).sqrt();
        let beta_dist = Normal::new(0.0, beta_std).unwrap();

        let mut effects = Vec::new();
        for _ in 0..self.num_branches {
            effects.push(
                (0..self.num_markers_per_branch)
                    .map(|ix| {
                        if included[ix] {
                            beta_dist.sample(&mut self.rng)
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<f32>>(),
            );
        }
        self.effects = Some(effects);
        self
    }

    pub fn build(&self) -> LinearModel {
        LinearModel {
            num_branches: self.num_branches,
            num_markers_per_branch: self.num_markers_per_branch,
            effects: self.effects.as_ref().unwrap().to_vec(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct LinearModel {
    num_branches: usize,
    num_markers_per_branch: usize,
    effects: Vec<Vec<f32>>,
}

impl LinearModel {
    fn af_branch_effects(&self, branch_ix: usize) -> Array<f32> {
        Array::new(
            &self.effects[branch_ix],
            dim4!(self.num_markers_per_branch.try_into().unwrap()),
        )
    }

    pub fn predict(&self, gen: &Genotypes) -> Vec<f32> {
        // I expect X to be column major
        let mut y_hat = Array::new(
            &vec![0.0; gen.num_individuals()],
            dim4![gen.num_individuals() as u64, 1, 1, 1],
        );
        // add all branch predictions
        for branch_ix in 0..self.num_branches {
            let af_x = gen.af_branch_data(branch_ix);
            let af_beta = self.af_branch_effects(branch_ix);
            y_hat += matmul(&af_x, &af_beta, MatProp::NONE, MatProp::NONE);
        }
        to_host(&y_hat)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn make_test_lm() {}

    #[test]
    fn test_forward_feed() {}
}
