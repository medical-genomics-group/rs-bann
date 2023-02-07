use crate::af_helpers::to_host;
use crate::data::genotypes::{Genotypes, GroupedGenotypes};
use arrayfire::{dim4, matmul, Array, MatProp};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Bernoulli, Distribution, Normal};
use serde::Serialize;

pub struct LinearModelBuilder {
    num_branches: usize,
    num_markers_per_branch: Vec<usize>,
    proportion_effective_markers: f32,
    effects: Option<Vec<Vec<f32>>>,
    rng: ChaCha20Rng,
}

impl LinearModelBuilder {
    pub fn new(num_markers_per_branch: &[usize]) -> Self {
        Self {
            num_branches: num_markers_per_branch.len(),
            num_markers_per_branch: num_markers_per_branch.to_vec(),
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
        let m = self.num_markers_per_branch.iter().sum::<usize>() * self.num_branches;
        let inclusion_dist = Bernoulli::new(self.proportion_effective_markers as f64).unwrap();
        let included: Vec<bool> = (0..m)
            .map(|_| inclusion_dist.sample(&mut self.rng))
            .collect();
        let m_incl = included.iter().filter(|b| **b).count();
        let beta_std = (heritability / m_incl as f32).sqrt();
        let beta_dist = Normal::new(0.0, beta_std).unwrap();

        let mut effects = Vec::new();
        let mut effect_ix = 0;
        for bix in 0..self.num_branches {
            let mut b_effects = Vec::new();
            for _mix in 0..self.num_markers_per_branch[bix] {
                if included[effect_ix] {
                    b_effects.push(beta_dist.sample(&mut self.rng))
                } else {
                    b_effects.push(0.0)
                }
                effect_ix += 1;
            }
            effects.push(b_effects);
        }
        self.effects = Some(effects);
        self
    }

    pub fn build(&self) -> LinearModel {
        LinearModel {
            num_branches: self.num_branches,
            num_markers_per_branch: self.num_markers_per_branch.clone(),
            effects: self.effects.as_ref().unwrap().to_vec(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct LinearModel {
    num_branches: usize,
    num_markers_per_branch: Vec<usize>,
    effects: Vec<Vec<f32>>,
}

impl LinearModel {
    pub fn effects(&self) -> &Vec<Vec<f32>> {
        &self.effects
    }

    fn af_branch_effects(&self, branch_ix: usize) -> Array<f32> {
        Array::new(
            &self.effects[branch_ix],
            dim4!(self.num_markers_per_branch[branch_ix].try_into().unwrap()),
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

    /// Sum of squared effects
    pub fn sum_of_squares(&self) -> f32 {
        self.effects
            .iter()
            .map(|v| v.iter().map(|e| e * e).sum::<f32>())
            .sum::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use crate::data::genotypes::{Genotypes, GenotypesBuilder};

    use super::{LinearModel, LinearModelBuilder};

    const SEED: u64 = 42;
    const NB: usize = 1;
    const NMPB: usize = 5;
    const N: usize = 10;

    fn make_test_lm(prop_eff: f32, h2: f32) -> LinearModel {
        LinearModelBuilder::new(&vec![NMPB; NB])
            .with_seed(SEED)
            .with_proportion_effective_markers(prop_eff)
            .with_random_effects(h2)
            .build()
    }

    fn make_test_gt() -> Genotypes {
        let mut gt = GenotypesBuilder::new()
            .with_seed(SEED)
            .with_random_x(vec![NMPB; NB], N, None)
            .build()
            .unwrap();
        gt.standardize();
        gt
    }

    #[test]
    fn predict() {
        let lm = make_test_lm(0.2, 0.6);
        let gt = make_test_gt();
        let exp: Vec<f32> = vec![
            -0.34460923,
            -0.34460923,
            0.029220968,
            0.28810832,
            0.45567015,
            0.029220968,
            -0.34460923,
            0.28810832,
            0.55880433,
            -0.6153052,
        ];
        assert_eq!(exp, lm.predict(&gt));
    }
}
