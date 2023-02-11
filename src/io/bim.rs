use crate::io::{chromosome::Chromosome, indexed_read::IndexedEntry};

#[derive(Debug)]
pub struct BimEntry {
    pub ix: usize,
    pub chromosome: Chromosome,
    pub id: String,
    pub centimorgan: usize,
    pub position: usize,
    pub allele_1: String,
    pub allele_2: String,
}

impl IndexedEntry for BimEntry {
    fn from_str(s: &str, ix: usize) -> Self {
        let fields = s.split_whitespace().collect::<Vec<&str>>();
        Self {
            ix,
            chromosome: fields[0].parse().unwrap(),
            id: fields[1].to_owned(),
            centimorgan: fields[2]
                .parse()
                .expect("Failed to convert 3rd col entry in .bim to int"),
            position: fields[3]
                .parse()
                .expect("Failed to convert 4th col entry in .bim to int"),
            allele_1: fields[4].to_owned(),
            allele_2: fields[5].to_owned(),
        }
    }
}

impl BimEntry {
    pub fn chr(&self) -> &Chromosome {
        &self.chromosome
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn position(&self) -> usize {
        self.position
    }
}
