use super::grouping::MarkerGrouping;
use std::cmp::Reverse;
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

pub struct SNPId2Ix {
    map: HashMap<String, usize>,
}

impl SNPId2Ix {
    pub fn from_bim(file: &Path) -> Self {
        let mut res = SNPId2Ix {
            map: HashMap::new(),
        };

        let file = File::open(file).unwrap();
        let mut reader = BufReader::new(file);
        let mut buffer = String::new();

        let mut ix = 0;
        while let Ok(bytes_read) = reader.read_line(&mut buffer) {
            if bytes_read == 0 {
                break;
            }
            let id = buffer
                .split_whitespace()
                .enumerate()
                .filter(|(ix, _)| *ix == 1)
                .map(|(_, e)| e)
                .collect::<Vec<&str>>()[0];

            res.map.insert(id.to_string(), ix);
            ix += 1;
            buffer.clear();
        }

        res
    }

    pub fn ix(&self, id: &str) -> Option<&usize> {
        self.map.get(id)
    }
}

pub struct CorrGraph {
    g: HashMap<usize, HashSet<usize>>,
}

impl CorrGraph {
    pub fn from_plink_ld(ld_file: &Path, bim_file: &Path) -> Self {
        let id2ix = SNPId2Ix::from_bim(bim_file);
        let mut res = CorrGraph { g: HashMap::new() };

        let file = File::open(ld_file).unwrap();
        let mut reader = BufReader::new(file);
        let mut buffer = String::new();

        let mut lix = 0;

        while let Ok(bytes_read) = reader.read_line(&mut buffer) {
            if bytes_read == 0 {
                break;
            }
            // skip header line
            if lix > 0 {
                let fields = buffer.split_whitespace().collect::<Vec<&str>>();
                // TODO: there should be a proper error message here
                let ix1 = id2ix.ix(fields[2]).unwrap();
                let ix2 = id2ix.ix(fields[5]).unwrap();
                res.g.entry(*ix1).or_insert_with(HashSet::new).insert(*ix2);
                res.g.entry(*ix2).or_insert_with(HashSet::new).insert(*ix1);
            }
            lix += 1;
            buffer.clear();
        }

        // insert isolated nodes
        for ix in id2ix.map.values() {
            if !res.g.contains_key(ix) {
                res.g.insert(*ix, HashSet::new());
            }
        }

        res
    }

    pub fn centered_grouping(&self) -> CenteredGrouping {
        let mut grouping = CenteredGrouping::new();

        let mut degrees: Vec<(usize, usize)> = self.g.iter().map(|(k, v)| (*k, v.len())).collect();
        // descending order by degree
        degrees.sort_by_key(|e| (Reverse(e.1), e.0));

        let mut no_centers = HashSet::<usize>::new();
        let mut gix = 0;
        for (cix, _) in degrees {
            if !no_centers.contains(&cix) {
                let neighbors = self.g.get(&cix).unwrap();
                if !neighbors.is_empty() {
                    let mut group = neighbors.iter().copied().collect::<Vec<usize>>();
                    group.push(cix);
                    // group.sort();
                    no_centers.extend(group.iter());
                    grouping.groups.insert(gix, group);
                    gix += 1;
                } else {
                    // find closest (by id) group
                    for d in 1..100 {
                        if let Some(n) = grouping.groups.get_mut(&(cix - d)) {
                            n.push(cix);
                            break;
                        } else if let Some(n) = grouping.groups.get_mut(&(cix + d)) {
                            n.push(cix);
                            break;
                        }
                    }
                }
            }
        }

        grouping
    }
}

/// Grouping of SNPs.
/// Group centers are selected as uncorrelated SNPs with largest
/// remaining degree. All SNPs correlated with a center SNP are
/// part of the same
pub struct CenteredGrouping {
    pub groups: HashMap<usize, Vec<usize>>,
}

impl CenteredGrouping {
    fn new() -> Self {
        CenteredGrouping {
            groups: HashMap::new(),
        }
    }
}

impl MarkerGrouping for CenteredGrouping {
    fn num_groups(&self) -> usize {
        self.groups.len()
    }

    fn group(&self, ix: usize) -> Option<&Vec<usize>> {
        self.groups.get(&ix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::path::Path;

    #[test]
    fn test_create_centered_grouping() {
        let base_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let base_path = Path::new(&base_dir);
        let ld_path = base_path.join("resources/test/small.ld");
        let bim_path = base_path.join("resources/test/small.bim");

        let g = CorrGraph::from_plink_ld(&ld_path, &bim_path);

        let grouping = g.centered_grouping();

        let exp_groups = vec![vec![0, 1, 2, 3], vec![3, 4, 5], vec![6, 7, 8, 9, 10]];

        for gix in 0..=2 {
            let mut group = grouping.groups.get(&gix).unwrap().clone();
            group.sort_unstable();
            assert_eq!(group, exp_groups[gix]);
        }
    }
}
