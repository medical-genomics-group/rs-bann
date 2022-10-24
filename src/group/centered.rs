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
        while let Ok(_) = reader.read_line(&mut buffer) {
            println!("{}", ix);
            print!("{}", buffer);
            let id = buffer
                .split_whitespace()
                .enumerate()
                .filter(|(ix, _)| *ix == 1)
                .map(|(_, e)| e)
                .collect::<Vec<&str>>()[0];

            res.map.insert(id.to_string(), ix);
            ix += 1;
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

        // let file = File::open(ld_file).unwrap();
        // let mut reader = BufReader::new(file);
        // let mut buffer = String::new();

        // let mut lix = 0;

        // while let Ok(_) = reader.read_line(&mut buffer) {
        //     // skip header line
        //     if lix > 0 {
        //         let fields = buffer.split_whitespace().collect::<Vec<&str>>();
        //         // TODO: there should be a proper error message here
        //         let ix1 = id2ix.ix(fields[2]).unwrap();
        //         let ix2 = id2ix.ix(fields[5]).unwrap();
        //         res.g.entry(*ix1).or_insert(HashSet::new()).insert(*ix2);
        //         res.g.entry(*ix2).or_insert(HashSet::new()).insert(*ix1);
        //     }
        //     lix += 1;
        // }

        // // insert isolated nodes
        // for ix in id2ix.map.values() {
        //     if !res.g.contains_key(ix) {
        //         res.g.insert(*ix, HashSet::new());
        //     }
        // }

        res
    }

    pub fn centered_grouping(&self) -> CenteredGrouping {
        let mut grouping = CenteredGrouping::new();

        let mut degrees: Vec<(usize, usize)> = self.g.iter().map(|(k, v)| (*k, v.len())).collect();
        // descending order by degree
        degrees.sort_by(|a, b| b.1.cmp(&a.1));

        let mut no_centers = HashSet::<usize>::new();
        let mut gix = 0;
        for (cix, _) in degrees {
            if !no_centers.contains(&cix) {
                let neighbors = self.g.get(&cix).unwrap();
                if !neighbors.is_empty() {
                    let mut group = neighbors.iter().map(|e| *e).collect::<Vec<usize>>();
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
                        } else if let Some(n) = grouping.groups.get_mut(&(cix + d)) {
                            n.push(cix);
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

        println!("{:?}", ld_path.as_os_str());
        println!("{:?}", bim_path.as_os_str());

        let g = CorrGraph::from_plink_ld(&ld_path, &bim_path);
        println!("Made a graph!");
        // let grouping = g.centered_grouping();

        // assert_eq!(*grouping.groups.get(&0).unwrap(), vec![2, 3, 4, 1]);
        // assert_eq!(*grouping.groups.get(&1).unwrap(), vec![4, 6, 7, 5]);
        // assert_eq!(*grouping.groups.get(&2).unwrap(), vec![9, 10, 11, 8]);
    }
}
