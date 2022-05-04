// TODO:
// preprocess, train and predict subcommands
extern crate blas_src;
extern crate openblas_src;

mod network;

fn main() {}

// TODO:
// Unless the groups get to large, I can do everything on col major files
// and col major bedvecs.
// the preprocessing routine only has to split the large .bed into groups
// following some annotation input.
fn preprocess() {
    unimplemented!();
}

// The following lower bounds for memory consumption are expected,
// if only a subset of all samples is loaded at a time
// n    pg  mem[mb]
// 10k  1k  10**4 * 1x10**3 * 0.25 = 10**7 * 0.75 b =  7.5 Mb
// I need fast random reading of this data.
fn train() {
    unimplemented!();
}

fn predict() {
    unimplemented!();
}
