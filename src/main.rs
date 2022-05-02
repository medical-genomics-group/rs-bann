// TODO:
// preprocess, train and predict subcommands
fn main() {}

// TODO:
// The preprocessing step should produce row major .bed files,
// split into marker groups, following some annotations.
// It is easier to pull random sets of individuals from a RM file.
// Col stats should also be computed and stored.
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
