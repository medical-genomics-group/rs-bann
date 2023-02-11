//! Useful elementary array / vec based functions

pub fn sum_of_squares(arr: &[f32]) -> f32 {
    arr.iter().map(|e| e * e).sum()
}

pub fn sum_of_abs(arr: &[f32]) -> f32 {
    arr.iter().map(|e| e.abs()).sum()
}
