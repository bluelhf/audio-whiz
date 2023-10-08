use std::f32::consts::PI;
use std::num::FpCategory;
use std::ops::{Add, Mul, Sub};

pub fn to_dbfs(sample: f32) -> Option<f32> {
    let amplitude = 20.0 * sample.log10();
    match amplitude.classify() {
        FpCategory::Zero | FpCategory::Normal => Some(amplitude),
        _ => None
    }
}

pub fn hann(data: &mut Vec<f32>, size: usize) {
    for (i, value) in data.iter_mut().enumerate() {
        *value *= 0.5 - 0.5 * ((2.0 * PI * i as f32) / (size as f32 - 1.0)).cos();
    }
}

pub fn lerp_index_fn<D, T, U, V>(data: D, index: f32, wrap: T) -> T
where
    D: Fn(usize) -> Option<T>,
    T: Sub<Output=U> + Add<V, Output=T> + Copy,
    U: Mul<f32, Output=V> + Add<T>,
    V: Add<T>
{
    let min = data(index.floor() as usize).unwrap_or(wrap);
    return min + (data((index + 1.0).floor() as usize).unwrap_or(wrap) - min) * index.fract();
}

pub fn lerp<A, B, C, D, E>(a: A, b: B, t: f32) -> E
where
    A: Add<D, Output=E> + Copy,
    B: Sub<A, Output=C>,
    C: Mul<f32, Output=D>
{
    a + (b - a) * t
}