use std::collections::HashSet;
use std::error::Error;
use std::f32::consts::PI;
use std::fmt::{Display, Formatter};
use std::ops::Range;
use std::sync::Arc;
use rustfft::{Fft, FftPlanner};
use rustfft::num_complex::Complex;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SignalFlag {
    TimeDomain,
    FrequencyDomain,
    AmplitudeCodomain,
    DBFSCodomain
}

#[derive(Debug, Clone)]
pub struct Signal {
    buffer: Vec<f32>,
    sample_rate: u32,
    flags: HashSet<SignalFlag>
}

impl Signal {
    const DEFAULT_FLAGS: [SignalFlag; 2] = [SignalFlag::AmplitudeCodomain, SignalFlag::TimeDomain];

    pub fn lerp(from: Signal, to: Signal, time: f32) -> Signal {
        let max = from.len().max(to.len());
        let mut from_data = from.samples();
        let mut to_data = to.samples();

        from_data.resize(max, 0.0);
        to_data.resize(max, 0.0);

        let mut data = Vec::with_capacity(max);
        for i in 0..max {
            let a = if from_data[i].is_finite() { from_data[i] } else { 0.0 };
            let b = if to_data[i].is_finite() { to_data[i] } else { 0.0 };
            data.push(a + (b - a) * time);
        }
        return Signal::with(to, data);
    }

    pub fn with(source: Signal, data: Vec<f32>) -> Signal {
        Signal { buffer: data, sample_rate: source.sample_rate, flags: source.flags }
    }

    pub fn new(sample_rate: u32) -> Self {
        Signal { buffer: Vec::new(), sample_rate, flags: HashSet::from(Signal::DEFAULT_FLAGS) }
    }

    pub fn resample(source: &Signal, sample_rate: u32) -> Signal {
        if source.sample_rate == sample_rate {
            return source.clone();
        }

        return Signal::new(sample_rate);
    }

    pub fn push(&mut self, samples: &[f32]) {
        for sample in samples {
            self.buffer.insert(0, *sample)
        }
    }

    pub fn truncate(&mut self, length: usize) {
        self.buffer.truncate(length);
    }

    pub fn len(&self) -> usize {
        return self.buffer.len();
    }

    pub fn samples(&self) -> Vec<f32> {
        return self.buffer.clone();
    }

    pub fn has_flag(self, flag: SignalFlag) -> bool {
        return self.flags.contains(&flag);
    }

    pub fn deflag(mut self, flag: SignalFlag) -> Self {
        self.flags.remove(&flag);
        self
    }

    pub fn flag(mut self, flag: SignalFlag) -> Self {
        self.flags.insert(flag);
        self
    }

    pub fn process_all(self, steps: Vec<Box<dyn ProcessingStep>>) -> Result<Self, ProcessingError> {
        let mut current = self.clone();
        for step in steps {
            match current.process(&step) {
                Ok(signal) => {
                    //println!("{:?} {:?}", &step.friendly_name(), signal);
                    current = signal;
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }
        return Ok(current);
    }

    pub fn process(self, step: &Box<dyn ProcessingStep>) -> Result<Self, ProcessingError> {
        let clone = self.clone();
        if !step.is_valid_input(clone) {
            return Err(ProcessingError {});
        }

        Ok(step.process(self))
    }
}


impl IntoIterator for Signal {
    type Item = f32;
    type IntoIter = std::vec::IntoIter<f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.samples().into_iter()
    }
}


pub trait ProcessingStep {
    fn friendly_name(&self) -> String {
        "Unknown Processing Step".to_string()
    }

    fn is_valid_input(&self, _signal: Signal) -> bool {
        true
    }

    fn process(&self, signal: Signal) -> Signal;
}

pub trait Window { }
/// Applies a Hann window to the input signal.
pub struct HannWindow;
impl HannWindow {
    pub fn new() -> Box<Self> {
        box HannWindow
    }
}

impl ProcessingStep for HannWindow {
    fn friendly_name(&self) -> String {
        "Hann Window".to_string()
    }
    fn is_valid_input(&self, signal: Signal) -> bool {
        signal.has_flag(SignalFlag::TimeDomain)
    }
    fn process(&self, signal: Signal) -> Signal {
        let samples = signal.samples();
        let mut windowed_samples = Vec::with_capacity(samples.len());
        let samples_len_f32 = samples.len() as f32;
        for (i, sample) in samples.iter().enumerate() {
            let theta = 2.0 * PI * i as f32;
            let term = (theta / samples_len_f32).cos();
            let total = 0.5 * (1.0 - term);
            windowed_samples.push(total * sample)
        }
        Signal::with(signal, windowed_samples)
    }
}

/// Demangles an FFT output signal by taking only the last half of it, i.e. the positive frequencies.
pub struct Demangle;
impl Demangle {
    pub fn new() -> Box<Self> {
        box Demangle
    }
}

impl ProcessingStep for Demangle {
    fn friendly_name(&self) -> String {
        "Demangle".to_string()
    }
    fn is_valid_input(&self, signal: Signal) -> bool {
        signal.has_flag(SignalFlag::FrequencyDomain)
    }

    fn process(&self, signal: Signal) -> Signal {
        let buffer = signal.samples();
        let len = buffer.len();
        let (_dc, rest) = buffer.as_slice().split_at(1);
        let (positive, _negative) = rest.split_at(len / 2);
        Signal::with(signal, positive.to_vec())
    }
}

pub struct Pad {
    target_chunk_size: usize
}

impl Pad {
    pub fn to_size(target_chunk_size: usize) -> Box<Self> {
        box Pad { target_chunk_size }
    }
}

impl ProcessingStep for Pad {
    fn friendly_name(&self) -> String {
        "Zero-pad".to_string()
    }
    fn process(&self, signal: Signal) -> Signal {
        let mut data = signal.samples();
        let len = signal.len();
        let chunk_len = self.target_chunk_size;

        let remainder = (len + chunk_len) % chunk_len;
        if remainder == 0 {
            return signal;
        }
        data.resize(len + chunk_len - remainder, 0.0);
        Signal::with(signal, data)
    }
}

pub struct FFT {
    backing_fft: Box<Arc<dyn Fft<f32>>>
}

impl FFT {
    pub fn with_buffer_size(buffer_size: usize) -> Box<Self> {
        box FFT { backing_fft: Box::new(
            FftPlanner::new().plan_fft_forward(buffer_size)) }
    }
}

impl ProcessingStep for FFT {
    fn friendly_name(&self) -> String {
        "FFT".to_string()
    }

    fn is_valid_input(&self, signal: Signal) -> bool {
        if !signal.clone().has_flag(SignalFlag::TimeDomain) {
            return false;
        }

        let signal_length = signal.len();
        let fft_size = self.backing_fft.len();
        return signal_length % fft_size <= 0 && signal_length >= fft_size;
    }

    fn process(&self, signal: Signal) -> Signal {
        let buffer: &mut [Complex<f32>] = &mut *signal.clone().into_iter()
            .map(|x| Complex::new(x, 0.0)).collect::<Vec<Complex<f32>>>();

        self.backing_fft.process(buffer);

        let amplitudes = buffer.to_vec().into_iter().map(|complex| {
            complex.norm() / buffer.len() as f32 * 2.0
        }).collect();

        Signal::with(signal, amplitudes)
            .deflag(SignalFlag::TimeDomain)
            .flag(SignalFlag::FrequencyDomain)
    }
}

pub struct ToDBFS;
impl ToDBFS {
    pub fn new() -> Box<Self> {
        box ToDBFS
    }
}
impl ProcessingStep for ToDBFS {
    fn friendly_name(&self) -> String {
        "To DBFS".to_string()
    }
    fn is_valid_input(&self, signal: Signal) -> bool {
        signal.has_flag(SignalFlag::AmplitudeCodomain)
    }

    fn process(&self, signal: Signal) -> Signal {
        Signal::with(signal.clone(), signal.clone().samples().into_iter().map(|x| 20.0 * x.log10()).collect())
            .deflag(SignalFlag::AmplitudeCodomain).flag(SignalFlag::DBFSCodomain)
    }
}

pub struct Subsample {
    pub factor: usize
}

impl Subsample {
    pub fn with_factor(factor: usize) -> Box<Self> {
        box Subsample { factor }
    }
}

impl ProcessingStep for Subsample {
    fn friendly_name(&self) -> String {
        "Subsample".to_string()
    }

    fn process(&self, signal: Signal) -> Signal {
        Signal::with(signal.clone(), signal.clone().samples().into_iter().step_by(self.factor).collect())
    }
}

pub struct Supersample {
    pub factor: usize,
    pub interpolation: Box<dyn Fn(f32, f32, f32) -> f32>
}

impl Supersample {
    pub fn with_cosine_interpolation(factor: usize) -> Box<Self> {
        box Supersample {
            factor, interpolation: Box::new(|from, to, time| {
                let other_time = (1.0 - (time * PI).cos()) * 0.5;
                from * (1.0 - other_time) + to * other_time
            })
        }
    }
}

impl ProcessingStep for Supersample {
    fn friendly_name(&self) -> String {
        "Cosine Supersample".to_string()
    }

    fn process(&self, signal: Signal) -> Signal {
        let data = signal.clone().samples();
        let mut output = Vec::with_capacity(data.len() * self.factor);
        for i in 1..data.len() {
            let from = data[i - 1];
            let to = data[i];
            output.push(from);
            for j in 1..self.factor {
                output.push((self.interpolation)(from, to, j as f32 / self.factor as f32));
            }
        }
        output.push(data[data.len() - 1]);
        return Signal::with(signal.clone(), output);
    }
}

pub struct LimitFrequencyRange {
    range: Range<f32>
}

impl LimitFrequencyRange {
    pub fn to(range: Range<f32>) -> Box<Self> {
        box LimitFrequencyRange { range }
    }

    fn bin(sample_rate: f32, length: f32, frequency: f32) -> usize {
        (frequency as f32 / (sample_rate / length)).floor() as usize
    }
}

impl ProcessingStep for LimitFrequencyRange {
    fn friendly_name(&self) -> String {
        "Limit Frequency Range".to_string()
    }

    fn is_valid_input(&self, _signal: Signal) -> bool {
        _signal.has_flag(SignalFlag::FrequencyDomain)
    }

    fn process(&self, signal: Signal) -> Signal {
        let sample_rate = signal.sample_rate as f32;
        let length = signal.len() as f32;
        let data = signal.clone().samples();

        let min = LimitFrequencyRange::bin(sample_rate, length, self.range.start);
        let max = LimitFrequencyRange::bin(sample_rate, length, self.range.end);
        Signal::with(signal, data[min..max].to_vec())
    }
}

pub struct Squish {
    factor: f32
}

impl Squish {
    pub fn by(factor: f32) -> Box<Self> {
        box Squish { factor }
    }
}

impl ProcessingStep for Squish {
    fn friendly_name(&self) -> String {
        "Squish".to_string()
    }

    fn is_valid_input(&self, _signal: Signal) -> bool {
        _signal.has_flag(SignalFlag::FrequencyDomain)
    }

    fn process(&self, signal: Signal) -> Signal {
        let data = signal.clone().samples();
        let mut new_data = Vec::with_capacity(data.len());
        let mut index = 0f32;
        while (index.floor() as usize) < data.len() {
            new_data.push(data[index.floor() as usize]);
            index += 1.0 + index * self.factor;
        }

        Signal::with(signal, new_data)
    }
}

#[derive(Debug)]
pub struct ProcessingError {

}

impl Display for ProcessingError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "an error occurred while processing")
    }
}

impl Error for ProcessingError {

}
