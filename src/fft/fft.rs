use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Index, Range};
use std::slice::SliceIndex;
use std::sync::{Mutex, PoisonError};
use realfft::{FftError, RealFftPlanner};
use rustfft::FftNum;
use rustfft::num_complex::Complex;
use crate::numtools::hann;


pub struct AudioView {
    samples: Mutex<Box<[f32]>>,
    sample_rate: u32,
}

pub trait TryIntoFrequencySpectrum<T: FftNum> {
    type Error;

    fn try_into_spectrum(&self, planner: &mut RealFftPlanner<T>) -> Result<FrequencySpectrum, Self::Error>;
}

impl AudioView {
    pub fn new(sample_rate: u32, data: Mutex<Box<[f32]>>) -> Self {
        Self {
            samples: data,
            sample_rate
        }
    }

    pub fn subview(&self, range: Range<usize>) -> AudioView {
        AudioView {
            samples: Mutex::new(self.samples.lock().unwrap()[range].into()),
            sample_rate: self.sample_rate
        }
    }
}

pub enum SpectrumError {
    SampleMutexPoisoned(PoisonError<()>),
    ConversionFailed(FftError)
}

impl Debug for SpectrumError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match self {
            Self::SampleMutexPoisoned(_) => "could not acquire samples mutex because the lock is poisoned",
            Self::ConversionFailed(_) => "an error occurred while performing the fft"
        })
    }
}

impl Display for SpectrumError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match self {
            Self::SampleMutexPoisoned(_) => "a previous error prevents us from listening to audio",
            Self::ConversionFailed(_) => "something went wrong while analysing the audio"
        })
    }
}

impl Error for SpectrumError {
    fn cause(&self) -> Option<&dyn Error> {
        match self {
            SpectrumError::SampleMutexPoisoned(cause) => Some(cause),
            SpectrumError::ConversionFailed(cause) => Some(cause)
        }
    }
}

impl<T> From<PoisonError<T>> for SpectrumError {
    fn from(_: PoisonError<T>) -> Self {
        SpectrumError::SampleMutexPoisoned(PoisonError::new(()))
    }
}

impl From<FftError> for SpectrumError {
    fn from(value: FftError) -> Self {
        SpectrumError::ConversionFailed(value)
    }
}

impl TryIntoFrequencySpectrum<f32> for AudioView {
    type Error = SpectrumError;

    fn try_into_spectrum(&self, planner: &mut RealFftPlanner<f32>) -> Result<FrequencySpectrum, Self::Error> {
        let sample_rate = self.sample_rate;
        let mut input;
        {
            let samples = self.samples.lock()?;
            input = vec![0f32; samples.len()];
            input.clone_from_slice(&*samples);
        }

        let fft_size = input.len();
        hann(&mut input, fft_size);

        let mut output = vec![Complex::new(0.0, 0.0); fft_size / 2 + 1];

        let fft = planner.plan_fft_forward(fft_size);
        fft.process(&mut input, &mut output)?;

        let bins = output.iter().map(|complex| (complex / fft_size as f32).norm()).collect();
        Ok(FrequencySpectrum { bins, sample_rate })
    }
}

pub struct Hertz(pub f32);

#[derive(Clone)]
pub struct FrequencySpectrum {
    bins: Box<[f32]>,
    sample_rate: u32,
}

impl Default for FrequencySpectrum {
    fn default() -> Self {
        Self {
            bins: Box::new([]),
            sample_rate: 1
        }
    }
}

impl FrequencySpectrum {
    pub fn merge<R>(&self, other: &FrequencySpectrum, reconciler: R) -> FrequencySpectrum
    where
        R: Fn(&FrequencySpectrum, (Hertz, f32), &FrequencySpectrum) -> f32
    {
        let merged_bins = self.bins.iter().enumerate().map(|(index, value)| {
            reconciler(self, (self.bin_to_hertz(index), *value), other)
        }).collect();

        FrequencySpectrum {
            sample_rate: self.sample_rate,
            bins: merged_bins
        }
    }

    pub fn get(&self, bin: usize) -> Option<f32> {
        self.bins.get(bin).map(|x| *x)
    }

    pub fn hertz_to_bin(&self, hertz: Hertz) -> f32 {
        let ratio = hertz.0 / self.nyquist_frequency() as f32;
        ratio * self.bins.len() as f32
    }

    pub fn bin_to_hertz(&self, bin: usize) -> Hertz {
        let ratio = bin as f32 / (self.bins.len() as f32);
        Hertz(ratio * self.nyquist_frequency() as f32)
    }

    pub fn nyquist_frequency(&self) -> u32 {
        self.sample_rate / 2
    }
}

impl Index<Hertz> for FrequencySpectrum {
    type Output = f32;

    fn index(&self, index: Hertz) -> &Self::Output {
        &self[self.hertz_to_bin(index) as usize]
    }
}

impl<Idx> Index<Idx> for FrequencySpectrum
where Idx: SliceIndex<[f32]> {
    type Output = Idx::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.bins[index]
    }
}