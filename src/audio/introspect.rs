use std::collections::VecDeque;
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;
use rodio::{Device, DeviceTrait, Sample, Source};
use rodio::cpal::{BuildStreamError, DefaultStreamConfigError, PlayStreamError, StreamConfig};
use rodio::cpal::traits::StreamTrait;
use crate::fft::fft::AudioView;

type BufferAccess<T> = Arc<Vec<RwLock<VecDeque<T>>>>;

pub fn introspect<I>(source: I, buffer_duration: Duration) -> (Introspectable<I::Item>, IntrospectedSource<I>)
    where
        I: Source + Send + 'static,
        I::Item: Sample + Default
{
    let sample_rate = source.sample_rate();
    let num_channels = source.channels() as usize;

    let target_buffer_size = sample_rate as f32 * buffer_duration.as_secs_f32();
    let buffer_size = target_buffer_size.ceil() as usize;

    let buffers: BufferAccess<I::Item> = Arc::new(
        (0..num_channels)
            .map(|_| RwLock::new(VecDeque::with_capacity(buffer_size)))
            .collect::<Vec<_>>());

    let readable = buffers.clone();
    let writable = buffers.clone();

    (Introspectable::new(readable, sample_rate, buffer_size), IntrospectedSource::new(writable, buffer_size, source))
}

pub fn introspect_device(device: &Device, buffer_duration: Duration) -> Result<(Introspectable<f32>, IntrospectedStream), IntrospectError> {
    let input_config = device.default_input_config().map_err(|e| IntrospectError::DefaultConfig(e))?.config();

    let sample_rate = input_config.sample_rate.0;
    let num_channels = input_config.channels as usize;

    let target_buffer_size = sample_rate as f32 * buffer_duration.as_secs_f32();
    let buffer_size = target_buffer_size.ceil() as usize;

    let buffers: BufferAccess<f32> = Arc::new(
        (0..num_channels)
            .map(|_| RwLock::new(VecDeque::with_capacity(buffer_size)))
            .collect::<Vec<_>>());

    let readable = buffers.clone();
    let writable = buffers.clone();

    Ok((
        Introspectable::new(readable, sample_rate, buffer_size),
        IntrospectedStream::new(writable, buffer_size, device, input_config)?
    ))
}

pub struct IntrospectedStream {
    buffer_size: usize,
    current_channel: usize,
    stream: Option<Box<dyn StreamTrait>>
}

impl IntrospectedStream {
    pub fn new(access: BufferAccess<f32>, buffer_size: usize, device: &Device, config: StreamConfig)
        -> Result<IntrospectedStream, IntrospectError> {
        let mut introspected = Self { buffer_size, current_channel: 0, stream: None };

        let stream = device.build_input_stream(&config, move |data: &[f32], _info| {
            for value in data {
                // conscious unwrap: propagate RwLock poisoning
                let mut writer = access[introspected.current_channel].write().unwrap();
                writer.push_front(value.clone());
                if writer.len() > introspected.buffer_size {
                    writer.pop_back();
                }
                introspected.current_channel = (introspected.current_channel + 1) % access.len();
            }
        }, |error| {
            eprintln!("{error}")
        }, None).map_err(|e| IntrospectError::BuildStream(e))?;
        stream.play().map_err(|e| IntrospectError::PlayStream(e))?;

        introspected.stream = Some(Box::new(stream));
        Ok(introspected)
    }
}

#[derive(Debug)]
pub enum IntrospectError {
    DefaultConfig(DefaultStreamConfigError),
    BuildStream(BuildStreamError),
    PlayStream(PlayStreamError)
}

impl Display for IntrospectError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BuildStream(err) => Display::fmt(err, f),
            Self::PlayStream(err) => Display::fmt(err, f),
            Self::DefaultConfig(err) => Display::fmt(err, f)
        }
    }
}

impl Error for IntrospectError {
    fn cause(&self) -> Option<&dyn Error> {
        Some(match self {
            Self::BuildStream(err) => err,
            Self::PlayStream(err) => err,
            Self::DefaultConfig(err) => err
        })
    }
}

#[derive(Debug)]
pub struct Introspectable<T> {
    access: BufferAccess<T>,
    sample_rate: u32,
    buffer_size: usize
}

impl<T> Default for Introspectable<T> {
    fn default() -> Self {
        Self {
            access: Arc::new(Vec::new()),
            sample_rate: 1,
            buffer_size: 1
        }
    }
}

impl<T: Clone + Default> Introspectable<T> {
    pub fn new(access: BufferAccess<T>, sample_rate: u32, buffer_size: usize) -> Self {
        Self { access, sample_rate, buffer_size }
    }

    pub fn channels(&self) -> Vec<Vec<T>> {
        self.access.iter().map(|channel| {
            let read = channel.read().unwrap();
            read.iter().cloned().chain((0..(self.buffer_size - read.len())).map(|_| T::default())).collect::<Vec<_>>()
        }).collect::<Vec<_>>()
    }
}

impl Introspectable<f32> {
    pub fn audio_views(&self) -> Vec<AudioView> {
        self.channels().iter().map(|channel| AudioView::new(self.sample_rate, Mutex::new(channel.clone().into_boxed_slice()))).collect()
    }
}


pub struct IntrospectedSource<I>
    where
        I: Source + Send + 'static,
        I::Item: Sample,
{
    access: BufferAccess<I::Item>,
    buffer_size: usize,
    original: Box<I>,
    current_channel: usize,
}

impl<I: Source<Item=S> + Send + 'static, S: Sample> IntrospectedSource<I> {
    fn new(access: BufferAccess<I::Item>, buffer_size: usize, inner: I) -> Self {
        Self { access, buffer_size, original: Box::new(inner), current_channel: 0 }
    }
}

impl<I: Source<Item=S> + Send + 'static, S: Sample + Clone> Iterator for IntrospectedSource<I> {
    type Item = S;

    fn next(&mut self) -> Option<Self::Item> {
        // conscious unwrap: propagate RwLock poisoning
        let mut writer = self.access[self.current_channel].write().unwrap();

        let value = self.original.next();

        if let Some(value) = value {
            writer.push_front(value.clone());
            if writer.len() > self.buffer_size {
                writer.pop_back();
            }
            self.current_channel = (self.current_channel + 1) % self.access.len();
        }

        value
    }
}

impl<I: Source<Item=S> + Send + 'static, S: Sample + Clone> Source for IntrospectedSource<I> {
    fn current_frame_len(&self) -> Option<usize> {
        self.original.current_frame_len()
    }

    fn channels(&self) -> u16 {
        self.original.channels()
    }

    fn sample_rate(&self) -> u32 {
        self.original.sample_rate()
    }

    fn total_duration(&self) -> Option<Duration> {
        self.original.total_duration()
    }
}