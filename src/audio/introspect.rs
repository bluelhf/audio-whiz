use std::collections::VecDeque;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;
use rodio::{Sample, Source};
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