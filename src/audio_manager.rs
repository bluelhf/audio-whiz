
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::sync::{Arc, Mutex};

use nannou_audio as audio;
use nannou_audio::Buffer;
use crate::dsp::{Squish, Demangle, FFT, HannWindow, LimitFrequencyRange, Pad, Signal, Subsample, Supersample, ToDBFS};

pub struct ForkingRenderer {
    forks: Vec<Box<dyn Fn(&Buffer) + Send>>,
}

impl ForkingRenderer {
    pub fn new() -> Self {
        Self { forks: Vec::new() }
    }

    pub fn add(&mut self, fork: Box<dyn Fn(&Buffer) + Send>) -> &mut Self {
        self.forks.push(fork);
        self
    }

    fn write_to_buffer(audio: &mut Audio, buffer: &mut Buffer) {
        let mut have_ended = vec![];
        let len_frames = buffer.len_frames();

        // Sum all of the sounds onto the buffer.
        for (i, sound) in audio.sounds.iter_mut().enumerate() {
            let mut frame_count = 0;
            let file_frames = sound.frames::<[f32; 2]>().filter_map(Result::ok);
            for (frame, file_frame) in buffer.frames_mut().zip(file_frames) {
                for (sample, file_sample) in frame.iter_mut().zip(&file_frame) {
                    *sample += *file_sample;
                }
                frame_count += 1;
            }

            // If the sound yielded less samples than are in the buffer, it must have ended.
            if frame_count < len_frames {
                have_ended.push(i);
            }
        }

        // Remove all sounds that have ended.
        for i in have_ended.into_iter().rev() {
            audio.sounds.remove(i);
        }
    }

    fn render(&mut self, audio: &mut Audio, buffer: &mut Buffer) {
        ForkingRenderer::write_to_buffer(audio, buffer);
        for fork in &self.forks {
            fork(buffer);
        }
    }
}

pub struct AudioManager {
    stream: audio::Stream<Audio>,
    pub(crate) fourier: Arc<Mutex<FourierTask>>,
}

impl AudioManager {
    pub fn new() -> Self {
        let audio_host = audio::Host::new();
        let fourier = FourierTask::new();

        let instance: Arc<Mutex<FourierTask>> = Arc::new(Mutex::new(fourier));
        let instance_clone = instance.clone();

        let renderer = Arc::new(Mutex::new(ForkingRenderer::new()));

        renderer.clone().lock().expect("failed to lock renderer").add(Box::new(move |buffer| {
            instance.lock().expect("failed to lock fourier").accept(buffer);
        }));

        let sounds = vec![];
        let model = Audio { sounds };

        let stream = audio_host
            .new_output_stream(model)
            .render(move |a: &mut Audio, b: &mut Buffer| {
                renderer.lock().expect("failed to lock renderer").render(a, b);
            }).build().expect("failed to build audio stream");


        stream.play().expect("failed to play audio stream");
        AudioManager { stream, fourier: instance_clone }
    }

    pub fn play(&self, sound: audrey::read::BufFileReader) -> Result<(), AudioManagementError> {
        self.stream.send(move |audio| {
            audio.sounds.push(sound);
        }).map_err(|e| {
            return AudioManagementError::new(Box::new(e));
        })
    }
}


pub struct FourierTask {
    signal: Arc<Mutex<Signal>>,
    pub(crate) output: Arc<Mutex<Signal>>,
}

impl FourierTask {
    fn new() -> FourierTask {
        FourierTask {
            signal: Arc::new(Mutex::new(Signal::new(44100))),
            output: Arc::new(Mutex::new(Signal::new(44100))),
        }
    }

    fn accept(&self, buffer: &Buffer) {
        let mut locked_signal = self.signal.lock().expect("failed to lock signal");
        *locked_signal = Signal::resample(&locked_signal, buffer.sample_rate());

        locked_signal.push(buffer);
        locked_signal.truncate(1024);
    }

    pub fn compute(&self) {
        let locked_signal = self.signal.lock()
            .expect("failed to lock signal").clone();

        match locked_signal.process_all(vec![
            HannWindow::new(),
            Pad::to_size(1024),
            FFT::with_buffer_size(1024),
            Demangle::new(),
            LimitFrequencyRange::to(20.0..18000.0),
            Squish::by(0.9),
            Subsample::with_factor(4),
            Supersample::with_cosine_interpolation(2),
            ToDBFS::new()
        ]) {
            Ok(signal) => {
                let mut output_lock = self.output.lock().expect("failed to obtain output lock");
                *output_lock = Signal::lerp(output_lock.clone(), signal.clone(), 0.13);
            }
            Err(e) => { panic!("{}", e) }
        }
    }
}


#[derive(Debug)]
pub struct AudioManagementError {
    reason: Box<dyn Error>,
}

impl Display for AudioManagementError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.reason, f)
    }
}

impl AudioManagementError {
    pub fn new(error: Box<dyn Error>) -> Self {
        AudioManagementError { reason: error }
    }
}

struct Audio {
    sounds: Vec<audrey::read::BufFileReader>,
}