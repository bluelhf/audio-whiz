#![feature(trait_alias)]
#![feature(try_blocks)]
#![feature(let_chains)]

mod audio;
mod numtools;
mod fft;

use std::any::Any;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::Duration;
use imageproc::drawing::{draw_antialiased_line_segment_mut, draw_text_mut};
use image::{Rgba, RgbaImage};
use imageproc::pixelops::interpolate;
use itertools::{EitherOrBoth, Itertools};
use nannou::lyon::geom::euclid::approxeq::ApproxEq;
use nannou::prelude::*;
use nannou::wgpu::{Device, Texture};
use nannou::winit::event::VirtualKeyCode;
use once_cell::sync;
use realfft::RealFftPlanner;
use rodio::{cpal, Decoder, DeviceTrait, OutputStream, Sink, Source};
use rodio::cpal::traits::HostTrait;
use rusttype::{Font, Scale};
use crate::audio::introspect::{introspect, introspect_device, Introspectable};
use crate::fft::fft::{FrequencySpectrum, Hertz, TryIntoFrequencySpectrum};
use crate::numtools::{lerp, lerp_index_fn, to_dbfs};

fn main() {
    nannou::app(model)
        .update(update)
        .loop_mode(LoopMode::RefreshSync)
        .run();
}
struct Model {
    introspect: Introspectable<f32>,
    planner: RealFftPlanner<f32>,
    spectra: Vec<FrequencySpectrum>,
    visualiser_texture: Texture,

    is_input: bool,
    device: Box<rodio::Device>,
    _stream: Option<Box<dyn Any>>,
    sink: Option<Sink>,
}

static FONT_BYTES: &[u8] = include_bytes!("liberation.ttf");
static FONT: sync::Lazy<Font<'_>> = sync::Lazy::new(|| Font::try_from_bytes(FONT_BYTES).unwrap());

fn model(app: &App) -> Model {
    app.new_window()
        .resized(on_resize)
        .view(view)
        .key_pressed(on_key_pressed)
        .dropped_file(on_dropped_file)
        .build()
        .unwrap();

    let window = app.main_window();
    let win = window.rect();
    let texture = build_texture(app.main_window().device(), win.wh());

    //<editor-fold desc="find output device" defaultstate="collapsed">
    let device = cpal::default_host().default_output_device().expect("default alsa device not found");
    //</editor-fold>

    let (_stream, stream_handle) = OutputStream::try_from_device(&device).unwrap();
    let sink = Sink::try_new(&stream_handle).unwrap();

    Model {
        _stream: Some(Box::new(_stream)), sink: Some(sink),
        introspect: Introspectable::default(),
        planner: RealFftPlanner::new(),
        spectra: Vec::new(),
        visualiser_texture: texture.into(),
        device: Box::new(device),
        is_input: false,
    }
}

fn on_key_pressed(_app: &App, model: &mut Model, key: Key) {
    match key {
        VirtualKeyCode::P => {
            if let Some(sink) = &model.sink {
                if sink.is_paused() {
                    sink.play();
                } else {
                    sink.pause();
                }
            }
        },
        VirtualKeyCode::I => {
            model.is_input ^= true;
            init_device(model);
        }
        VirtualKeyCode::C => {
            let current_device_name = model.device.name().ok();
            let device_names = cpal::default_host().devices().iter_mut()
                .flatten().filter_map(|d| d.name().ok()).collect::<Vec<_>>();

            device_names.iter().cloned().skip_while(|name| {
                current_device_name.clone().map_or(true, |other| *name != other)
            }).skip(1).chain(device_names.iter().cloned())
            .filter_map(|name| cpal::default_host().devices().iter_mut().flatten().find(|d| d.name().map_or(false, |other| *name == other)))
            .find_map(|device| {
                model.device = Box::new(device);
                init_device(model)
            });

            eprintln!("{:?}", model.device.name());
        }
        _ => {}
    }
}

fn init_device(model: &mut Model) -> Option<()> {
    let device = &model.device;

    let name = device.name().unwrap_or("?".to_string());
    if name.contains("PCH") {
        eprintln!("DANGEROUS DEVICE !!! CPAL WILL CRASH !!! skipping...");
        return None;
    }

    if model.is_input {
        let (introspect, introspected_stream) = introspect_device(
            model.device.as_ref(), Duration::from_secs(1)).ok()?;
        model.introspect = introspect;
        model.sink = None;

        model._stream = Some(Box::new(introspected_stream));


        Some(())
    } else if !model.is_input {
        match OutputStream::try_from_device(device) {
            Ok((stream, stream_handle)) => {
                model.sink.as_ref().map(|sink| sink.stop());
                model.sink = Some(Sink::try_new(&stream_handle).ok()?);
                model._stream = Some(Box::new(stream));
                Some(())
            }
            Err(_) => None
        }
    } else {
        None
    }
}

fn on_dropped_file(_app: &App, model: &mut Model, file: PathBuf) {
    let _: Option<()> = try {
        if model.is_input {
            model.is_input = false;
            init_device(model);
        }

        if let Some(sink) = &model.sink {
            let file = BufReader::new(File::open(file).ok()?);
            let source = Decoder::new(file).ok()?.convert_samples::<f32>();

            let (introspect, introspected) = introspect(source, Duration::from_millis(1000));

            if !sink.empty() {
                sink.clear();
            }

            model.introspect = introspect;
            sink.append(introspected);
            sink.play();
        }
    };
}

fn on_resize(app: &App, model: &mut Model, win: Vec2) {
    model.visualiser_texture = build_texture(app.main_window().device(), win);
}

fn build_texture(device: &Device, size: Vec2) -> Texture {
    wgpu::TextureBuilder::new()
        .size([size[0] as u32, size[1] as u32])
        .format(wgpu::TextureFormat::Rgba8Unorm)
        .usage(wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING)
        .build(device)
}

fn update(_app: &App, model: &mut Model, update: Update) {
    let delta_time = update.since_last.as_secs_f32();

    // Smoothing over time, specified in percentage change in (0.01s)
    // For example, 0.2 here means that the spectrum moves 20 % closer to new values every 0.01 seconds
    let lerp_per_cs = 0.2f32;

    // FFT size for the low frequency resolution, high time resolution transform
    const LOW_RES_FFT_SIZE: usize = 8192;

    model.spectra = model.spectra.iter().cloned().zip_longest(
        model.introspect.audio_views().iter().map(|view| {
            let lo_res_spectrum = view.subview(0..LOW_RES_FFT_SIZE).try_into_spectrum(&mut model.planner).unwrap();
            let hi_res_spectrum = view.try_into_spectrum(&mut model.planner).unwrap();

            hi_res_spectrum.merge(&lo_res_spectrum, |_, (frequency, value), other| {
                let approximate_bin = other.hertz_to_bin(frequency);
                lerp(value, lerp_index_fn(|x| other.get(x), approximate_bin, 0f32), 0.6f32)
            })
        })
    ).map(|eob| {
        match eob {
            EitherOrBoth::Both(old, new) => new.merge(&old, |_, (frequency, value), old| {
                let approximate_bin = old.hertz_to_bin(frequency);
                let old_value = lerp_index_fn(|x| old.get(x), approximate_bin, 0f32);

                lerp(old_value, value, 1.0 - (1.0 - lerp_per_cs).pow(delta_time * 100.0))
            }),
            EitherOrBoth::Left(_) => FrequencySpectrum::default(),
            EitherOrBoth::Right(new) => new
        }
    }).collect();
}

fn view(app: &App, model: &Model, frame: Frame) {
    frame.clear(BLACK);

    let texture_size = model.visualiser_texture.size();
    let mut image = RgbaImage::new(texture_size[0], texture_size[1]);

    const MIN_DBFS: f32 = -120.0;
    const MAX_HEIGHT: f32 = 1.0;

    const MARGIN: f32 = 0.05;

    let margin_y = MARGIN * image.height() as f32;
    let margin_x = margin_y;

    let drawing_area_width = (image.width() as f32 - margin_x * 2.0) as usize;
    let drawing_area_height = (image.height() as f32 - margin_y * 2.0) as usize;

    let dbfs_to_y = |amplitude| {
        margin_y + map_range(map_range(amplitude, MIN_DBFS, 0.0, 0f32, 1f32)
                      .clamp(0.0, 1.0), 0.0, 1.0, drawing_area_height as f32, drawing_area_height as f32 * (1.0 - MAX_HEIGHT))
    };



    for spectrum in &model.spectra {
        let octaves: f32 = (spectrum.nyquist_frequency() as f32 / 10.0).log2();

        let min = margin_x;
        let max = drawing_area_width as f32;
        let step = drawing_area_width as f32 / octaves;

        let mut px = None;
        let mut py = None;

        for x in (margin_x as usize)..(margin_x as usize + drawing_area_width) {
            let zero_aligned_x = x as f32 - min;
            let hertz_target = Hertz(map_range(2.0f32.pow((zero_aligned_x - max) / step), 0f32, 1.0, 0f32, spectrum.nyquist_frequency() as f32));

            let index = spectrum.hertz_to_bin(hertz_target);
            let sample = lerp_index_fn(|index| { spectrum.get(index) }, index, 0.0);

            // skip if NaN, inf, or some other nasty number
            let Some(amplitude) = to_dbfs(sample) else {
                continue;
            };

            let y = dbfs_to_y(amplitude);

            draw_antialiased_line_segment_mut(&mut image,
                                              (px.unwrap_or(x) as i32, py.unwrap_or(y) as i32),
                                              (x as i32, y as i32),
                                              Rgba([u8::MAX; 4]), interpolate);

            px = Some(x);
            py = Some(y);
        }

        let image_height = image.height() as i32;
        for hertz in (-4i32..6).map(|x| 261.626f32 * 2f32.pow(x as f32)).chain([0f32, spectrum.nyquist_frequency() as f32]) {
            let ratio = hertz / (spectrum.nyquist_frequency() as f32);
            let x = if ratio.approx_eq(&0.0) { min } else { ((max + min) * 2.0.ln() + step * ratio.ln()) / (2.0.ln()) };

            draw_text_mut(&mut image, Rgba([u8::MAX; 4]),
                          x as i32, image_height - (margin_y * 0.9) as i32,
                          Scale::uniform(16.0), &FONT,
                          &*format!("{hertz:.0}"));
        }
    }

    const DBFS_TICK_COUNT: usize = 8;
    for amplitude in (0..=DBFS_TICK_COUNT).map(|index| index as f32 / DBFS_TICK_COUNT as f32 * MIN_DBFS) {
        let y = dbfs_to_y(amplitude);
        draw_text_mut(&mut image, Rgba([u8::MAX; 4]),
                      margin_x as i32 / 10, y as i32,
                      Scale::uniform(16.0), &FONT,
                      &*format!("{amplitude:.0}"));
    }

    let flat_samples = image.as_flat_samples();
    model.visualiser_texture.upload_data(
        app.main_window().device(),
        &mut *frame.command_encoder(),
        &flat_samples.as_slice(),
    );

    let draw = app.draw();
    draw.texture(&model.visualiser_texture);
    draw.to_frame(app, &frame).unwrap();
}