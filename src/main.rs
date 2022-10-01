#![feature(box_syntax)]
mod audio_manager;
mod dsp;

use std::cmp::Ordering::Equal;
use std::path::Path;

use audio_manager::AudioManager;
use nannou::prelude::*;

fn main() {
    nannou::app(model)
        .update(update)
        .simple_window(view)
        .run();
}

struct Model {
    manager: AudioManager
}

fn model(_app: &App) -> Model {
    let manager = AudioManager::new();
    manager.play(audrey::read::open(Path::new("./music/hotel_california.wav"))
            .expect("failed to read audio")).expect("failed to play audio");
    Model { manager }
}

fn update(_app: &App, _model: &mut Model, _update: Update) {
    _model.manager.fourier.lock()
            .expect("failed to lock fourier task").compute();
}

fn view(_app: &App, _model: &Model, frame: Frame) {
    let draw = _app.draw();
    let lock = _model.manager.fourier.lock()
            .expect("failed to lock fourier task");
    
    let samples = lock.output.lock().expect("failed to lock output signal").samples();

    let max = samples.clone().into_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Equal))
            .unwrap_or(0.0);

    let (width, height) = frame.rect().w_h();
    let bar_width = width / samples.len() as f32;
    for i in 0..samples.len() {
        let value = samples[i];
        let bar_height = (value + 100.0) / (max + 100.0) * (height - 10.0) + 10.0;
        if !bar_height.is_finite() {
            continue;
        }
        let rect = Rect::from_w_h(bar_width, bar_height)
                .bottom_left_of(frame.rect())
                .shift_x(i as f32 * bar_width);

        draw.rect().wh(rect.wh()).xy(rect.xy()).color(WHITE).finish();
    }

    frame.clear(BLACK);
    draw.to_frame(_app, &frame).expect("failed to draw to frame");

}
