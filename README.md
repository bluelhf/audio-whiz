# audio-whiz
https://user-images.githubusercontent.com/52505120/193426671-60e6ef9b-9617-4750-8496-f983779a0d31.mp4

A Rust-built music visualiser, built as an exercise to practice both the Rust programming language and homegrown digital signal processing.

| DSP Feature         | Description                                                                                                                         |
|---------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| Pad                 | Zero-pads the FFT input to a desired buffer size (to make it valid input in the first place)                                        |
| Hann window         | Applies the [Hann function](https://en.wikipedia.org/wiki/Hann_function) to an input signal for cleaner FFT results.                |
| FFT                 | Performs a [Fast Fourier Transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform) on sample data to obtain frequency ranges |
| Demangle            | Takes the FFT output in `[DC, +Freq, -Freq]` format and returns only the positive frequency range.                                  |
| Squish              | Squishes the frequency domain by a factor _0 < f < 1_ by skipping _i * f_ elements after each element at index _i_.                 |
| LimitFrequencyRange | Limits the range of frequencies for FFT output to the given range.                                                                  |
| ToDBFS              | Converts FFT output from amplitude ranges to [decibels relative to full scale](https://en.wikipedia.org/wiki/DBFS).                 |
| Subsample           | Subsamples data by a given factor by only taking every `factor`-th element.                                                         |
| Supersample         | Supersamples data using a given interpolation function and supersampling factor.                                                    |
> **Note**  
> Subsampling and supersampling can be used together to smooth the output signal while retaining scale. Additionally, the Supersample step provides
> cosine interpolation by default (via `Supersample::with_cosine_interpolation(...)`)

# Usage
Prepare for some fun and exciting platform-specific setup!

1. **First, follow the [`nannou` setup steps](https://web.archive.org/web/20220520173632/https://guide.nannou.cc/getting_started/platform-specific_setup.html)**[^1]
2. Either
    - Download **and extract** the sources from GitHub (with the `Code > Download ZIP` button at the top right), or
    - Clone the repository with [Git's](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) `git clone https://github.com/bluelhf/audio-whiz` command
3. Navigate to the project directory (folder) n the terminal, and run it with the `cargo run --release` command.
4. You are done! Enjoy the visualisation!
> **Note**  
> Because the project's dependencies are gigantic (about half the average `node_modules` directory), it's recommended to use a
> fast linker like [mold](https://github.com/rui314/mold). If you have mold installed, you can use it by running `mold -run cargo run --release` instead.

## How do I play good music instead?
You can change the song by changing the path in the `model(...)` function in `src/main.rs`.
```diff
 fn model(_app: &App) -> Model {
     let manager = AudioManager::new();
-    manager.play(audrey::read::open(Path::new("./music/second_reality.wav"))
+    manager.play(audrey::read::open(Path::new("./music/good_music.wav"))
             .expect("failed to read audio")).expect("failed to play audio");
     Model { manager }
 }
```
> **Note**  
> Supported audio formats are limited. They are listed in the [`audrey` library's README](https://web.archive.org/web/20220614094422/https://github.com/RustAudio/audrey#supported-formats).[^2]
[^1]: Archived from [the original](https://guide.nannou.cc/getting_started/platform-specific_setup.html) at 2022-05-20T17:36:32Z
[^2]: Archived from [the original](https://github.com/RustAudio/audrey#supported-formats) at 2022-06-14T09:44:22Z
