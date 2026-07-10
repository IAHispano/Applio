<h1 align="center">
  <a href="https://applio.org" target="_blank"><img src="https://github.com/IAHispano/Applio/assets/133521603/78e975d8-b07f-47ba-ab23-5a31592f322a" alt="Applio"></a>
</h1>

<p align="center">
    <img alt="Contributors" src="https://img.shields.io/github/contributors/iahispano/applio?style=for-the-badge&color=FFFFFF" />
    <img alt="Release" src="https://img.shields.io/github/release/iahispano/applio?style=for-the-badge&color=FFFFFF" />
    <img alt="Stars" src="https://img.shields.io/github/stars/iahispano/applio?style=for-the-badge&color=FFFFFF" />
    <img alt="Fork" src="https://img.shields.io/github/forks/iahispano/applio?style=for-the-badge&color=FFFFFF" />
    <img alt="Issues" src="https://img.shields.io/github/issues/iahispano/applio?style=for-the-badge&color=FFFFFF" />
</p>

<p align="center">A simple, high-quality voice conversion tool, focused on ease of use and performance.</p>

<p align="center">
  <a href="https://applio.org" target="_blank">🌐 Website</a>
  •
  <a href="https://docs.applio.org" target="_blank">📚 Documentation</a>
  •
  <a href="https://discord.gg/wY7gmqTyEV" target="_blank">☎️ Discord</a>
</p>

<p align="center">
  <a href="https://github.com/IAHispano/Applio-Plugins" target="_blank">🛒 Plugins</a>
  •
  <a href="https://huggingface.co/IAHispano/Applio/tree/main/Compiled" target="_blank">📦 Compiled</a>
  •
  <a href="https://applio.org/playground" target="_blank">🎮 Playground</a>
  •
  <a href="https://colab.research.google.com/github/iahispano/applio/blob/main/assets/Applio.ipynb" target="_blank">🔎 Google Colab (UI)</a>
  •
  <a href="https://colab.research.google.com/github/iahispano/applio/blob/main/assets/Applio_NoUI.ipynb" target="_blank">🔎 Google Colab (No UI)</a>
</p>

> [!NOTE]  
> Applio will no longer receive frequent updates. Going forward, development will focus mainly on security patches, dependency updates, and occasional feature improvements. This is because the project is already stable and mature with limited room for further improvements.

## Introduction

Applio is a powerful voice conversion tool focused on simplicity, quality, and performance. Whether you're an artist, developer, or researcher, Applio offers a straightforward platform for high-quality voice transformations. Its flexible design allows for customization through plugins and configurations, catering to a wide range of projects.

## Terms of Use and Commercial Usage

Using Applio responsibly is essential.

- Users must respect copyrights, intellectual property, and privacy rights.
- Applio is intended for lawful and ethical purposes, including personal, academic, and investigative projects.
- Commercial usage is permitted, provided users adhere to legal and ethical guidelines, secure appropriate rights and permissions, and comply with the [MIT license](./LICENSE).

The source code and model weights in this repository are licensed under the permissive [MIT license](./LICENSE), allowing modification, redistribution, and commercial use.

However, if you choose to use this official version of Applio (as provided in this repository, without significant modification), you must also comply with our [Terms of Use](./TERMS_OF_USE.md). These terms apply to our integrations, configurations, and default project behavior, and are intended to ensure responsible and ethical use without limiting their use in any way.

For commercial use, we recommend contacting us at [support@applio.org](mailto:support@applio.org) to ensure your usage aligns with ethical standards. All audio generated with Applio must comply with applicable copyright laws. If you find Applio helpful, consider supporting its development [through a donation](https://ko-fi.com/iahispano).

By using the official version of Applio, you accept full responsibility for complying with both the MIT license and our Terms of Use. Applio and its contributors are not liable for misuse. For full legal details, see the [Terms of Use](./TERMS_OF_USE.md).

## Getting Started

### 1. Installation

Run the installation script based on your operating system:

- **Windows:** Double-click `run-install.bat`.
- **Linux/macOS:** Execute `run-install.sh`.

> [!NOTE]
> On Linux, the realtime voice conversion tab imports the `sounddevice` Python package unconditionally at startup, which requires the system-level PortAudio shared library. If it is not already present, `run-applio.sh` (`python app.py`) will fail to start entirely — even if you never intend to use the realtime tab — with:
> ```
> OSError: PortAudio library not found
> ```
> Install it before running Applio, e.g. on Debian/Ubuntu:
> ```
> sudo apt-get install -y libportaudio2
> ```
> (or the equivalent package for your distribution).

### 2. Running Applio

Start Applio using:

- **Windows:** Double-click `run-applio.bat`.
- **Linux/macOS:** Run `run-applio.sh`.

This launches the Gradio interface in your default browser.

If the GUI doesn't come up in your environment, Applio can also run fully headless via the CLI, e.g.:

```
python core.py infer --pitch 0 --index_rate 0.75 --volume_envelope 1 --protect 0.5 --hop_length 128 --f0_method rmvpe --input_path input.wav --output_path output.wav --pth_path logs/<model>/model.pth --index_path logs/<model>/model.index --split_audio False --f0_autotune False --clean_audio False --clean_strength 0.5 --export_format WAV --embedder_model contentvec --sid 0
```

This is the same entry point used by the official [No-UI Colab notebook](https://colab.research.google.com/github/iahispano/applio/blob/main/assets/Applio_NoUI.ipynb) and requires no GUI/browser.

### 3. Optional: TensorBoard Monitoring

To monitor training or visualize data:

- **Windows:** Run `run-tensorboard.bat`.
- **Linux/macOS:** Run `run-tensorboard.sh`.

For more detailed instructions, visit the [documentation](https://docs.applio.org).

## References

Applio is made possible thanks to these projects and their references:

- [gradio-screen-recorder](https://huggingface.co/spaces/gstaff/gradio-screen-recorder) by gstaff
- [rvc-cli](https://github.com/blaisewf/rvc-cli) by blaisewf

### Contributors

<a href="https://github.com/IAHispano/Applio/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=IAHispano/Applio" />
</a>
