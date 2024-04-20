<h1 align="center">
  <a href="https://applio.org" target="_blank"><img src="https://github.com/IAHispano/Applio/assets/133521603/a5cc5c72-ed68-48a5-954f-db9f1dc4e7de" alt="Applio"></a>
</h1>
  
<p align="center">
    <img alt="Contributors" src="https://img.shields.io/github/contributors/iahispano/applio?style=for-the-badge&color=00AA68" />
    <img alt="Release" src="https://img.shields.io/github/release/iahispano/applio?style=for-the-badge&color=00AA68" />
    <img alt="Stars" src="https://img.shields.io/github/stars/iahispano/applio?style=for-the-badge&color=00AA68" />
    <img alt="Fork" src="https://img.shields.io/github/forks/iahispano/applio?style=for-the-badge&color=00AA68" />
    <img alt="Issues" src="https://img.shields.io/github/issues/iahispano/applio?style=for-the-badge&color=00AA68" />
</p>
  
<p align="center">VITS-based Voice Conversion focused on simplicity, quality and performance</p>

<p align="center">
  <a href="https://applio.org" target="_blank">🌐 Website</a>
  •
  <a href="https://docs.applio.org" target="_blank">📚 Documentation</a>
  •
  <a href="https://discord.gg/iahispano" target="_blank">☎️ Discord</a>
</p>

<p align="center">
  <a href="https://github.com/IAHispano/Applio-Plugins" target="_blank">🛒 Plugins</a>
  •
  <a href="https://huggingface.co/IAHispano/Applio/tree/main/Compiled" target="_blank">📦 Compiled</a>
  •
  <a href="https://applio.org/playground" target="_blank">🎮 Playground</a>
  •
  <a href="https://colab.research.google.com/github/iahispano/applio/blob/master/assets/Applio.ipynb" target="_blank">🔎 Google Colab (UI)</a>
  •
  <a href="https://colab.research.google.com/github/iahispano/applio/blob/master/assets/Applio_NoUI.ipynb" target="_blank">🔎 Google Colab (No UI)</a>
</p>

## Content Table
- [**Installation**](#installation)
  - [Windows](#windows)
  - [Linux](#linux)
  - [Makefile](#makefile)
- [**Usage**](#usage)
  - [Windows](#windows-1)
  - [Linux](#linux-1)
  - [Makefile](#makefile-1)
- [**Repository Enhancements**](#repository-enhancements)
- [**References**](#references)
  - [Contributors](#contributors)

## Installation
Download the latest version from [GitHub Releases](https://github.com/IAHispano/Applio-RVC-Fork/releases) or use the [Compiled Versions](https://huggingface.co/IAHispano/Applio/tree/main/Compiled).

### Windows
```bash
./run-install.bat
```

### Linux
Certain Linux-based operating systems may encounter complications with the installer. In such instances, we suggest installing the `requirements.txt` within a Python environment version 3.9 to 3.11.
```bash
chmod +x run-install.sh
./run-install.sh
```

### Makefile
For platforms such as [Paperspace](https://www.paperspace.com/)
```
make run-install
```

## Usage
Visit [Applio Documentation](https://docs.applio.org/) for a detailed UI usage explanation.

### Windows
```bash
./run-applio.bat
```

### Linux
```bash
chmod +x run-applio.sh
./run-applio.sh
```

### Makefile
For platforms such as [Paperspace](https://www.paperspace.com/)
```
make run-applio
```

## Repository Enhancements

This repository has undergone significant enhancements to improve its functionality and maintainability:

- **Modular Codebase:** Restructured codebase following a modular approach for better organization, readability, and maintenance.
- **Hop Length Implementation:** Implemented hop length, courtesy of [@Mangio621](https://github.com/Mangio621/Mangio-RVC-Fork), boosting efficiency and performance, especially on Crepe (formerly Mangio-Crepe).
- **Translations in 30+ Languages:** Added support for translations in over 30 languages, enhancing accessibility for a global audience.
- **Cross-Platform Compatibility:** Ensured seamless operation across various platforms for a consistent user experience.
- **Optimized Requirements:** Fine-tuned project requirements for enhanced performance and resource efficiency.
- **Streamlined Installation:** Simplified installation process for a user-friendly setup experience.
- **Hybrid F0 Estimation:** Introduced a personalized 'hybrid' F0 estimation method utilizing nanmedian, combining F0 calculations from various methods to achieve optimal results.
- **Easy-to-Use UI:** Implemented a user-friendly interface for intuitive interaction.
- **Optimized Code & Dependencies:** Enhanced code and streamlined dependencies for improved efficiency.
- **Plugin System:** Introduced a plugin system for extending functionality and customization.
- **Overtraining Detector:** Implemented an overtraining detector which halts training once a specified epoch limit is reached, preventing excessive training.
- **Model Search:** Integrated a model search feature directly into the application interface, facilitating easy model discovery.
- **Enhancements in Pretrained Models:** Introduced additional functionalities such as custom pretrained models, allowing users to utilize their preferred pretrained models without requiring RVC1 pretrained models upon installation.
- **Voice Blender:** Developed a voice blender feature that combines two trained models to create a new one, offering versatility in model generation.
- **Accessibility Improvements:** Enhanced accessibility with descriptive tooltips indicating the function of each element in the user interface, making it more user-friendly for all users.
- **New F0 Extraction Methods:** Introduced new F0 extraction methods such as FCPE or Hybrid, expanding options for pitch extraction.
- **Output Format Selection:** Implemented an output format selection feature, allowing users to choose the format in which they want to save their audio files.
- **Hashing System:** Implemented a hashing system where each created model is assigned a unique ID to prevent unauthorized duplication or theft.
- **Model Download System:** Added support for downloading models from various websites such as Google Drive, Yandex, Pixeldrain, Discord, Hugging Face, or Applio.org, enhancing model accessibility.
- **TTS Enhancements:** Improved Text-to-Speech functionality with support for uploading TXT files, increasing flexibility in input methods.
- **Split Audio:** Implemented audio splitting functionality which divides audio into segments for inference, subsequently merging them to create the final audio, resulting in faster processing times and potentially better outcomes.
- **Discord Presence:** Displayed presence on Discord indicating active usage of Applio, with plans to incorporate different statuses based on activities within the application.
- **Flask Integration:** Integration with Flask, initially disabled by default, allows for automatic model downloads from the web by simply clicking the Applio button next to the model download button in the settings tab.
- **Support Tab:** Added a support tab enabling users to record their screen to demonstrate encountered issues, facilitating faster issue resolution by allowing users to create GitHub issues for review and troubleshooting.

These enhancements contribute to a more robust and scalable codebase, making the repository more accessible for contributors and users alike.

## Contributions
- **Backend Contributions:** If you want to contribute to the backend, make your pull requests [here](https://github.com/blaise-tk/RVC_CLI).
- **Frontend Contributions:** For interface or script-related contributions, feel free to contribute to this repository.

We appreciate all contributions ❤️

## References
- [gradio-screen-recorder](https://huggingface.co/spaces/gstaff/gradio-screen-recorder) by gstaff
- [RVC_CLI](https://github.com/blaise-tk/RVC_CLI) by blaise-tk

### Contributors
<a href="https://github.com/IAHispano/Applio/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=IAHispano/Applio" />
</a>
