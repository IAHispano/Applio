# Applio

Welcome to **Applio**, the ultimate voice cloning tool meticulously optimized for unrivaled power, modularity, and a user-friendly experience.

[![Precompiled Versions](https://img.shields.io/badge/Precompiled%20Versions-ffffff?style=flat-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAALEsAACxLAaU9lqkAAAHzSURBVDhPrVFNaBNBFP5m9idJU9tE2srS5KSNlxT00AhSEEH01GMOIkpPBSmCehE99aIgVOrJgzdP6kE8qoQKEqmIV00PhRQ3jSbdTbRVm7rbnfHNJLV4lX7w3sx8733z3szDvkIIkQyCoEB2otls9vfof8B6q0a9Xu8b2V66bTSez7BguU+FhT3eQfb8Q549d4sxttVL3RNKKS3xaeEVX797GkOxHkuQZF4Ikb3zZjmcOJvP5wNFc+UUwrXFWd6Y1yK3YWlOadx12o9Y4O7cqZyDKzpA2BUy7j6bRprBWwsw/yTdLfTNwL3HaeBnB0gF4KtPp1WuEuwJ2cYYLAPDjokbF9rUOpA+IHDzYhtIUOsxCyxqjalcJdDCYrHIpLA7aPwAOiEW38dx+XoGM9cy+Lhiak7HWKKjcpVGu99+5ZL9buoR7BY9NkJE98nPo5AxCXP0C1USgGkAO8MIT76ctQ8efqAr8u+VY0ATMOg4EIcxYMMc92DlfLCkrTmYFBNfgY3Kca1RDiZdLah4PyWlEsAg2eZ219RecSomGZgRV//WbbVWq2UObb29b7RfT/FwxQZPIUoWlqQUkfHrwySTm0zauXBn6MyLVnLyquM4q0q3C14qlQY9z5uoVqtH6UyPovG57hHf9wvlcpnm8ncK/wvgD6Orstc1XrkKAAAAAElFTkSuQmCC&link=https://huggingface.co/IAHispano/applio/tree/main/Applio%20V3%20Precompiled)](https://huggingface.co/IAHispano/applio/tree/main/Applio%20V3%20Precompiled)
![GitHub Release](https://img.shields.io/github/v/release/iahispano/applio-rvc-fork?style=flat-square)
![GitHub Repo stars](https://img.shields.io/github/stars/iahispano/applio-rvc-fork?style=flat-square)
![GitHub forks](https://img.shields.io/github/forks/iahispano/applio-rvc-fork?style=flat-square)
[![Support Discord](https://img.shields.io/discord/1096877223765606521?style=flat-square)](https://discord.gg/iahispano)
[![Issues](https://img.shields.io/github/issues/iahispano/applio-rvc-fork?style=flat-square)](https://github.com/IAHispano/Applio-RVC-Fork/issues)
[![Open In Collab](https://img.shields.io/badge/google_colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/iahispano/applio/blob/master/assets/Applio.ipynb)

## Content Table
- [**Installation**](#installation)
  - [Windows](#windows)
  - [Linux](#linux)
  - [Using Makefile](#using-makefile-for-platforms-such-as-paperspace)
- [**Usage**](#usage)
  - [Windows](#windows-1)
  - [Linux](#linux-1)
  - [Using Makefile](#using-makefile-for-platforms-such-as-paperspace-1)
- [**Repository Enhancements**](#repository-enhancements)
- [**Credits**](#credits)
  - [Contributors](#contributors)

## Installation
Download the latest version from [GitHub Releases](https://github.com/IAHispano/Applio-RVC-Fork/releases) or use [Precompiled Versions](https://huggingface.co/IAHispano/applio/tree/main/Applio%20V3%20Precompiled).

### Windows
```bash
./run-install.bat
```

### Linux
```bash
chmod +x run-install.sh
./run-install.sh
```

### Using Makefile (for platforms such as [Paperspace](https://www.paperspace.com/))
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

### Using Makefile (for platforms such as [Paperspace](https://www.paperspace.com/))
```
make run-applio
```

## Repository Enhancements

This repository has undergone significant improvements to enhance its functionality and maintainability:

- **Code Modularization:** The codebase has been restructured to follow a modular approach. This ensures better organization, readability, and ease of maintenance.
- **Hop Length Implementation:** Special thanks to [@Mangio621](https://github.com/Mangio621/Mangio-RVC-Fork) for introducing hop length implementation. This enhancement enhances the efficiency and performance on Crepe (previously known as Mangio-Crepe).
- **Translations to +30 Languages:** The repository now supports translations in over 30 languages, making it more accessible to a global audience.
- **Cross-Platform Compatibility:** With multiplatform compatibility, this repository can seamlessly operate across various platforms, providing a consistent experience to users.
- **Optimized Requirements:** The project's requirements have been fine-tuned for improved performance and resource utilization.
- **Simple Installation:** The installation process has been streamlined, ensuring a straightforward and user-friendly experience for setup.

These enhancements contribute to a more robust and scalable codebase, making the repository more accessible for contributors and users alike.

## Contributions
- **Backend Contributions:** If you want to contribute to the backend, make your pull requests [here](https://github.com/blaise-tk/RVC_CLI).
- **Frontend Contributions:** For interface or script-related contributions, feel free to contribute to this repository.

We appreciate all contributions ❤️

## Planned Features
- Implement: Support for Apple Devices ([Issue Link](https://github.com/pytorch/pytorch/issues/77764))
- Implement: rmvpe_gpu
- Implement: Theme selector, RPC toggle & version checker
- Implement: Overtraining detector
- Implement: Autotune
- Implement: Training stop
- Fix: Model fusion

## Credits
- [VITS](https://github.com/jaywalnut310/vits) by jaywalnut310
- [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) by RVC-Project
- [Mangio-RVC-Fork](https://github.com/Mangio621/Mangio-RVC-Fork) by Mangio621
- [Mangio-RVC-Tweaks](https://github.com/alexlnkp/Mangio-RVC-Tweaks) by alexlnkp
- [RVG_tts](https://github.com/Foxify52/RVG_tts) by Foxify52
- [RMVPE](https://github.com/Dream-High/RMVPE) by Dream-High
- [ContentVec](https://github.com/auspicious3000/contentvec/) by auspicious3000
- [HIFIGAN](https://github.com/jik876/hifi-gan) by jik876
- [Gradio](https://github.com/gradio-app/gradio) by gradio-app
- [FFmpeg](https://github.com/FFmpeg/FFmpeg) by FFmpeg
- [audio-slicer](https://github.com/openvpi/audio-slicer) by openvpi
- [Ilaria-Audio-Analyzer](https://github.com/TheStingerX/Ilaria-Audio-Analyzer) by TheStingerX
- [gradio-screen-recorder](https://huggingface.co/spaces/gstaff/gradio-screen-recorder) by gstaff
- [RVC_CLI](https://github.com/blaise-tk/RVC_CLI) by blaise-tk

### Contributors
<a href="https://github.com/IAHispano/Applio/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=IAHispano/Applio" />
</a>
