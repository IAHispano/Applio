# 🍏 Applio-RVC-Fork

> [!NOTE]
> Applio-RVC-Fork is designed to complement existing repositories, and as such, certain features may be in experimental stages, potentially containing bugs. Additionally, there might be instances of coding practices that could be improved or refined. It is not intended to replace any other repository.

[![Discord](https://img.shields.io/badge/SUPPORT_DISCORD-37a779?style=for-the-badge)](https://discord.gg/IAHispano) [![Discord Bot](https://img.shields.io/badge/DISCORD_BOT-37a779?style=for-the-badge)](https://bot.applio.org) [![Docs](https://img.shields.io/badge/DOCS-37a779?style=for-the-badge)](https://docs.applio.org)

## 📚 Table of Contents

_This README has been enhanced by incorporating the features introduced in Applio-RVC-Fork to the original [Mangio-RVC-Fork README](https://github.com/Mangio621/Mangio-RVC-Fork/blob/main/README.md), along with additional details and explanations._

1. [Improvements of Applio Over RVC](#-improvements-of-applio-rvc-fork-over-rvc)
2. [Additional Features of This Repository](#️-additional-features-of-this-repository)
3. [Todo Tasks](#-todo-tasks)
4. [Installation](#-installation)
5. [Running the Web GUI (Inference & Train)](#-running-the-web-gui-inference--train)
6. [Running the CLI (Inference & Train)](#-running-the-cli-inference--train)
7. [Credits](#-credits)
8. [Thanks to all RVC, Mangio and Applio contributors](#-thanks-to-all-rvc-mangio-and-applio-contributors)

## 🎯 Improvements of Applio-RVC-Fork Over RVC

_The comparisons are with respect to the original [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) repository._

### f0 Inference Algorithm Overhaul

- Applio features a comprehensive overhaul of the f0 inference algorithm, including:
  - Addition of the pyworld dio f0 method.
  - Alternative method for calculating crepe f0.
  - Introduction of the torchcrepe crepe-tiny model.
  - Customizable crepe_hop_length for the crepe algorithm via both the web GUI and CLI.

### f0 Crepe Pitch Extraction for Training

- Works on paperspace machines but not local MacOS/Windows machines (Potential memory leak).

### Paperspace Integration (Under maintenance, so it cannot be used for the moment.)

- Applio seamlessly integrates with Paperspace, providing the following features:
  - Paperspace argument on infer-web.py (--paperspace) for sharing a Gradio link.
  - A dedicated make file tailored for Paperspace users.

### Access to Tensorboard

- Applio grants easy access to Tensorboard via a Makefile and a Python script.

### CLI Functionality

- Applio introduces command-line interface (CLI) functionality, with the addition of the --cli flag in infer-web.py for CLI system usage.

### f0 Hybrid Estimation Method

- Applio offers a novel f0 hybrid estimation method by calculating nanmedian for a specified array of f0 methods, ensuring the best results from multiple methods (CLI exclusive).
- This hybrid estimation method is also available for f0 feature extraction during training.

### UI Changes

#### Inference:

- A complete interface redesign enhances user experience, with notable features such as:
  - Audio recording directly from the interface.
  - Convenient drop-down menus for audio and .index file selection.
  - An advanced settings section with new features like autotune and formant shifting.

#### Training:

- Improved training features include:
  - A total epoch slider now limited to 10,000.
  - Increased save frequency limit to 100.
  - Default recommended options for smoother setup.
  - Better adaptation to high-resolution screens.
  - A drop-down menu for dataset selection.
  - Enhanced saving system options, including Save all files, Save G and D files, and Save model for inference.

#### UVR:

- Applio ensures compatibility with all VR/MDX models for an extended range of possibilities.

#### TTS (Text-to-Speech, New):

- Introducing a new Text-to-Speech (TTS) feature using RVC models.
- Support for multiple languages and Edge-tts/Google-tts.

#### Resources (New):

- Users can now upload models, backups, datasets, and audios from various storage services like Drive, Huggingface, Discord, and more.
- Download audios from YouTube with the ability to automatically separate instrumental and vocals, offering advanced options and UVR support.

#### Extra (New):

- Combine instrumental and vocals with ease, including independent volume control for each track and the option to add effects like reverb, compressor, and noise gate.
- Significant improvements in the processing interface, allowing tasks such as merging models, modifying information, obtaining information, or extracting models effortlessly.

## ⚙️ Additional Features of This Repository

In addition to the aforementioned improvements, this repository offers the following features:

### Enhanced Tone Leakage Reduction

- Implements tone leakage reduction by replacing source features with training-set features using top1 retrieval. This helps in achieving cleaner audio results.

### Efficient Training

- Provides a seamless and speedy training experience, even on relatively modest graphics cards. The system is optimized for efficient resource utilization.

### Data Efficiency

- Supports training with a small dataset, yielding commendable results, especially with audio clips of at least 10 minutes of low-noise speech.

### Overtraining Detection

- This feature keeps track of the current progress trend and stops the training if no improvement is found after 100 epochs.
  - During the 100 epochs with no improvement, no progress is saved. This allows you to continue training from the best-found epoch.
  - A `.pth` file of the best epoch is saved in the logs folder under `name_[epoch].pth`, and in the weights folder as `name_fittest.pth`. These files are the same.

### Mode Collapse Detection

- This feature restarts training before a mode collapse by lowering the batch size until it can progress past the mode collapse.
  - If a mode collapse is overcome but another one occurs later, it will reset the batch size to its initial setting. This helps maintain training speed when dealing with multiple collapses.

## 📝 Todo Tasks

- [ ] **Investigate GPU Detection Issue:** Address the GPU detection problem and ensure proper utilization of Nvidia GPU.
- [ ] **Fix Mode Collapse Prevention Feature:** Refine the mode collapse prevention feature to maintain graph consistency during retraining.
- [ ] **Resolve CUDA Compatibility Issue:** Investigate and resolve the cuFFT error related to CUDA compatibility.
- [ ] **Refactor infer-web.py:** Organize the code of infer-web.py into different files for each tab, enhancing modularity.
- [ ] **Expand UVR Model Options:** Integrate additional UVR models to provide users with more options and flexibility.
- [x] **Enhance Installation Process:** Improve the system installation process for better user experience and clarity. [Applio Installer.exe](https://github.com/IAHispano/Applio-Installer/releases)
- [ ] **Implement Automatic Updates:** Add automatic update functionality to keep the application current with the latest features.
- [ ] **Multilingual Support:** Include more translations for various languages.
- [ ] **Diversify TTS Methods:** Introduce new TTS methods and enhance customization options for a richer user experience.
- [ ] **CLI Improvement:** Enhance the CLI functionality and introduce a pipeline for a more streamlined user experience.
- [ ] **Dependency Updates:** Keep dependencies up-to-date by regularly updating to the latest versions.
- [ ] **Dataset Creation Assistant:** Develop an assistant for creating datasets to simplify and guide users through the process.

## ✨ Installation

### Automatic installation (Windows):

To quickly and effortlessly install Applio along with all the necessary models and configurations on Windows, you can use the [Applio Installer.exe](https://github.com/IAHispano/Applio-Installer/releases) or the [install_Applio.bat](https://github.com/IAHispano/Applio-RVC-Fork/releases) script available in the releases section.

### Manual installation (Windows/MacOS):

**Note for MacOS Users**: When using `faiss 1.7.2` under MacOS, you may encounter a Segmentation Fault: 11 error. To resolve this issue, install `faiss-cpu 1.7.0` using the following command if you're installing it manually with pip:

```bash
pip install faiss-cpu==1.7.0
```

Additionally, you can install Swig on MacOS using brew:

```bash
brew install swig
```

Install requirements:
_Before this install ffmpeg, wget, git and python (This fork just works with 3.9.X on Linux)_

```bash
wget https://github.com/IAHispano/Applio-RVC-Fork/releases/download/v2.0.0/install_Applio-linux.sh
chmod +x install_Applio-linux.sh && ./install_Applio-linux.sh
```

### Manual installation (Paperspace):

```bash
cd Applio-RVC-Fork
make install # Do this everytime you start your paperspace machine
```

## 🪄 Running the Web GUI (Inference & Train)

_Use --paperspace or --colab if on cloud system._

```bash
python infer-web.py --pycmd python --port 3000
```

## 💻 Running the CLI (Inference & Train)

```bash
python infer-web.py --pycmd python --cli
```

```bash
Applio-RVC-Fork CLI

Welcome to the CLI version of RVC. Please read the documentation on README.MD to understand how to use this app.

You are currently in 'HOME':
    go home            : Takes you back to home with a navigation list.
    go infer           : Takes you to inference command execution.

    go pre-process     : Takes you to training step.1) pre-process command execution.
    go extract-feature : Takes you to training step.2) extract-feature command execution.
    go train           : Takes you to training step.3) being or continue training command execution.
    go train-feature   : Takes you to the train feature index command execution.

    go extract-model   : Takes you to the extract small model command execution.

HOME:
```

Typing 'go infer' for example will take you to the infer page where you can then enter in your arguments that you wish to use for that specific page. For example typing 'go infer' will take you here:

```bash
HOME: go infer
You are currently in 'INFER':
    arg 1) model name with .pth in ./weights: mi-test.pth
    arg 2) source audio path: myFolder\MySource.wav
    arg 3) output file name to be placed in './audio-outputs': MyTest.wav
    arg 4) feature index file path: logs/mi-test/added_IVF3042_Flat_nprobe_1.index
    arg 5) speaker id: 0
    arg 6) transposition: 0
    arg 7) f0 method: harvest (pm, harvest, crepe, crepe-tiny)
    arg 8) crepe hop length: 160
    arg 9) harvest median filter radius: 3 (0-7)
    arg 10) post resample rate: 0
    arg 11) mix volume envelope: 1
    arg 12) feature index ratio: 0.78 (0-1)
    arg 13) Voiceless Consonant Protection (Less Artifact): 0.33 (Smaller number = more protection. 0.50 means Dont Use.)

Example: mi-test.pth saudio/Sidney.wav myTest.wav logs/mi-test/added_index.index 0 -2 harvest 160 3 0 1 0.95 0.33

INFER: <INSERT ARGUMENTS HERE OR COPY AND PASTE THE EXAMPLE>
```

## 🏆 Credits

Applio owes its existence to the collaborative efforts of various repositories, including Mangio-RVC-Fork, and all the other credited contributors. Without their contributions, Applio would not have been possible. Therefore, we kindly request that if you appreciate the work we've accomplished, you consider exploring the projects mentioned in our credits.

Our goal is not to supplant RVC or Mangio; rather, we aim to provide a contemporary and up-to-date alternative for the entire community.

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
- [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) by Anjok07
- [audio-slicer](https://github.com/openvpi/audio-slicer) by openvpi
- [Ilaria-Audio-Analyzer](https://github.com/TheStingerX/Ilaria-Audio-Analyzer) by Ilaria

> [!WARNING]  
> If you believe you've made contributions to the code utilized in Applio and should be acknowledged in the credits, please feel free to open a pull request (PR). It's possible that we may have unintentionally overlooked your contributions, and we appreciate your proactive approach in ensuring proper recognition.

## 🙏 Thanks to all RVC, Mangio and Applio contributors

### RVC:

<a href="https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=liujing04/Retrieval-based-Voice-Conversion-WebUI" />
</a>

### Applio & Mangio:

<a href="https://github.com/IAHispano/Applio-RVC-Fork/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=IAHispano/Applio-RVC-Fork" />
</a>
