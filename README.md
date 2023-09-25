# 🍏 Applio-RVC-Fork (V2)
Applio is a user-friendly fork of Mangio-RVC-Fork/RVC, designed to provide an intuitive interface, especially for newcomers.
<br>

[![Discord](https://img.shields.io/badge/SUPPORT_DISCORD-37a779?style=for-the-badge)](https://discord.gg/IAHispano) [![Docs](https://img.shields.io/badge/DOCS-37a779?style=for-the-badge)](https://docs.applio.org)


## 📚 Table of Contents
  1. [Improvements of Applio Over RVC](#-improvements-of-applio-over-rvc)
  2. [Additional Features of This Repository](#️-additional-features-of-this-repository)
  3. [Planned Features for Future Development](#️-planned-features-for-future-development)
  4. [Installation](#-installation)
  5. [Running the Web GUI (Inference & Train)](#-running-the-web-gui-inference--train)
  6. [Running the CLI (Inference & Train)](#-running-the-cli-inference--train)
  7. [Credits](#-credits)
  8. [Thanks to all RVC, Mangio and Applio contributors](#-thanks-to-all-rvc-mangio-and-applio-contributors)


## 🎯 Improvements of Applio Over RVC
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
- Applio introduces command-line interface (CLI) functionality, with the addition of the --is_cli flag in infer-web.py for CLI system usage.

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
- Support for multiple languages and Edge-tts/Bark-tts.

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

### Universal Compatibility
- Acceleration support for AMD/Intel graphics cards and enhanced acceleration for Intel ARC graphics cards, including IPEX compatibility.

## 🛠️ Planned Features for Future Development
As part of the ongoing development of this fork, the following features are planned to be added:

- Incorporating an inference batcher script based on user feedback. This enhancement will allow for processing 30-second audio samples at a time, improving output quality and preventing memory errors during inference.
- Implementing an automatic removal mechanism for old generations to optimize storage space usage. This feature ensures that the repository remains efficient and organized over time.
- Streamlining the training process for Paperspace machines to further improve efficiency and resource utilization during training tasks.

## ✨ Installation

### Automatic installation (Windows):
To quickly and effortlessly install Applio along with all the necessary models and configurations on Windows, you can use the [install_Applio.bat](https://github.com/IAHispano/Applio-RVC-Fork/releases) script available in the releases section.

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
*Using pip (Python 3.9.8 is stable with this fork)*
```bash
for Nvidia graphics cards:
  pip install -r assets/requirements/requirements.txt

for AMD / Intel graphics cards：
  pip install -r assets/requirements/requirements-dml.txt

for Intel ARC graphics cards on Linux / WSL using Python 3.10: 
  pip install -r assets/requirements/requirements-ipex.txt
```

### Manual installation (Paperspace):
```bash
cd Applio-RVC-Fork
make install # Do this everytime you start your paperspace machine
```

## 🪄 Running the Web GUI (Inference & Train) 
*Use --paperspace or --colab if on cloud system.*
```bash
python infer-web.py --pycmd python --port 3000
```

## 💻 Running the CLI (Inference & Train) 
```bash
python infer-web.py --pycmd python --is_cli
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

+ [Retrieval-based-Voice-Conversion-WebUI](Retrieval-based-Voice-Conversion-WebUI)
+ [Mangio-RVC-Fork](https://github.com/Mangio621/Mangio-RVC-Fork)
+ [RVG_tts](https://github.com/Foxify52/RVG_tts)
+ [ContentVec](https://github.com/auspicious3000/contentvec/)
+ [VITS](https://github.com/jaywalnut310/vits)
+ [HIFIGAN](https://github.com/jik876/hifi-gan)
+ [Gradio](https://github.com/gradio-app/gradio)
+ [FFmpeg](https://github.com/FFmpeg/FFmpeg)
+ [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
+ [audio-slicer](https://github.com/openvpi/audio-slicer)
+ [RMVPE](https://github.com/Dream-High/RMVPE)


## 🙏 Thanks to all RVC, Mangio and Applio contributors
### RVC:

<a href="https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=liujing04/Retrieval-based-Voice-Conversion-WebUI" />
</a>

### Mangio:

<a href="https://github.com/Mangio621/Mangio-RVC-Fork/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=Mangio621/Mangio-RVC-Fork" />
</a>

### Applio:

<a href="https://github.com/IAHispano/Applio-RVC-Fork/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=IAHispano/Applio-RVC-Fork" />
</a>
