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
  
<p align="center">VITS-based Voice Conversion focused on simplicity, quality, and performance.</p>

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

## Table of Contents

- [Installation](#installation)
  - [Windows](#windows)
  - [macOS](#macos)
  - [Linux](#linux)
  - [Makefile](#makefile)
- [Usage](#usage)
  - [Windows](#windows-1)
  - [macOS](#macos-1)
  - [Linux](#linux-1)
  - [Makefile](#makefile-1)
- [Technical Information](#technical-information)
- [Repository Enhancements](#repository-enhancements)
- [Commercial Usage](#commercial-usage)
- [References](#references)
  - [Contributors](#contributors)

## Installation

Download the latest version from [GitHub Releases](https://github.com/IAHispano/Applio-RVC-Fork/releases) or use the [Compiled Versions](https://huggingface.co/IAHispano/Applio/tree/main/Compiled).

### Windows

```bash
./run-install.bat
```

### macOS

For macOS, you need to install the requirements in a Python environment version 3.9 to 3.11. Here are the steps:

```bash
python3 -m venv .venv
source .venv/bin/activate
chmod +x run-install.sh
./run-install.sh
```

### Linux

Certain Linux-based operating systems may encounter complications with the installer. In such instances, we suggest installing the `requirements.txt` within a Python environment version 3.9 to 3.11.

```bash
chmod +x run-install.sh
./run-install.sh
```

### Makefile

For platforms such as [Paperspace](https://www.paperspace.com/):

```bash
make run-install
```

## Usage

Visit [Applio Documentation](https://docs.applio.org/) for a detailed UI usage explanation.

### Windows

```bash
./run-applio.bat
```

### macOS

```bash
chmod +x run-applio.sh
./run-applio.sh
```

### Linux

```bash
chmod +x run-applio.sh
./run-applio.sh
```

### Makefile

For platforms such as [Paperspace](https://www.paperspace.com/):

```bash
make run-applio
```

## Technical Information

Applio uses an enhanced version of the Retrieval-based Voice Conversion (RVC) model, a powerful technique for transforming the voice of an audio signal to sound like another person. This advanced implementation of RVC in Applio enables high-quality voice conversion while maintaining simplicity and performance.

### 0. Pre-Learning: Key Concepts in Speech Processing and Voice Conversion

This section introduces fundamental concepts in speech processing and voice conversion, paving the way for a deeper understanding of the RVC pipeline:

#### 1. Speech Representation

- **Phoneme:** The smallest unit of sound in a language that distinguishes one word from another. Examples: /k/, /æ/, /t/.
- **Spectrogram:** A visual representation of the frequency content of a sound over time, showing how the intensity of different frequencies changes over the duration of the audio.
- **Mel-Spectrogram:** A type of spectrogram that mimics human auditory perception, emphasizing frequencies that are more important to human hearing.
- **Speaker Embedding:** A vector representation that captures the unique acoustic characteristics of a speaker's voice, encoding information about pitch, tone, timbre, and other vocal qualities.

#### 2. Text-to-Speech (TTS)

- **TTS Model:** A machine learning model that generates artificial speech from written text.
- **Encoder-Decoder Architecture:** A common architecture in TTS models, where an encoder processes the text and pitch information to create a latent representation, and a decoder uses this representation to synthesize the audio signal.
- **Transformer Architecture:** A powerful neural network architecture particularly well-suited for sequence modeling, allowing the model to handle long sequences of text or audio and capture relationships between elements.

#### 3. Voice Conversion

- **Voice Conversion (VC):** The process of transforming the voice of a speaker in an audio signal to sound like another speaker.
- **Speaker Adaptation:** The process of adapting a TTS model to a specific speaker, often by training on a small dataset of the speaker's voice.
- **Retrieval-Based VC (RVC):** A voice conversion approach where speaker embeddings are retrieved from a database and used to guide the TTS model in synthesizing audio with the target speaker's voice.

#### 4. Additional Concepts

- **ContentVec:** A powerful self-supervised learning model for speech representation, excelling at capturing speaker-specific information.
- **FAISS:** A library for efficient similarity search, used to retrieve speaker embeddings that are similar to the extracted ContentVec embedding.
- **Neural Source Filter (NSF):** A module that models audio generation as a filtering process, allowing the model to produce high-quality and realistic audio signals by learning complex relationships between the source signal and the output waveform.

#### 5. Why are these concepts important?

Understanding these concepts is essential for appreciating the mechanics and capabilities of the RVC pipeline:

- **Speech Representation:** Different representations capture different aspects of speech, allowing for effective analysis and manipulation.
- **TTS Models:** The TTS model forms the foundation of RVC, providing the ability to synthesize audio from text and pitch.
- **Voice Conversion:** Voice conversion aims to transfer a speaker's identity to a different audio signal.
- **ContentVec and Speaker Embeddings:** ContentVec provides a powerful way to extract speaker-specific information, which is crucial for accurate voice conversion.
- **FAISS:** This library enables efficient speaker embedding retrieval, facilitating the selection of appropriate target voices.
- **NSF:** The NSF is a critical component of the TTS model, contributing to the generation of realistic and high-quality audio.

### 1. Model Architecture

The RVC model comprises two main components:

#### A. Encoder-Decoder Network

This network synthesizes audio based on text and pitch information while incorporating speaker characteristics from the ContentVec embedding.

**Encoder:**

- **Input:** Phoneme sequences (text representation) and pitch information (optional).
- **Embeddings:**
  - Phonemes are represented as vectors using linear layers, creating a dense representation of the text input.
  - Pitch is usually converted to a one-hot encoding or a continuous value and embedded similarly.
- **Transformer Encoder:** Processes the embedded features in a highly parallel manner.

  It employs:

  - **Self-Attention:** Allows the encoder to attend to different parts of the input sequence to understand the relationships between words and their context.
  - **Feedforward Networks (FFN):** Apply non-linear transformations to further refine the features captured by self-attention.
  - **Layer Normalization:** Stabilizes training and improves performance by normalizing the outputs of each layer.
  - **Dropout:** A regularization technique to prevent overfitting.
  - **Output:** Produces a latent representation of the input text and pitch, capturing their relationships and serving as the input for the decoder.

**Decoder:**

- **Input:** The latent representation from the encoder.
- **Transformer Decoder:** Receives the encoder output and utilizes:
  - **Self-Attention:** Allows the decoder to attend to different parts of the generated sequence to maintain consistency and coherence in the output audio.
  - **Encoder-Decoder Attention:** Enables the decoder to incorporate information from the input text and pitch into the audio generation process.
- **Neural Source Filter (NSF):** A powerful component for generating audio, modeling the generation process as a filter applied to a source signal. It uses:
  - **Upsampling:** Increases the resolution of the latent representation to match the desired length of the audio signal.
  - **Residual Blocks:** Learn complex and non-linear relationships between input features and the output audio, contributing to realistic and detailed waveforms.
  - **Source Module:** Generates the excitation signal (often harmonic) that drives the NSF. It combines sine waves (for voiced sounds) and noise (for unvoiced sounds) to create a natural source signal.
  - **Noise Convolution:** Convolves noise with the harmonic signal to introduce additional variation and realism.
  - **Final Convolutional Layer:** Converts the filtered output to a single-channel audio waveform.
- **Output:** Synthesized audio signal.

#### B. ContentVec Speaker Embedding Extractor

Extracts speaker-specific information from the input audio.

- **Input:** The preprocessed audio signal.
- **Processing:** The ContentVec model, trained on a massive dataset of speech data, processes the input audio and extracts a speaker embedding vector, capturing the unique acoustic properties of the speaker's voice.
- **Output:** A speaker embedding vector representing the voice of the speaker.

### 2. Training Stage

The RVC model is trained using a combination of two key losses:

- **Generative Loss:**
  - **Mel-Spectrogram:** The Mel-spectrogram is computed for both the target audio and the generated audio.
  - **L1 Loss:** Measures the absolute difference between the Mel-spectrograms of the target and generated audio, encouraging the decoder to produce audio with a similar spectral profile.
- **Discriminative Loss:**
  - **Multi-Period Discriminator:** Tries to distinguish between real and generated audio at different time scales, using convolution layers to capture long-term dependencies in the audio.
  - **Adversarial Training:** The generator tries to fool the discriminator by producing audio that sounds real, while the discriminator is trained to correctly identify generated audio.
- **Optional KL Divergence Loss:** Measures the difference between the distributions of latent variables generated by the encoder and a posterior encoder (which infers the latent representation from the target audio). Encourages the model to learn a more efficient and stable latent representation.

### 3. Inference Stage

The inference stage utilizes the trained model to convert the voice of an audio input to sound like a target speaker. Here's a breakdown:

**Input:**

- Phoneme sequences (text representation).
- Pitch information (optional).
- Target speaker ID (identifies the desired voice).

**Steps:**

- **ContentVec Embedding Extraction:**
  - The ContentVec model processes the input audio and extracts a speaker embedding vector, capturing the voice characteristics of the speaker.
- **Optional Embedding Retrieval:**
  - **FAISS Index:** Used to efficiently search for speaker embeddings similar to the extracted ContentVec embedding. It helps guide the voice conversion process toward a specific speaker when multiple speakers are available.
  - **Embedding Retrieval:** The FAISS index is queried using the extracted ContentVec embedding, and similar embeddings are retrieved.
- **Embedding Manipulation:**
  - **Blending:** The extracted ContentVec embedding can be blended with retrieved embeddings using the index_rate parameter, allowing control over how much the target speaker's voice influences the conversion.
- **Encoder-Decoder Processing:**
  - **Encoder:** Encodes the phoneme sequences and pitch into a latent representation, capturing the relationships between them.
  - **Decoder:** Synthesizes the audio signal, incorporating the speaker characteristics from the ContentVec embedding (potentially blended with retrieved embeddings).
- **Post-Processing:**
  - **Resampling:** Adjusts the sampling rate of the generated audio if needed.
  - **RMS Adjustment:** Adjusts the volume (RMS) of the output audio to match the input audio.

### 4. Key Techniques

- **Transformer Architecture:** The Transformer architecture is a powerful tool for sequence modeling, enabling the encoder and decoder to efficiently process long sequences and capture complex relationships within the data.
- **Neural Source Filter (NSF):** Models audio generation as a filtering process, allowing the model to produce high-quality and realistic audio signals by learning complex relationships between the source signal and the output waveform.
- **Flow-Based Generative Model:** Enables the model to learn complex probability distributions for the audio signal, leading to more realistic and diverse generated speech.
- **Multi-period Discriminator:** Helps improve the quality and realism of the generated audio by evaluating the audio at

different temporal scales and providing feedback to the generator.

- **Relative Positional Encoding:** Helps the model understand the relative positions of elements within the input sequences, improving the model's ability to handle long sequences and maintain context.

### 5. Future Challenges

Despite the advancements in Retrieval-Based Voice Conversion, several challenges and areas for future research remain:

- **Speaker Generalization:** Improving the ability of models to generalize to unseen speakers with minimal data.
- **Real-time Processing:** Enhancing the efficiency of models to support real-time voice conversion applications.
- **Emotional Expression:** Better capturing and transferring emotional nuances in voice conversion.
- **Noise Robustness:** Improving the robustness of voice conversion models to handle noisy and low-quality input audio.

## Repository Enhancements

This repository has undergone significant enhancements to improve its functionality and maintainability:

- **Modular Codebase:** Restructured codebase for better organization, readability, and maintenance.
- **Hop Length Implementation:** Improved efficiency and performance, especially on Crepe (formerly Mangio-Crepe), thanks to [@Mangio621](https://github.com/Mangio621/Mangio-RVC-Fork).
- **Translations in 30+ Languages:** Added support for over 30 languages.
- **Cross-Platform Compatibility:** Ensured seamless operation across various platforms.
- **Optimized Requirements:** Fine-tuned project requirements for enhanced performance.
- **Streamlined Installation:** Simplified installation process for a user-friendly setup.
- **Hybrid F0 Estimation:** Introduced a personalized 'hybrid' F0 estimation method utilizing nanmedian.
- **Easy-to-Use UI:** Implemented an intuitive user interface.
- **Plugin System:** Introduced a plugin system for extending functionality.
- **Overtraining Detector:** Implemented a detector to prevent excessive training.
- **Model Search:** Integrated model search feature for easy discovery.
- **Pretrained Models:** Added support for custom pretrained models.
- **Voice Blender:** Developed a feature to combine two trained models to create a new one.
- **Accessibility Improvements:** Enhanced with descriptive tooltips for UI elements.
- **New F0 Extraction Methods:** Introduced methods like FCPE or Hybrid for pitch extraction.
- **Output Format Selection:** Added feature to choose audio file formats.
- **Hashing System:** Assigned unique IDs to models to prevent unauthorized duplication.
- **Model Download System:** Supported downloads from various platforms.
- **TTS Enhancements:** Improved Text-to-Speech functionality.
- **Split Audio:** Implemented audio splitting for faster processing.
- **Discord Presence:** Displayed usage status on Discord.
- **Flask Integration:** Enabled automatic model downloads via Flask.
- **Support Tab:** Added a tab for screen recording to report issues.

These enhancements contribute to a more robust and scalable codebase, making the repository more accessible for contributors and users alike.

## Commercial Usage

For commercial purposes, please adhere to the guidelines outlined in the [MIT license](./LICENSE) governing this project. Prior to integrating Applio into your application, we kindly request that you contact us at support@applio.org to ensure ethical use.

Please note, the use of Applio-generated audio files falls under your own responsibility and must always respect applicable copyrights. We encourage you to consider supporting the continuous development and maintenance of Applio through a donation. 

Your cooperation and support are greatly appreciated. Thank you!

## References

Applio is possible to these projects and those cited in their references.

- [gradio-screen-recorder](https://huggingface.co/spaces/gstaff/gradio-screen-recorder) by gstaff
- [rvc-cli](https://github.com/blaise-tk/rvc-cli) by blaisewf

### Contributors

<a href="https://github.com/IAHispano/Applio/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=IAHispano/Applio" />
</a>
