# HSVC — Harmonic-Spectral Voice Conversion (48 kHz)

**HSVC** (*Harmonic-Spectral Voice Conversion*) is Applio's native 48 kHz voice conversion architecture designed as a modern, real-time-capable counterpart to classic RVC v2 (both architectures remain available side-by-side in training, and inference/realtime auto-detect checkpoints automatically).

---

## Architecture Overview

Instead of upsampling raw waveforms through transposed convolutions (`ConvTranspose1d`), HSVC operates directly in the **harmonic-spectral STFT domain** at 48 kHz (`hop = 240` samples = `5 ms`, `n_fft = 2048`, `win = 960` Hann window):

```text
ContentVec (50 Hz -> 100 Hz linear) ──► Speaker-Invariant Bottleneck (768 -> 256 -> 384)
                                                    │
F0 + VUV Conf + 10 ms Energy + HF Energy ──► Prosody Frontend (19-d Fourier/Dynamics)
                                                    │
                                     ┌──────────────▼──────────────┐
                                     │   Neural Timbre Memory Bank │  (16 phoneme/register prototypes/spk)
                                     └──────────────┬──────────────┘
                                                    │
                     100 Hz Bounded-Lookahead ConvNeXt-FiLM Backbone (8 blocks, 20 ms lookahead)
                                                    │
                     200 Hz Sub-Frame ConvNeXt-FiLM Refiner (4 blocks, 5 ms lookahead)
                                                    │
             ┌──────────────────────────────────────┴──────────────────────────────────────┐
             ▼                                                                             ▼
   Bandlimited Alias-Free Source                                             FreqCoord 2D Spectral Refiner
  • Aligned + golden-ratio phase-scattered pulses                           • 128-bin ERB + log/lin freq coords
  • Strictly energy-gated glottal-sync noise                                • 4 × 2D ConvNeXt-GLU FiLM blocks
             │                                                                             │
             └─────────────────────────► 256-Band ERB Minimum-Phase ◄──────────────────────┘
                                          Complex Harmonic & Noise Filtering
                                                    │
                                   Silence-Gated 2048-pt iSTFT (48 kHz)
```

### Core Modules
- [`model.py`](model.py) — `HSVCConfig`, `HSVCGenerator`, checkpoint auto-detection (`is_hsvc_checkpoint`), and FP32/FP16 package export/load.
- [`source.py`](source.py) — `BandlimitedSourceGenerator`: closed-form alias-free raised-cosine-tapered harmonic pulses (`aligned` + 7-class golden-ratio `scattered`) and strictly energy-gated glottal-synchronized noise (`-240 dBFS` pure digital silence floor).
- [`spectral.py`](spectral.py) — ERB/Mel minimum-phase Hilbert cepstral basis (`build_erb_interpolation_matrix`), bounded-lookahead 1D `StreamingConvNeXtFiLMBlock`, 2D `FreqCoordSpectralBlock`, and `NeuralTimbreBank`.
- [`discriminators.py`](discriminators.py) — `DualDomainDiscriminator` (compact MPD + 3-scale Hann-windowed Multi-Resolution STFT discriminators), utterance-level GRL `UtteranceSpeakerAdversary`, and SpecAugment-hardened `HardenedSpectralSpeakerCritic`.
- [`losses.py`](losses.py) — `HSVCLoss`: 4-scale MR-STFT spectral convergence + log-magnitude, HF-weighted log-Mel L1, temporal/frequency Mel deltas, 10 ms RMS loudness matching, and explicit silence-floor suppression.
- [`streaming.py`](streaming.py) — `HSVCStreamer`: bounded-context, phase-continuous overlap-add streaming wrapper.

### Key Design Choices
- **Zero transposed convolutions & zero unmodulated STFT residuals (`use_spec_res=False`)**: eliminates HiFi-GAN checkerboard aliasing and prevents stationary `200 Hz` (`sr / hop_dec`) STFT comb lines.
- **Strict energy-gated excitation**: silence frames drop cleanly below `-108` to `-120 dBFS` in live realtime streaming without needing aggressive noise gates.
- **Frozen GRL-purified bottleneck on fine-tuning**: keeps cross-speaker disentanglement intact during single-speaker fine-tuning while adapting timbre in ~1.9–2.0× less time and ~1.4 GB less VRAM than RVC v2.
- **Drop-in Applio compatibility**: works with Applio's existing ContentVec embedder, RMVPE / FCPE / SwiftF0 pitch extractors, FAISS `.index` retrieval, WebUI Inference, and live Realtime `VoiceChanger`.

---

## Pretrained Model Info (`HSVC-Base-48k`)

| Property | Value |
|---|---|
| **Architecture / Version** | `hsvc` (`v1`), `48000 Hz`, ContentVec (768-d) |
| **Parameters** | Generator `19.57M` (`78.4 MB` FP32 / `39.2 MB` FP16) · Discriminator `10.99M` (`44.0 MB`) |
| **Pretraining Corpus** | `104.45 hours` (`102,810` clips) across `145` speakers & singers in `9` languages (VCTK, Expresso, VocalSet, GTSinger — strictly holding out unseen eval speakers) |
| **Training Schedule** | `100,000 steps` (`batch_size=12`, ~`267 hours` audio seen), including `90k → 100k` Frozen-ContentVec Phonetic-Anchored Polish (`use_spec_res=False`) |
| **Expected Local Path** | `rvc/models/pretraineds/hsvc/hsvc_base_48k.pth` & `hsvc_base_48k_d.pth` |
| **HuggingFace Auto-Download** | `https://huggingface.co/IAHispano/Applio/resolve/main/Resources/hsvc/` (override via `APPLIO_HSVC_PRETRAIN_URL`) |

### Checksums (100k Release)
- `hsvc_base_48k.pth` (FP32, 78.4 MB): `a4f2d05e2fb4acb1cf1f9e58bebbb8f6addb45682339b922e1fe9b885b91e51e`
- `hsvc_base_48k_fp16.pth` (FP16, 39.2 MB): `82c0a85a638610d9711d4301565bb80f66dd57e81dfe9554ca621710dbeb0529`
- `hsvc_base_48k_d.pth` (Disc, 44.0 MB): `5acb06cb9ec33c4354502ec0b65adfbf22610da2b28906794abd99e63e69058b`

### Quick Usage
- **WebUI Train:** Select `Architecture → HSVC` in the Train tab (`48000 Hz` and `HSVC` vocoder are selected automatically).
- **CLI Train:**
  ```bash
  python core.py train --model-name my_voice --architecture HSVC --sample-rate 48000 --total-epoch 50 --save-every-epoch 10 --batch-size 6
  ```
- **Inference & Realtime:** Select any `.pth` model (HSVC or classic RVC v2); Applio automatically detects the architecture.
