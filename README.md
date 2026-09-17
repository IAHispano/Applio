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

### Recommended: Windows installer

1. Download `Applio-Setup-<version>.exe` from the [releases page](https://github.com/IAHispano/Applio/releases) and run it (per-user install, no admin rights needed).
2. Launch **Applio** from the Start Menu. On first launch the setup screen checks every dependency (Python env, engine packages, ffmpeg, web build) and installs whatever is missing — just press **Install / Repair** and wait.
3. Every later launch re-runs the checks, so a broken environment is caught before you hit Convert.

### Developers (any OS)

Prerequisites: Python 3.10–3.12, Node.js 20+, ffmpeg.

```bash
npm install
npm run dev    # API :8000 + web :3000 with hot reload
# open http://localhost:3000/  (setup screen)
```

First run: open the app, press **Install / Repair** on the setup screen (creates `.venv`, installs torch + requirements, builds the UI). Useful scripts: `npm test` (smoke suite), `npm run format` (Biome), `npm run build`.

### Other ways to run

- **TensorBoard:** built into the UI — open the TensorBoard tab and press Launch.
- **Docker:** `docker compose -f docker/docker-compose.yml up --build`.
- **Colab/Kaggle:** see `assets/Applio.ipynb` / `assets/Applio_Kaggle.ipynb` (CLI-only: `assets/Applio_NoUI.ipynb`).
- **Package the installer yourself:** `npm run build && npm run dist:win` (`.exe` lands in `app/desktop/dist-installers/`).

## References

Applio is made possible thanks to these projects and their references:

- [gradio-screen-recorder](https://huggingface.co/spaces/gstaff/gradio-screen-recorder) by gstaff
- [rvc-cli](https://github.com/blaisewf/rvc-cli) by blaisewf

### Contributors

<a href="https://github.com/IAHispano/Applio/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=IAHispano/Applio" />
</a>
