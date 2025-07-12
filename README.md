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
  <a href="https://discord.gg/urxFjYmYYh" target="_blank">☎️ Discord</a>
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

> [!NOTE]  
> Applio will no longer receive frequent updates. Going forward, development will focus mainly on security patches, dependency updates, and occasional feature improvements. This is because the project is already stable and mature with limited room for further improvements. Pull requests are still welcome and will be reviewed.

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

### 2. Running Applio

Start Applio using:

- **Windows:** Double-click `run-applio.bat`.
- **Linux/macOS:** Run `run-applio.sh`.

This launches the Gradio interface in your default browser.

### 3. Optional: TensorBoard Monitoring

To monitor training or visualize data:

- **Windows:** Run `run-tensorboard.bat`.
- **Linux/macOS:** Run `run-tensorboard.sh`.

For more detailed instructions, visit the [documentation](https://docs.applio.org).

## Docker Support

Applio provides full Docker support with Docker Compose for easy deployment and management, especially useful for NAS systems and container orchestration platforms like Portainer.

### Prerequisites

- Docker Engine 20.10+ and Docker Compose v2.0+
- For GPU support: NVIDIA Docker runtime and compatible GPU drivers

### Quick Start with Docker Compose

1. **Clone the repository:**
   ```bash
   git clone https://github.com/IAHispano/Applio.git
   cd Applio
   ```

2. **Copy and configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env file with your preferred settings
   ```

3. **Run with Docker Compose:**
   ```bash
   # CPU-only version
   docker compose up -d
   
   # OR GPU-enabled version (requires NVIDIA Docker)
   docker compose --profile gpu up -d
   ```

4. **Access Applio:**
   Open your browser and navigate to `http://localhost:6969`

### Configuration Options

The `.env` file contains configuration options:

- `APPLIO_HOST`: Host interface (default: 0.0.0.0)
- `APPLIO_PORT`: Port number (default: 6969)
- `APPLIO_SHARE`: Enable public sharing (default: false)
- `APPLIO_DEBUG`: Enable debug mode (default: false)
- `CUDA_VISIBLE_DEVICES`: GPU selection for GPU profile

### Volume Persistence

The Docker setup automatically creates persistent volumes for:
- `applio_logs`: Application logs
- `applio_models`: AI models
- `applio_audio`: Audio files
- `applio_pretrained`: Pre-trained models
- `applio_weights`: Model weights
- `applio_cache`: Application cache
- `applio_config`: Configuration files

### Portainer Stack Deployment

For Portainer users, you can deploy Applio as a stack:

1. In Portainer, go to "Stacks" → "Add Stack"
2. Choose "Repository" and enter: `https://github.com/IAHispano/Applio.git`
3. Set the compose file path to `docker-compose.yaml`
4. Configure environment variables as needed
5. Deploy the stack

**App Template:** You can also use the included `portainer-app-template.json` for easier deployment through Portainer's app templates feature.

### Docker Commands

```bash
# Start services
docker compose up -d

# Start with GPU support
docker compose --profile gpu up -d

# Stop services
docker compose down

# View logs
docker compose logs -f applio

# Update and rebuild
docker compose down
docker compose build --no-cache
docker compose up -d

# Remove all data (WARNING: This will delete all your models and data!)
docker compose down -v
```

### Troubleshooting

**Permission Issues:** Ensure Docker daemon is running and your user has Docker permissions.

**GPU Not Detected:** Verify NVIDIA Docker runtime is installed and configured properly.

**Port Conflicts:** Change the `APPLIO_PORT` in your `.env` file if port 6969 is in use.

**Volume Issues:** Check Docker volume permissions and available disk space.

## References

Applio is made possible thanks to these projects and their references:

- [gradio-screen-recorder](https://huggingface.co/spaces/gstaff/gradio-screen-recorder) by gstaff
- [rvc-cli](https://github.com/blaisewf/rvc-cli) by blaisewf

### Contributors

<a href="https://github.com/IAHispano/Applio/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=IAHispano/Applio" />
</a>
