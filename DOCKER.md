# Docker Installation Guide for Applio

This guide provides detailed instructions for setting up and running Applio using Docker and Docker Compose.

## Prerequisites

### System Requirements
- Docker Engine 20.10 or later
- Docker Compose v2.0 or later
- At least 4GB RAM (8GB recommended)
- 10GB free disk space for models and data

### GPU Support (Optional)
For GPU acceleration, you need:
- NVIDIA GPU with compute capability 3.5 or higher
- NVIDIA Docker runtime
- Compatible GPU drivers

## Installation Steps

### 1. Install Docker

#### Linux (Ubuntu/Debian)
```bash
# Update package index
sudo apt-get update

# Install Docker
sudo apt-get install -y docker.io docker-compose-plugin

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group (logout/login required)
sudo usermod -aG docker $USER
```

#### Other Linux Distributions
Follow the official Docker installation guide for your distribution:
https://docs.docker.com/engine/install/

#### Windows
Download and install Docker Desktop from:
https://docs.docker.com/desktop/windows/install/

#### macOS
Download and install Docker Desktop from:
https://docs.docker.com/desktop/mac/install/

### 2. Install NVIDIA Docker (For GPU Support)

#### Linux
```bash
# Add NVIDIA Docker repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker
sudo systemctl restart docker
```

#### Windows/macOS
GPU support in Docker Desktop requires:
- Docker Desktop with WSL2 backend (Windows)
- NVIDIA Docker toolkit properly configured

## Running Applio

### 1. Clone Repository
```bash
git clone https://github.com/IAHispano/Applio.git
cd Applio
```

### 2. Configure Environment
```bash
# Copy example environment file
cp .env.example .env

# Edit configuration (optional)
nano .env  # or your preferred editor
```

### 3. Start Services

#### CPU-only Version
```bash
docker compose up -d
```

#### GPU-enabled Version
```bash
docker compose --profile gpu up -d
```

### 4. Access Application
Open your browser and navigate to:
```
http://localhost:6969
```

## Configuration Options

### Environment Variables (.env file)

| Variable | Default | Description |
|----------|---------|-------------|
| `APPLIO_HOST` | `0.0.0.0` | Host interface to bind to |
| `APPLIO_PORT` | `6969` | Port number for the web interface |
| `APPLIO_SHARE` | `false` | Enable public sharing via Gradio |
| `APPLIO_DEBUG` | `false` | Enable debug mode |
| `CUDA_VISIBLE_DEVICES` | `` | GPU devices to use (empty for CPU) |

### Volume Mappings

The Docker setup creates persistent volumes for:
- **applio_logs**: Application logs and training logs
- **applio_models**: Downloaded and custom models
- **applio_audio**: Audio files and samples
- **applio_pretrained**: Pre-trained model files
- **applio_weights**: Model weights and checkpoints
- **applio_cache**: Application cache
- **applio_config**: Configuration files

## Portainer Deployment

### Using Portainer Stacks

1. Access your Portainer instance
2. Navigate to "Stacks" → "Add Stack"
3. Choose "Repository" option
4. Enter repository URL: `https://github.com/IAHispano/Applio.git`
5. Set compose file path: `docker-compose.yaml`
6. Configure environment variables:
   - `APPLIO_PORT`: Your desired port (default: 6969)
   - `APPLIO_HOST`: Leave as 0.0.0.0
   - Other variables as needed
7. Deploy the stack

### Using Portainer App Templates

If you prefer using app templates, you can create a custom template with the `portainer-app-template.json` file included in this repository:

1. In Portainer, go to "Settings" → "App Templates"
2. Add a new template URL pointing to the raw file:
   ```
   https://raw.githubusercontent.com/IAHispano/Applio/main/portainer-app-template.json
   ```
3. Or manually import the template JSON content
4. The template provides both CPU and GPU variants of Applio

Alternatively, you can use the following template configuration:

```json
{
  "type": 2,
  "title": "Applio",
  "description": "AI Voice Conversion Tool",
  "categories": ["AI", "Audio"],
  "platform": "linux",
  "repository": {
    "url": "https://github.com/IAHispano/Applio",
    "stackfile": "docker-compose.yaml"
  },
  "env": [
    {
      "name": "APPLIO_PORT",
      "label": "Port",
      "default": "6969"
    },
    {
      "name": "APPLIO_HOST",
      "label": "Host",
      "default": "0.0.0.0"
    }
  ]
}
```

## Management Commands

### Useful Docker Commands

```bash
# View running containers
docker compose ps

# View logs
docker compose logs -f applio

# Stop services
docker compose down

# Restart services
docker compose restart

# Update and rebuild
docker compose down
docker compose pull
docker compose build --no-cache
docker compose up -d

# Access container shell
docker compose exec applio bash

# Remove all data (WARNING: Deletes all models and data!)
docker compose down -v
```

### Backup and Restore

#### Backup Volumes
```bash
# Create backup directory
mkdir -p backups

# Backup volumes
docker run --rm -v applio_models:/data -v $(pwd)/backups:/backup alpine tar czf /backup/applio_models.tar.gz -C /data .
docker run --rm -v applio_logs:/data -v $(pwd)/backups:/backup alpine tar czf /backup/applio_logs.tar.gz -C /data .
```

#### Restore Volumes
```bash
# Restore volumes
docker run --rm -v applio_models:/data -v $(pwd)/backups:/backup alpine tar xzf /backup/applio_models.tar.gz -C /data
docker run --rm -v applio_logs:/data -v $(pwd)/backups:/backup alpine tar xzf /backup/applio_logs.tar.gz -C /data
```

## Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Check what's using the port
sudo netstat -tulnp | grep :6969

# Change port in .env file
echo "APPLIO_PORT=7000" >> .env
```

#### Permission Denied
```bash
# Ensure user is in docker group
sudo usermod -aG docker $USER
# Logout and login again
```

#### GPU Not Detected
```bash
# Test NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Check if GPU profile is used
docker compose --profile gpu up -d
```

#### Container Won't Start
```bash
# Check logs
docker compose logs applio

# Check system resources
docker system df
docker system prune -f
```

#### Volume Issues
```bash
# Check volume status
docker volume ls
docker volume inspect applio_models

# Reset volumes (WARNING: Deletes all data!)
docker compose down -v
docker compose up -d
```

## Security Considerations

1. **Network Security**: By default, Applio binds to all interfaces (0.0.0.0). For production use, consider:
   - Using a reverse proxy (nginx, traefik)
   - Restricting access with firewall rules
   - Using HTTPS with SSL certificates

2. **Data Protection**: Volumes contain sensitive data:
   - Regular backups are recommended
   - Consider encryption for sensitive models
   - Monitor disk usage

3. **Updates**: Keep Docker images updated:
   ```bash
   docker compose pull
   docker compose up -d
   ```

## Support

For issues related to:
- Docker setup: Check this guide and Docker documentation
- Applio functionality: Visit [Applio documentation](https://docs.applio.org)
- Community support: Join the [Discord server](https://discord.gg/urxFjYmYYh)

## Contributing

If you find issues with the Docker setup or have improvements, please:
1. Open an issue on GitHub
2. Submit a pull request with your changes
3. Follow the contribution guidelines in the main repository