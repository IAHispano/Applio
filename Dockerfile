# syntax=docker/dockerfile:1
FROM python:3.10-bullseye

# Expose the required port
EXPOSE 6969

# Set up working directory
WORKDIR /app

# Install system dependencies, clean up cache to keep image size small
RUN apt update && \
    apt install -y -qq ffmpeg && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy application files into the container
COPY . .

# Create a virtual environment in the app directory and install dependencies
RUN python3 -m venv /app/.venv && \
    . /app/.venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir python-ffmpeg && \
    pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121 && \
    if [ -f "requirements.txt" ]; then pip install --no-cache-dir -r requirements.txt; fi

# Define volumes for persistent storage
VOLUME ["/app/logs/"]

# Set environment variables if necessary
ENV PATH="/app/.venv/bin:$PATH"

# Run the app
ENTRYPOINT ["python3"]
CMD ["app.py"]
