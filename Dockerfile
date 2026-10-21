# syntax=docker/dockerfile:1
FROM python:3.12-trixie

COPY --from=ghcr.io/astral-sh/uv:0.13.3 /uv /uvx /bin/

# Expose the required port
EXPOSE 6969

# Set up working directory
WORKDIR /app

# Install system dependencies, clean up cache to keep image size small
RUN apt update && \
    apt install -y -qq ffmpeg && \
    apt install -y -qq libportaudio2 && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Use the image's Python and keep installed files independent of the uv cache
ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# Install locked dependencies first so code changes don't invalidate this layer
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# Copy application files into the container
COPY . .

# Define volumes for persistent storage
VOLUME ["/app/logs/"]

# Set environment variables if necessary
ENV PATH="/app/.venv/bin:$PATH"

# Run the app
ENTRYPOINT ["python3"]
CMD ["app.py", "--server-name", "0.0.0.0", "--port", "6969"]
