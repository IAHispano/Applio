# ---- base: Python RVC engine + Node for the web stack ----
# NOTE: for NVIDIA GPU training/inference, build from a CUDA image instead, e.g.
# `--build-arg BASE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04` (needs Node+Python added).
FROM python:3.10-slim AS base
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    nodejs npm ffmpeg libsndfile1 git curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY package.json package-lock.json* ./
COPY app/api/package.json ./app/api/
COPY app/web/package.json ./app/web/
COPY app/desktop/package.json ./app/desktop/
RUN npm install --workspaces --include-workspace-root
COPY app ./app
RUN npm run build --workspace=@applio/api && npm run build --workspace=@applio/web

# ---- api: Express gateway + Python engine ----
FROM base AS api
EXPOSE 8000
ENV API_PORT=8000 APPLIO_ROOT=/app
CMD ["node", "app/api/dist/index.js"]

# ---- web: Next.js standalone ----
FROM base AS web
EXPOSE 3000
ENV PORT=3000 HOSTNAME=0.0.0.0
CMD ["node", "app/web/.next/standalone/server.js"]
