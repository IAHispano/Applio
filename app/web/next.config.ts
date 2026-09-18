import type { NextConfig } from "next";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  // Single-folder deploy for Electron resources + Docker + Colab.
  // The tracing root keeps the flat classic layout in monorepos.
  output: "standalone",
  outputFileTracingRoot: import.meta.dirname,
  reactStrictMode: true,
  allowedDevOrigins: ["127.0.0.1", "localhost"],
  experimental: {
    // /api/* is proxied to the Express gateway via rewrites below. Next
    // buffers proxied request bodies (default 10MB) — model/audio uploads
    // (.pth, datasets, samples) are hundreds of MB, so raise the limit to
    // match the API's 1GB multer cap. Without this, uploads are silently
    // truncated to the first 10MB and imports fail.
    middlewareClientMaxBodySize: "1024mb",
  },
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/api/:path*` }];
  },
};

export default nextConfig;
