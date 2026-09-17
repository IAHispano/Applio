import type { NextConfig } from "next";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  // Single-folder deploy for Electron resources + Docker + Colab.
  // The tracing root keeps the flat classic layout in monorepos.
  output: "standalone",
  outputFileTracingRoot: import.meta.dirname,
  reactStrictMode: true,
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/api/:path*` }];
  },
};

export default nextConfig;
