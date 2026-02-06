/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: { ignoreBuildErrors: true },
  turbopack: {},
  images: { unoptimized: true },

  // High-performance settings for audio processing
  staticPageGenerationTimeout: 600,

  experimental: {
    serverActions: {
      bodySizeLimit: '100mb', // Large enough for high-fidelity WAV files
    },
  },

  webpack: (config) => {
    config.module.rules.push({
      test: /\.(mp3|wav|m4a)$/,
      use: {
        loader: "file-loader",
        options: {
          publicPath: "/_next/static/audio/",
          outputPath: "static/audio/",
          name: "[name].[ext]",
        },
      },
    });
    return config;
  },
};

module.exports = nextConfig;