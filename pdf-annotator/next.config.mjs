/** @type {import('next').NextConfig} */
const nextConfig = {
  webpack: (config) => {
    // pdfjs-dist needs canvas to be aliased to false in Next.js
    config.resolve.alias.canvas = false;
    return config;
  },
};

export default nextConfig;
