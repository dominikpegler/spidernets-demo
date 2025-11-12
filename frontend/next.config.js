/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // basePath: '/spidernets',
  async rewrites() { return [{ source: '/calibration', destination: '/html/index.html', }] }
}

module.exports = nextConfig
