/** @type {import('next').NextConfig} */
const apiPort = process.env.DISS_API_PORT || "8642"; // keep in sync with Makefile API_PORT
const nextConfig = {
  async rewrites() {
    return [{ source: "/api/v1/:path*", destination: `http://localhost:${apiPort}/api/v1/:path*` }];
  },
};
export default nextConfig;
