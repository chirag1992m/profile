/**
 * @type {import('next').NextConfig}
 */
const nextConfig = {
    // Only use static export for production builds
    ...(process.env.NODE_ENV === 'production' && { output: 'export' }),

    // Development-specific settings for better hot reload
    ...(process.env.NODE_ENV === 'development' && {
        // Enable fast refresh
        reactStrictMode: true,
        // Ensure hot reload works properly
        swcMinify: false,
    }),

    // Optional: Change links `/me` -> `/me/` and emit `/me.html` -> `/me/index.html`
    trailingSlash: true,

    // Optional: Prevent automatic `/me` -> `/me/`, instead preserve `href`
    skipTrailingSlashRedirect: true,

    // Optional: Change the output directory `out` -> `dist`
    // distDir: 'dist',
    images: {
        unoptimized: true,
        // Add the external domains your images come from
        remotePatterns: [
            {
                protocol: 'https',
                hostname: 'c.statcounter.com',
            },
        ],
    },
}

module.exports = nextConfig
