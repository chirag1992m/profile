/**
 * @type {import('next').NextConfig}
 */
const nextConfig = {
    output: 'export',

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
