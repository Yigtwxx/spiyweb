// @ts-check
import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';

// `site` is required by the sitemap integration and for canonical / OG URLs: the
// Vercel production address. `base` is the sub-path the page lives at; if the page
// ever moves under a hub site, set both together and every asset path follows
// (see src/lib/base.ts).
export default defineConfig({
    site: 'https://spiyweb.vercel.app',
    base: '/',
    output: 'static',
    integrations: [sitemap({ filter: (page) => !/\/social\/?$/.test(page) })],
    // 'auto', not 'always': inlining both sheets tripled the HTML and cost the
    // phone ~0.4 s of first paint (Lighthouse on production), more than the two
    // cached, parallel stylesheet requests.
    build: { inlineStylesheets: 'auto' },
});
