/* robots.txt, built from `site` so the sitemap line always names the address the
   page is served at (astro.config.mjs), never a stale one. Crawling stays allowed
   even while the site is kept out of search (lib/seo.ts): a crawler has to read a
   page to see its noindex. The sitemap is only offered once the site is meant to
   be found. */
import type { APIRoute } from 'astro';
import { withBase } from '../lib/base';
import { INDEXABLE } from '../lib/seo';

export const GET: APIRoute = ({ site }) => {
    const sitemap = new URL(withBase('sitemap-index.xml'), site).href;
    const lines = ['User-agent: *', 'Allow: /', ...(INDEXABLE ? ['', `Sitemap: ${sitemap}`] : [])];
    return new Response(`${lines.join('\n')}\n`, {
        headers: { 'Content-Type': 'text/plain; charset=utf-8' },
    });
};
