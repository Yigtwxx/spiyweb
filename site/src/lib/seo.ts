/* Whether search engines may index the site. Kept out for now (the owner's call,
   2026-09-28: the page will change before it is meant to be found). Flipping this to
   true lets every page be indexed and puts the sitemap line back in robots.txt; a
   page that passes `noindex` to the layout stays out either way. */
export const INDEXABLE = false;
