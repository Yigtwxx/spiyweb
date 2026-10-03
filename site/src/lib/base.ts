/* Prefix a site-root-relative path with the configured `base`, so the same
   components work at `/` and at `/spiyweb` on a hub site. */
const base = import.meta.env.BASE_URL.replace(/\/$/, '');

export function withBase(path: string): string {
    return `${base}/${path.replace(/^\//, '')}`;
}
