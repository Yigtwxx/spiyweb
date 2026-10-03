/* Writes the favicon set from one drawing, scripts/favicon.source.svg: the
   spiyweb mark, a spider drawn as a graph (knees and feet are nodes, legs are
   edges) over a faint orb web, with the energy seed in its abdomen; frost on a
   navy square with rounded, transparent corners (Safari draws a light hairline
   around a fully opaque favicon). The SVG is copied as is; the two PNG sizes
   and a three-size `favicon.ico` are rendered from it. The same drawing lives
   inline in components/Logo.astro. */

import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.dirname(here);
const out = (name) => path.join(root, 'public', name);

const svg = await readFile(path.join(here, 'favicon.source.svg'), 'utf8');

await writeFile(out('favicon.svg'), svg, 'utf8');
// Rasterised at twice the target size, then scaled down: the 64-unit viewBox
// read at the default 72 dpi would be upscaled for the 180 px icon.
const png = (size) =>
    sharp(Buffer.from(svg), { density: Math.ceil((72 * size * 2) / 64) })
        .resize(size, size)
        .png()
        .toBuffer();
await writeFile(out('favicon-32.png'), await png(32));
await writeFile(out('apple-touch-icon.png'), await png(180));

// A PNG-in-ICO container with 16, 32 and 48 px images.
const sizes = [16, 32, 48];
const images = await Promise.all(sizes.map(png));
const header = Buffer.alloc(6 + 16 * sizes.length);
header.writeUInt16LE(0, 0);
header.writeUInt16LE(1, 2);
header.writeUInt16LE(sizes.length, 4);
let offset = header.length;
sizes.forEach((size, i) => {
    const e = 6 + 16 * i;
    header.writeUInt8(size, e);
    header.writeUInt8(size, e + 1);
    header.writeUInt16LE(1, e + 4);
    header.writeUInt16LE(32, e + 6);
    header.writeUInt32LE(images[i].length, e + 8);
    header.writeUInt32LE(offset, e + 12);
    offset += images[i].length;
});
await writeFile(out('favicon.ico'), Buffer.concat([header, ...images]));
console.log('wrote favicon.svg, favicon-32.png, apple-touch-icon.png, favicon.ico');
