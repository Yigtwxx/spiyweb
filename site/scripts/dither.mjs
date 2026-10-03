/* Build-time image pipeline.

   Reads `images.manifest.json`, fetches each public-domain source once into
   `.cache/images/`, and writes ordered-dither halftones to `public/images/`:
   three tones (ink, mid, paper) laid out on a Bayer 8x8 grid, so the result is
   the same on every machine and every build. Two colourways come out of each
   source — navy-on-paper for the body of the page, paper-on-navy for the
   hero field. The paper tone
   is transparent: the page background shows through, which is what lets the
   same PNG sit on either field.

   Also emits `src/data/images.generated.ts` so the footer credits and every
   <picture> are derived from the manifest rather than typed by hand. Paths in
   it are relative to the site root; components prefix them with
   `import.meta.env.BASE_URL`, so the site can live under a sub-path such as
   `/spiyweb` on a hub without touching the data. */

import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile, access } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const manifestPath = path.join(root, 'images.manifest.json');
const cacheDir = path.join(root, '.cache', 'images');
const outDir = path.join(root, 'public', 'images');
const generatedPath = path.join(root, 'src', 'data', 'images.generated.ts');

const WIDTHS = [720, 1440];
const USER_AGENT = 'spiyweb-site-build/0.1 (https://github.com/Yigtwxx/spiyweb)';

/** Colourways. Each is a two-level halftone: one tone is painted, the other is
   left transparent so the page's own field shows through. On paper the darks
   are painted navy; on the navy field it is the *lights* that are painted
   paper, so the darks of the engraving become the field itself. */
const TONES = {
    // `gamma` < 1 lifts the mid-tones so an engraving reads as line-work on the
    // paper rather than a solid wash; the navy field wants the opposite.
    paper: { paint: 'dark', rgb: [0x13, 0x29, 0x4b], gamma: 0.6 }, // navy darks on paper
    navy: { paint: 'light', rgb: [0xe8, 0xf2, 0xf9], gamma: 1.1 }, // paper lights on navy
    frost: { paint: 'light', rgb: [0x9f, 0xd6, 0xec], gamma: 1.1 }, // frost lights on navy
    azure: { paint: 'dark', rgb: [0x1a, 0x5d, 0x94], gamma: 0.6 }, // azure darks on paper (hover)
    ink: { paint: 'dark', rgb: [0x0c, 0x1a, 0x2c], gamma: 0.6 }, // ink darks on paper
};

// Bayer 8x8 threshold matrix, normalised to (0, 1).
const BAYER = [
    [0, 32, 8, 40, 2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
].map((row) => row.map((v) => (v + 0.5) / 64));

async function exists(file) {
    try {
        await access(file);
        return true;
    } catch {
        return false;
    }
}

async function fetchWithBackoff(url, attempts = 5) {
    let wait = 3000;
    for (let i = 0; i < attempts; i++) {
        const res = await fetch(url, { headers: { 'User-Agent': USER_AGENT } });
        if (res.ok) return Buffer.from(await res.arrayBuffer());
        if (res.status !== 429 && res.status < 500) throw new Error(`${res.status} for ${url}`);
        await new Promise((r) => setTimeout(r, wait));
        wait *= 2;
    }
    throw new Error(`gave up fetching ${url}`);
}

async function sourceFor(entry) {
    await mkdir(cacheDir, { recursive: true });
    const ext = path.extname(new URL(entry.source_url).pathname) || '.jpg';
    const file = path.join(cacheDir, `${entry.id}${ext}`);
    if (!(await exists(file))) {
        process.stdout.write(`  fetching ${entry.id} … `);
        await writeFile(file, await fetchWithBackoff(entry.source_url));
        process.stdout.write('ok\n');
    }
    return file;
}

/** Greyscale, cropped, contrast-stretched luminance at the target width. */
async function luminance(file, entry, width) {
    let img = sharp(file).rotate();
    // `rotate()` auto-orients by EXIF, but `metadata()` reports the stored,
    // pre-rotation size; orientations 5–8 swap the axes.
    const stored = await img.metadata();
    const swapped = (stored.orientation ?? 1) >= 5;
    const meta = swapped
        ? { width: stored.height, height: stored.width }
        : { width: stored.width, height: stored.height };
    if (entry.crop) {
        const { x, y, w, h } = entry.crop;
        // Clamped: x + w = 1 can round one pixel past the edge.
        const left = Math.round(meta.width * x);
        const top = Math.round(meta.height * y);
        img = img.extract({
            left,
            top,
            width: Math.min(Math.round(meta.width * w), meta.width - left),
            height: Math.min(Math.round(meta.height * h), meta.height - top),
        });
    }
    const contrast = entry.contrast ?? 1.35;
    const { data, info } = await img
        .resize({ width, withoutEnlargement: false, kernel: 'lanczos3' })
        .greyscale()
        .normalise({ lower: 1, upper: 99 })
        .linear(contrast, -(contrast - 1) * 128) // stretch around mid grey
        .raw()
        .toBuffer({ resolveWithObject: true });
    return { data, width: info.width, height: info.height, gamma: entry.gamma ?? 1 };
}

/** Two-level ordered dither into an RGBA buffer. */
function dither({ data, width, height, gamma }, tone) {
    const out = Buffer.alloc(width * height * 4);
    const { paint, rgb } = TONES[tone];
    gamma = gamma * TONES[tone].gamma;
    for (let y = 0; y < height; y++) {
        const row = BAYER[y & 7];
        for (let x = 0; x < width; x++) {
            const i = y * width + x;
            let v = data[i] / 255; // lightness
            if (gamma !== 1) v = Math.pow(v, gamma);
            const coverage = paint === 'dark' ? 1 - v : v;
            const painted = coverage > row[x & 7];
            const o = i * 4;
            if (painted) {
                out[o] = rgb[0];
                out[o + 1] = rgb[1];
                out[o + 2] = rgb[2];
                out[o + 3] = 255;
            } else {
                out[o + 3] = 0;
            }
        }
    }
    return { out, width, height };
}

async function writePng({ out, width, height }, file) {
    await sharp(out, { raw: { width, height, channels: 4 } })
        .png({ palette: true, colours: 2, compressionLevel: 9, effort: 10 })
        .toFile(file);
}

async function main() {
    const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
    await mkdir(outDir, { recursive: true });
    const generated = {};

    for (const entry of manifest.images) {
        console.log(`${entry.id}`);
        const file = await sourceFor(entry);
        const variants = {};
        for (const width of WIDTHS) {
            const lum = await luminance(file, entry, width);
            for (const tone of entry.tones ?? ['paper', 'navy']) {
                const name = `${entry.id}-${width}-${tone}.png`;
                await writePng(dither(lum, tone), path.join(outDir, name));
                variants[tone] ??= {};
                variants[tone][width] = `images/${name}`;
            }
            variants.aspect = lum.width / lum.height;
        }
        generated[entry.id] = {
            alt: entry.alt,
            aspect: variants.aspect,
            variants: Object.fromEntries(Object.entries(variants).filter(([k]) => k !== 'aspect')),
            credit: {
                title: entry.title,
                author: entry.author,
                year: entry.year ?? null,
                licence: entry.licence,
                licence_url: entry.licence_url,
                page_url: entry.page_url,
                source: entry.source ?? 'Wikimedia Commons',
            },
        };
    }

    const hash = createHash('sha1').update(JSON.stringify(generated)).digest('hex').slice(0, 8);
    const ts = [
        '/* Generated by scripts/dither.mjs from images.manifest.json — do not edit. */',
        `export const imagesHash = '${hash}';`,
        '',
        'export type Tone = "paper" | "navy" | "frost" | "azure" | "ink";',
        'export type ImageVariant = Tone;',
        'export interface ImageCredit {',
        '  title: string;',
        '  author: string;',
        '  year: number | null;',
        '  licence: string;',
        '  licence_url: string;',
        '  page_url: string;',
        '  source: string;',
        '}',
        'export interface GeneratedImage {',
        '  alt: string;',
        '  aspect: number;',
        '  variants: Partial<Record<ImageVariant, Record<number, string>>>;',
        '  credit: ImageCredit;',
        '}',
        '',
        `export const images = ${JSON.stringify(generated, null, 2)} as const satisfies Record<string, GeneratedImage>;`,
        '',
        'export type ImageId = keyof typeof images;',
        '',
    ].join('\n');
    await writeFile(generatedPath, ts, 'utf8');
    console.log(
        `wrote ${Object.keys(generated).length} images → ${path.relative(root, generatedPath)}`,
    );
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
