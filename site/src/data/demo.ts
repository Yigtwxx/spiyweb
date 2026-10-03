/* A query, replayed: bare `spiyweb` takes the window over, an app in the next
   terminal asks one question, and the monitor plays the web hop by hop. The
   energies are the canonical trace (tests/test_propagate.py) dressed in passage
   names, so the run here and the trace further down the page tell one story. A
   frame either appends a line or, when it names an `id` that already exists,
   replaces that line: that is how a ranking bar grows while the energy spreads.
   The banner above the output is static. */
import { version } from './commands';

export type Tone =
    'ok' | 'warn' | 'bad' | 'dim' | 'accent' | 'plain' | 'blue' | 'hop0' | 'hop1' | 'hop2' | 'hop3';

export interface Span {
    text: string;
    tone?: Tone;
    /** Pin the span to this many text cells: braille has no glyph in the page's
     *  mono font, and a fallback font's width would push the box edge out. */
    cells?: number;
    /** Draw the text as braille dots (lib/braille.ts) instead of font glyphs. */
    braille?: boolean;
}

export interface Frame {
    /** Milliseconds to wait before this frame lands. */
    wait: number;
    /** Lines with the same id replace each other. */
    id?: string;
    /** `type` frames are typed character by character. */
    type?: boolean;
    /** Where a field note beside the run may point: `name` opens its span of lines,
     *  `name-end` closes it (`data-note-for` on the note). */
    anchor?: string;
    spans: Span[];
}

/**
 * The banner: SPIYWEB in block letters (`█` for the letters, `╗╔═╝║╚` for their
 * shadow). The page draws the cells as an SVG rather than as text, because block
 * and box-drawing glyphs leave gaps between lines in a browser font.
 */
export const WORDMARK_BLOCK = '█';
export const wordmark: readonly string[] = [
    '███████╗██████╗ ██╗██╗   ██╗██╗    ██╗███████╗██████╗ ',
    '██╔════╝██╔══██╗██║╚██╗ ██╔╝██║    ██║██╔════╝██╔══██╗',
    '███████╗██████╔╝██║ ╚████╔╝ ██║ █╗ ██║█████╗  ██████╔╝',
    '╚════██║██╔═══╝ ██║  ╚██╔╝  ██║███╗██║██╔══╝  ██╔══██╗',
    '███████║██║     ██║   ██║   ╚███╔███╔╝███████╗██████╔╝',
    '╚══════╝╚═╝     ╚═╝   ╚═╝    ╚══╝╚══╝ ╚══════╝╚═════╝ ',
];
export const WORDMARK_COLUMNS = Math.max(...wordmark.map((line) => line.length));
/** Five gradient bands, left to right: frost-white into glacier blue. */
export const WORDMARK_BANDS = ['#e3f2f9', '#c4e6f4', '#9fd6ec', '#74bde3', '#4f9fd6'] as const;
/** The shadow glyphs and the rule under the banner. */
export const WORDMARK_SHADOW = '#2a527d';

/** The two text lines beside or under the mark. */
export const banner = {
    version: `spiyweb ${version} · live monitor`,
    context: 'listening in ./.spiyweb',
    hint: 'not top-k. the whole web.',
    commands: '? shortcuts · /help',
} as const;

/* The spider from the welcome box (src/spiyweb/pet.py, FULL), drawn on braille. */
const SPIDER: readonly string[] = [
    '⡆   ⡠          ⢄   ⢰',
    '⢷⡀ ⠰⡇          ⢸⠆ ⢀⡾',
    '⠈⠓⠦⢤⣝⣓⣦⣤⡀⣤⣤⢀⣤⣴⣚⣫⡤⠴⠚⠁',
    '   ⣀⡤⢴⣿⢿⣼⣿⣿⣧⡿⣿⡦⢤⣀   ',
    ' ⣴⠋⣡⠖⠋⢰⣿⣿⣿⣿⣿⣿⡆⠙⠲⣌⠙⣦ ',
    ' ⢻⡀⠳⣄ ⠘⢿⣿⣿⣿⣿⡿⠃ ⣠⠞⢀⡟ ',
    ' ⠈⠳ ⠈⠙⠦⣄⡀⠉⠉⢀⣠⠴⠋⠁ ⠞⠁ ',
];
const BOX = 72;
const welcomeText: Span[][] = [
    [],
    [
        { text: 'Welcome to spiyweb', tone: 'plain' },
        { text: ` ${version}`, tone: 'dim' },
        { text: '  live monitor', tone: 'dim' },
    ],
    [],
    [{ text: 'listening in ./.spiyweb', tone: 'dim' }],
    [{ text: '/help for commands', tone: 'dim' }],
    [],
    [],
];
const hopTone = (row: number): Tone =>
    (['hop0', 'hop0', 'hop1', 'hop1', 'hop2', 'hop2', 'hop3'] as const)[row] ?? 'hop3';
const welcome: Frame[] = [
    { wait: 450, spans: [{ text: `╭${'─'.repeat(BOX - 2)}╮`, tone: 'dim' }] },
    ...SPIDER.map((line, row): Frame => {
        const right = welcomeText[row] ?? [];
        const used = right.reduce((n, s) => n + s.text.length, 0);
        return {
            wait: 45,
            anchor: row === 0 ? 'welcome' : undefined,
            spans: [
                { text: '│ ', tone: 'dim' },
                { text: line, tone: hopTone(row), cells: line.length, braille: true },
                { text: '    ' },
                ...right,
                { text: ' '.repeat(Math.max(0, BOX - 4 - line.length - 4 - used)) },
                { text: ' │', tone: 'dim' },
            ],
        };
    }),
    {
        wait: 45,
        anchor: 'welcome-end',
        spans: [{ text: `╰${'─'.repeat(BOX - 2)}╯`, tone: 'dim' }],
    },
];

/* One ranking row: the passage, a bar as long as its energy, the number beside it. */
const FULL = 5.625; // the top energy of the canonical trace
const BAR = 20;
const EIGHTHS = ['', '▏', '▎', '▍', '▌', '▋', '▊', '▉'];
function bar(energy: number): string {
    const cells = (energy / FULL) * BAR;
    const whole = Math.floor(cells);
    return '█'.repeat(whole) + (EIGHTHS[Math.round((cells - whole) * 8)] ?? '');
}
const row = (name: string, text: string, energy: number, tone: Tone, tail: Span[] = []): Span[] => [
    { text: '  ' },
    { text: name.padEnd(14), tone: 'plain' },
    { text: text.padEnd(28), tone: 'dim' },
    { text: bar(energy).padEnd(BAR + 1), tone },
    { text: energy.toFixed(2).padStart(5), tone: 'plain' },
    ...tail,
];
const spinning = (glyph: string, hop: number): Span[] => [
    { text: ` ${glyph} `, tone: 'accent' },
    { text: 'Spreading', tone: 'accent' },
    { text: '... ', tone: 'dim' },
    { text: `(hop ${hop}/3 · any key to skip)`, tone: 'dim' },
];

export const frames: Frame[] = [
    {
        wait: 300,
        id: 'prompt',
        type: true,
        spans: [{ text: '$ ', tone: 'dim' }, { text: 'spiyweb' }],
    },
    ...welcome,
    {
        wait: 300,
        id: 'status',
        spans: [
            { text: ' ○ ', tone: 'dim' },
            { text: 'no app attached yet', tone: 'dim' },
            { text: ' - run your project in another terminal', tone: 'dim' },
        ],
    },
    {
        wait: 1300,
        id: 'status',
        anchor: 'attach',
        spans: [
            { text: ' ● ', tone: 'ok' },
            { text: 'attached', tone: 'ok' },
            { text: ' - waiting for the next query', tone: 'dim' },
        ],
    },
    { wait: 700, spans: [{ text: '' }] },
    {
        wait: 200,
        id: 'query',
        anchor: 'attach-end',
        spans: [
            { text: ' ◆ ', tone: 'warn' },
            { text: 'query caught  ', tone: 'warn' },
            { text: '"who signed off on the release?"', tone: 'plain' },
            { text: '  explore', tone: 'dim' },
        ],
    },
    { wait: 300, spans: [{ text: '─'.repeat(BOX), tone: 'dim' }] },
    // hop 0: first contact, by cosine, and nowhere else
    { wait: 350, id: 'status2', spans: spinning('◐', 0) },
    {
        wait: 250,
        id: 'A',
        anchor: 'seed',
        spans: row('changelog#41', 'v2 rollout approved Tuesday', 5.625, 'hop0', [
            { text: '  seed', tone: 'dim' },
        ]),
    },
    {
        wait: 200,
        id: 'C',
        anchor: 'seed-end',
        spans: row('minutes#07', 'Tuesday review: decisions', 4.375, 'hop0', [
            { text: '  seed', tone: 'dim' },
        ]),
    },
    // hop 1: A and C forward 60%, split by edge weight
    { wait: 900, id: 'status2', spans: spinning('◓', 1) },
    {
        wait: 300,
        id: 'dup',
        anchor: 'vote',
        spans: [
            { text: '  ' },
            { text: 'changelog#41b'.padEnd(14), tone: 'dim' },
            { text: 'near-copy of changelog#41'.padEnd(28), tone: 'dim' },
            { text: '✂ cut  ', tone: 'warn' },
            { text: '+1 vote → changelog#41', tone: 'warn' },
        ],
    },
    {
        wait: 300,
        id: 'B',
        anchor: 'vote-end',
        spans: row('changelog#42', 'rollout checklist, step 4', 2.25, 'hop1'),
    },
    {
        wait: 250,
        id: 'D',
        spans: row('email#113', 'Re: v2 — signed, M. Okafor', 1.125, 'hop1'),
    },
    {
        wait: 250,
        id: 'E',
        spans: row('wiki/ci', 'pipeline stages', 0.875, 'dim', [
            { text: '  < 1.50, dies', tone: 'dim' },
        ]),
    },
    // hop 2: two weak paths converge on D
    { wait: 1000, id: 'status2', spans: spinning('◑', 2) },
    {
        wait: 350,
        id: 'D',
        anchor: 'converge',
        spans: row('email#113', 'Re: v2 — signed, M. Okafor', 2.875, 'hop2', [
            { text: '  ← two paths', tone: 'ok' },
        ]),
    },
    // hop 3: the last forward, then the energy runs out
    { wait: 900, id: 'status2', spans: spinning('◒', 3) },
    {
        wait: 300,
        id: 'F',
        anchor: 'converge-end',
        spans: row('wiki/owners', 'release owners rota', 1.725, 'hop3'),
    },
    {
        wait: 900,
        id: 'status2',
        anchor: 'stop',
        spans: [
            { text: ' ✓ ', tone: 'ok' },
            { text: 'stopped: threshold', tone: 'ok' },
            { text: '  next hop 1.04 < 1.50 · nobody asked for k', tone: 'dim' },
        ],
    },
    { wait: 250, spans: [{ text: '─'.repeat(BOX), tone: 'dim' }] },
    {
        wait: 200,
        spans: [
            { text: ' ledger  ', tone: 'accent' },
            {
                text: 'seed 10.00 · 5 lit · depth 3 · 1 duplicate voted · 0 disputed',
                tone: 'plain',
            },
        ],
    },
    {
        wait: 150,
        anchor: 'stop-end',
        spans: [
            {
                text: ' ranked: changelog#41 · minutes#07 · email#113 · changelog#42 · wiki/owners',
                tone: 'dim',
            },
        ],
    },
];
