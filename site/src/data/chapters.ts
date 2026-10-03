/* The four movements of one query through the web, in the order propagation
   runs them. Each is a chapter on the page: the spider detail that stands for
   it, the edge layers or settings in play, and its line of the canonical trace
   (CLAUDE.md §2.6, unrounded as tests/test_propagate.py pins it). */
import type { ImageId } from './images.generated';

export interface LedgerLine {
    hop: string;
    text: string;
    note?: string;
}

export interface Chapter {
    n: 1 | 2 | 3 | 4;
    slug: string;
    verb: string;
    question: string;
    vignette: ImageId;
    vignetteNote: string;
    plate: ImageId;
    plateCaption: string;
    body: readonly string[];
    /** What the chapter touches: edge layers, config fields. */
    tags: readonly string[];
    tagsLabel: string;
    ledger: readonly LedgerLine[];
}

export const chapters: readonly Chapter[] = [
    {
        n: 1,
        slug: 'seed',
        verb: 'Seed',
        question: 'Where does the question land?',
        vignette: 'snare',
        vignetteNote: 'The only moment cosine similarity decides anything.',
        plate: 'suspended',
        plateCaption:
            '*Epeira diadema suspended by its thread*, Popular Science Monthly, 1896. One thread down, and it has landed.',
        body: [
            'The query embedding goes into the graph as a *seed of energy*, 10.0 of it. It makes first contact with its nearest atoms by cosine, and the seed is split between them in proportion to how close each one is.',
            'From there on, similarity steps aside. The energy travels along *typed edges*: shared entities are the main fuel, structure (same document, same section, the next chunk) and chunk-to-proposition links carry the rest. A layer can be switched off without touching the core.',
        ],
        tagsLabel: 'edge layers',
        tags: ['semantic', 'entity', 'structural', 'derivation', 'learned'],
        ledger: [{ hop: 'HOP 0', text: 'Q = 10.0  →  A 5.625 · C 4.375', note: 'split .9 : .7' }],
    },
    {
        n: 2,
        slug: 'spread',
        verb: 'Spread',
        question: 'How far does one thread carry?',
        vignette: 'garden-web',
        vignetteNote: 'A weak thread fades faster than a strong one. That is the whole rule.',
        plate: 'weiditz',
        plateCaption:
            'After Hans Weiditz, *Minerva beside a spider web*, c. 1532. Every strand pulls on its neighbours.',
        body: [
            'Each lit node keeps 40% of what it holds and forwards 60%. The decay is *multiplicative*, never a fixed subtraction, so a path of weak links dies quickly and a path of strong ones carries.',
            'What it forwards is *divided by edge weight*: one strong neighbour takes most of it, five equal neighbours take about 12% each. When a neighbour turns out to be a near-copy, its edge is cut and its share goes to the others, and the idea it copied gets *a vote instead of a second slot*.',
        ],
        tagsLabel: 'config',
        tags: ['damping 0.60', 'proportional split', 'dedup → vote'],
        ledger: [
            {
                hop: 'HOP 1',
                text: 'A forwards 3.375  →  A′ cut, idea A: 2 votes',
                note: 'duplicate',
            },
            { hop: '', text: 'B 2.250 · D 1.125   C forwards 2.625  →  D 1.750 · E 0.875 ✕' },
        ],
    },
    {
        n: 3,
        slug: 'converge',
        verb: 'Converge',
        question: 'What happens where two weak paths meet?',
        vignette: 'geometric',
        vignetteNote: 'D was never the closest passage. It finishes third.',
        plate: 'vault',
        plateCaption:
            'After Hans Weiditz, *Vault with a spider web*, c. 1532. Every thread runs back to the hub.',
        body: [
            'Energy that arrives by more than one path *adds up*. Neither A nor C alone would have lifted D far, but together they hand it 2.875, and a passage no similarity search ranked near the top ends up in the answer. That is *converging evidence, for free*.',
            'A question with two parts goes in as two *coloured seeds*. Where the colours meet is a *bridge*: the passage a multi-hop answer has to cross, and the one plain top-k is least likely to return.',
        ],
        tagsLabel: 'mechanisms',
        tags: ['additive accumulation', 'coloured seeds', 'bridge detection'],
        ledger: [
            {
                hop: 'HOP 2',
                text: 'D = 1.125 + 1.750 = 2.875  →  F 1.725',
                note: 'converging evidence',
            },
        ],
    },
    {
        n: 4,
        slug: 'stop',
        verb: 'Stop',
        question: 'Who decides when it is enough?',
        vignette: 'argiope',
        vignetteNote: 'Nobody asked for five results. The energy ran out.',
        plate: 'luyken',
        plateCaption:
            'Jan Luyken, *Two men study a spider web*, c. 1700. Looking until there is nothing left to see.',
        body: [
            'The web stops on its own when the next hop would carry less than *15% of the energy injected*. Relative, not absolute, so it behaves the same when a follow-up question starts on warm ground or a profile changes the seed.',
            'There is *no `k` parameter*. `max_nodes` and `max_hop` exist only as overflow guards. The caller picks a profile, `precise`, `explore` or `compare`, and each carries its own damping, floor and seed width; a follow-up question keeps 20–30% of the last turn’s warmth.',
        ],
        tagsLabel: 'profiles',
        tags: ['precise', 'explore (default)', 'compare'],
        ledger: [{ hop: 'HOP 3', text: 'F would forward 1.035 < 1.500', note: 'the web stops' }],
    },
];
