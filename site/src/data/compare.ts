/* Three ways to fill a context window, side by side: plain top-k, iterative
   retrieval (IRCoT-style query rewriting, the stronger Phase 1 baseline) and
   spiyweb. Each column is a method, not a named product. */

export interface Approach {
    id: 'topk' | 'iter' | 'web';
    name: string;
    kind: string;
}

export interface CompareRow {
    label: string;
    cells: Record<Approach['id'], string>;
}

export const approaches: readonly Approach[] = [
    { id: 'topk', name: 'Plain top-k', kind: 'the k nearest chunks' },
    { id: 'iter', name: 'Iterative retrieval', kind: 'an LLM rewrites, searches again' },
    { id: 'web', name: 'spiyweb', kind: 'a seed of energy, spreading' },
];

export const rows: readonly CompareRow[] = [
    {
        label: 'Starts from',
        cells: {
            topk: 'the chunks most similar to the question',
            iter: 'the same, then a rewritten question, then another',
            web: 'its nearest atoms, then the threads between passages',
        },
    },
    {
        label: 'Evidence that is only connected',
        cells: {
            topk: 'never seen, unless it is also similar',
            iter: 'found if a rewrite happens to name it',
            web: 'reached through shared entities, a hop or two out',
        },
    },
    {
        label: 'Ten near-copies of one passage',
        cells: {
            topk: 'ten slots, one fact',
            iter: 'ten slots, one fact',
            web: 'one slot, and ten votes for it',
        },
    },
    {
        label: 'When it stops',
        cells: {
            topk: 'at k, whatever is left',
            iter: 'after a fixed number of rounds',
            web: 'when the energy runs out',
        },
    },
    {
        label: 'Why this passage?',
        cells: {
            topk: 'a similarity score',
            iter: 'the rewritten query that found it',
            web: 'the activation path that carried energy to it',
        },
    },
    {
        label: 'Saying “I don’t know”',
        cells: {
            topk: 'no signal: k results, always',
            iter: 'no signal',
            web: 'a confidence score, gap warnings and a refusal report',
        },
    },
    {
        label: 'LLM calls per question',
        cells: {
            topk: 'none',
            iter: 'about 4',
            web: 'about 2.3, none of them inside the core',
        },
    },
];
