/* The ledger: Phase 1 numbers from the README's results table and RESULTS.md,
   1000 questions per run, paired bootstrap intervals. Shown here: the runs where
   the web beats both baselines. */
export interface Measure {
    figure: string;
    reading: string;
    what: string;
    set: string;
    against?: string;
}

export const measures: readonly Measure[] = [
    {
        figure: '.5094',
        reading: 'S@5 on MuSiQue',
        what: 'Ahead of both baselines',
        set: 'MuSiQue, tuning set (seed 42), 1000 questions',
        against: 'iterative .4631 · top-k .3090 · +.046, CI [+.030, +.062]',
    },
    {
        figure: '.5073',
        reading: 'S@5, fresh questions',
        what: 'It holds on a second draw',
        set: 'MuSiQue, confirmation set (seed 123), configuration frozen',
        against: 'iterative .4420 · top-k .3046 · +.065, CI [+.048, +.082]',
    },
    {
        figure: '.7130',
        reading: 'S@5 on 2WikiMultihopQA',
        what: 'It transfers unchanged',
        set: '2WikiMultihopQA, 1000 questions, the MuSiQue configuration as it was',
        against: 'iterative .687 · top-k .468 · +.026, CI [+.016, +.037]',
    },
    {
        figure: '~2.3',
        reading: 'LLM calls per question',
        what: 'Cheaper than the baseline it beats',
        set: 'at query time, every run above',
        against: 'iterative retrieval needs about 4',
    },
];

/* How the figures are scored, in the box beside them. */
export const method = {
    formula: 'S@5 = 0.65 · support recall + 0.35 · Novelty@5',
    lines: [
        'Novelty@5 counts relevant passages the web returns and plain top-k never does.',
        'Paired bootstrap intervals over the same 1000 questions for every method.',
        'The winning configuration is applied unchanged to every dataset after the first.',
    ],
} as const;
