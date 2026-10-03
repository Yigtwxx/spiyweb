/* The output contract (CLAUDE.md §2.5): everything one `retrieve()` returns
   besides the passages. Each card names the accessor exactly as the library
   spells it. */
export interface Output {
    id: string;
    title: string;
    call: string;
    body: string;
}

export const outputs: readonly Output[] = [
    {
        id: 'votes',
        title: 'Votes per idea',
        call: 'answer.votes()',
        body: 'How many *separate sources* repeated each idea before their copies were cut. Corpus support, counted by document.',
    },
    {
        id: 'paths',
        title: 'Activation paths',
        call: 'answer.paths()',
        body: 'How the energy reached every passage, hop by hop, and how many paths converged on it. Hand them to the model *as explanations*, not debug output.',
    },
    {
        id: 'clusters',
        title: 'Theme clusters',
        call: 'answer.clusters()',
        body: 'What lit up, grouped: *these are separate themes*, and this is where they intersect.',
    },
    {
        id: 'confidence',
        title: 'Confidence',
        call: 'answer.confidence',
        body: 'Total energy held, nodes lit, hop depth reached. The library reports; *you decide* what counts as “I don’t know”.',
    },
    {
        id: 'gaps',
        title: 'Corpus-gap warnings',
        call: 'answer.gaps()',
        body: 'Two dense clusters with *no bridge between them*: the corpus is missing the passage that would connect them.',
    },
    {
        id: 'refusal',
        title: 'Refusal report',
        call: 'answer.refusal()',
        body: 'When confidence is low, a template-built account of *why*: which clusters lit, where the bridge is missing, where the energy died. No model writes it.',
    },
    {
        id: 'conflicts',
        title: 'Contradiction records',
        call: 'answer.conflicts',
        body: 'Pairs of sources that oppose each other, *both kept*, with `build_conflict_question()` turning each into a question for the user.',
    },
];
