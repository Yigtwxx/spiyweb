/* Where the web thins: the known risks from CLAUDE.md §8, as the page states
   them. Qualitative on purpose; the numbers live in RESULTS.md. */
export interface Limit {
    title: string;
    body: string;
}

export const limits: readonly Limit[] = [
    {
        title: 'Spreading activation is not new.',
        body: 'HippoRAG, GraphRAG, LightRAG and RAPTOR already walk graphs. The claim here is narrower: *duplicates turned into votes*, coloured bridges, and the honesty outputs. Prior art for the vote mechanism is welcome.',
    },
    {
        title: 'Busy nodes get a thin share.',
        body: 'A proportional split punishes a passage connected to everything: its energy is ground into many small pieces. Softening it, weighting by `sim ** alpha`, is an open question.',
    },
    {
        title: 'Repetition is not truth.',
        body: 'A vote measures *corpus support*. Counting by source rather than by chunk limits the damage of one document copied ten times, but a corpus that is wrong everywhere still votes for the wrong thing.',
    },
    {
        title: 'Indexing costs model calls.',
        body: 'Proposition extraction and the ambiguous half of entity extraction use an LLM at index time. On a fixed benchmark that is fine; on your corpus, *measure it before you point it at a large one*.',
    },
    {
        title: 'Contradiction detection is early.',
        body: 'The mechanism for disputed sources is in place and tested; the index-time detector that finds them misses most real pairs today. Treat *no dispute flagged* as “none found”, not “none exist”.',
    },
    {
        title: 'The learned layer can drift.',
        body: 'Reinforcement without forgetting pulls the graph toward whatever was asked most. That is why it lives *in its own layer*, off by default, and why consolidation prunes rather than merges.',
    },
];
