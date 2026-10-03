/* The settled invariants from CLAUDE.md (§2, §3, §7), as the page tells them.
   Each is a thing the library will never do, whatever a benchmark says. */
import type { IconName } from '../components/sketch/SketchIcon.astro';

export interface Rule {
    id: string;
    icon: IconName;
    title: string;
    body: string;
    margin: string;
}

export const rules: readonly Rule[] = [
    {
        id: 'core',
        icon: 'lock',
        title: 'Never puts a language model in the core.',
        body: '`core/` takes arrays and a config and returns numbers: *no model, no vector store, no network, no filesystem*. LLM calls happen at index time, for propositions and ambiguous entities, or in your own code.',
        margin: 'Arrays in, numbers out. Every later phase leans on that.',
    },
    {
        id: 'nok',
        icon: 'nopost',
        title: 'Never asks you for k.',
        body: 'The web stops on a *relative energy threshold*. A `k=` argument would quietly bring back the cut-off the method argues against, so there is none; slice `answer.passages` if you want fewer.',
        margin: 'Take five if you like. It will not pretend five is a property of the corpus.',
    },
    {
        id: 'dispute',
        icon: 'scale',
        title: 'Never picks a winner in silence.',
        body: 'A contradiction is *negative charge*: opposing atoms damp each other instead of adding up. When one survives into the result, *both sides enter the context, flagged as disputed*, with a ready-made question for the user.',
        margin: 'Disputed is an answer too.',
    },
    {
        id: 'votes',
        icon: 'web',
        title: 'Never throws a duplicate away.',
        body: 'A near-copy found while the energy spreads loses its edge, and *the idea it repeats gains a vote*, counted per source, never per chunk. Plain MMR would drop it and lose the signal.',
        margin: 'Repetition is support, not content. Not truth either: a vote counts sources.',
    },
    {
        id: 'learned',
        icon: 'book',
        title: 'Never lets what it learned rewrite the graph.',
        body: 'Usage reinforcement lives in a *separate layer you can switch off*. The base graph is never mutated; edges nobody ever used are pruned offline, and merging nodes waits until it can be undone.',
        margin: 'Switch the learned layer off and you get back the graph you built.',
    },
    {
        id: 'deps',
        icon: 'key',
        title: 'Never installs what you did not ask for.',
        body: '`pip install spiyweb` pulls in *nothing else*, and `import spiyweb` loads no numpy, torch or FAISS. Index-time tools come as extras, and CI checks the claim on the built wheel, not on the checkout.',
        margin: 'The wheel job measures the artifact, not the promise.',
    },
];
