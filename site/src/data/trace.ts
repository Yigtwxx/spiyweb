/* The canonical trace (CLAUDE.md §2.6) as a picture: seed Q = 10.0, damping
   0.60, floor 1.5. Energies are the unrounded values tests/test_propagate.py
   pins; the page never shows a number the test suite would not. */

export type NodeId = 'Q' | 'A' | 'C' | 'Ad' | 'B' | 'D' | 'E' | 'F';

export interface TraceNode {
    id: NodeId;
    label: string;
    x: number;
    y: number;
    /** Energy held from each step on; a later step overrides an earlier one. */
    energy: Partial<Record<Step, number>>;
    /** First step the node is lit at. */
    from: Step;
    kind?: 'seed' | 'duplicate' | 'dies' | 'bridge';
}

export interface TraceEdge {
    a: NodeId;
    b: NodeId;
    weight: number;
    /** Step at which energy runs along it. */
    lit: Step;
    kind?: 'cut' | 'dies';
}

export type Step = 0 | 1 | 2 | 3 | 4;
export const STEPS: readonly Step[] = [0, 1, 2, 3, 4];

export const nodes: readonly TraceNode[] = [
    { id: 'Q', label: 'Q', x: 70, y: 190, energy: { 0: 10 }, from: 0, kind: 'seed' },
    { id: 'A', label: 'A', x: 215, y: 110, energy: { 1: 5.625 }, from: 1 },
    { id: 'C', label: 'C', x: 215, y: 270, energy: { 1: 4.375 }, from: 1 },
    { id: 'Ad', label: 'A′', x: 330, y: 38, energy: {}, from: 2, kind: 'duplicate' },
    { id: 'B', label: 'B', x: 390, y: 95, energy: { 2: 2.25 }, from: 2 },
    {
        id: 'D',
        label: 'D',
        x: 400,
        y: 200,
        energy: { 2: 1.125, 3: 2.875 },
        from: 2,
        kind: 'bridge',
    },
    { id: 'E', label: 'E', x: 370, y: 318, energy: { 2: 0.875 }, from: 2, kind: 'dies' },
    { id: 'F', label: 'F', x: 560, y: 200, energy: { 3: 1.725 }, from: 3 },
];

export const edges: readonly TraceEdge[] = [
    { a: 'Q', b: 'A', weight: 0.9, lit: 1 },
    { a: 'Q', b: 'C', weight: 0.7, lit: 1 },
    { a: 'A', b: 'Ad', weight: 0.95, lit: 2, kind: 'cut' },
    { a: 'A', b: 'B', weight: 0.8, lit: 2 },
    { a: 'A', b: 'D', weight: 0.4, lit: 2 },
    { a: 'C', b: 'D', weight: 0.6, lit: 2 },
    { a: 'C', b: 'E', weight: 0.3, lit: 2, kind: 'dies' },
    { a: 'D', b: 'F', weight: 0.5, lit: 3 },
];

/** What each step says, in the caption under the web (read out as it changes). */
export const captions: Record<Step, string> = {
    0: 'Seed. The question enters the graph as *10.0 units of energy*. Nothing else is lit yet.',
    1: 'Hop 0. First contact, by cosine: the seed is split *.9 : .7* between A and C. This is the only step similarity decides.',
    2: 'Hop 1. A and C forward 60% of what they hold. A′ is a near-copy of A: its edge is cut, its share goes to B and D, and *idea A gets a second vote*. E receives 0.875, under the 1.5 floor, and dies.',
    3: 'Hop 2. D is reached by two paths: *1.125 + 1.750 = 2.875*. Converging evidence. D forwards 1.725 to F.',
    4: 'Hop 3. F would forward 1.035, under the floor, so *the web stops itself*. Nobody set a result count.',
};

/** The final ranking, as `retrieve()` returns it. */
export const ranking: readonly { id: NodeId; energy: number; note?: string }[] = [
    { id: 'A', energy: 5.625, note: '2 votes' },
    { id: 'C', energy: 4.375 },
    { id: 'D', energy: 2.875, note: 'never the most similar' },
    { id: 'B', energy: 2.25 },
    { id: 'F', energy: 1.725 },
];
