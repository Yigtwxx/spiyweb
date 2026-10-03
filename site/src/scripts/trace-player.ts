/* Steps the canonical trace (components/Trace.astro). The markup ships at the
   last step; here it rewinds to the seed the first time it scrolls into view and
   plays forward once, a hop every STEP_MS. Any control stops the autoplay and
   hands the steps to the reader. Under reduced motion it never autoplays: the
   finished web stays, and the controls still step through it. */
import { captions, type Step } from '../data/trace';

const LAST: Step = 4;
const STEP_MS = 1900;
const HOP_LABELS: Record<Step, string> = {
    0: 'seed · 10.0',
    1: 'hop 0 · first contact',
    2: 'hop 1 · spread',
    3: 'hop 2 · converge',
    4: 'hop 3 · stopped',
};

const plain = (s: string): string => s.replace(/\*/g, '');
const SVG = 'http://www.w3.org/2000/svg';

/** Energy arriving at step `s`: a spark down every thread lit then, and a ring
 *  from every node that lights. Decoration only; the state is the CSS step. */
function sparkle(svg: SVGSVGElement, s: number): void {
    for (const line of svg.querySelectorAll<SVGLineElement>(`.trace__flow[data-lit="${s}"]`)) {
        const [x1, y1, x2, y2] = ['x1', 'y1', 'x2', 'y2'].map((a) => Number(line.getAttribute(a)));
        const dies = line.closest('.trace__edge--dies, .trace__edge--cut') !== null;
        const spark = document.createElementNS(SVG, 'circle');
        spark.setAttribute('r', dies ? '3' : '4.5');
        spark.setAttribute('class', 'trace__spark');
        svg.append(spark);
        spark
            .animate(
                [
                    { transform: `translate(${x1}px, ${y1}px)`, opacity: 1 },
                    { transform: `translate(${x2}px, ${y2}px)`, opacity: dies ? 0.2 : 1 },
                ],
                { duration: 700, easing: 'cubic-bezier(0.3, 0, 0.2, 1)' },
            )
            .finished.then(
                () => spark.remove(),
                () => spark.remove(),
            );
    }
    for (const node of svg.querySelectorAll<SVGGElement>(`.trace__node[data-from="${s}"]`)) {
        const cx = Number(node.dataset.cx);
        const cy = Number(node.dataset.cy);
        const ring = document.createElementNS(SVG, 'circle');
        ring.setAttribute('cx', String(cx));
        ring.setAttribute('cy', String(cy));
        ring.setAttribute('r', node.dataset.r ?? '14');
        ring.setAttribute('class', 'trace__pop');
        ring.style.transformOrigin = `${cx}px ${cy}px`;
        svg.append(ring);
        ring.animate(
            [
                { transform: 'scale(1)', opacity: 0.9 },
                { transform: 'scale(2.2)', opacity: 0 },
            ],
            {
                duration: 900,
                delay: s === 0 ? 0 : 520,
                easing: 'cubic-bezier(0.22, 1, 0.36, 1)',
                fill: 'backwards',
            },
        ).finished.then(
            () => ring.remove(),
            () => ring.remove(),
        );
    }
}

export function initTrace(): void {
    const root = document.querySelector<HTMLElement>('[data-trace]');
    if (!root) return;
    const range = root.querySelector<HTMLInputElement>('[data-trace-range]');
    const hop = root.querySelector<HTMLElement>('[data-trace-hop]');
    const live = root.querySelector<HTMLElement>('[data-trace-live]');
    const prev = root.querySelector<HTMLButtonElement>('[data-trace-prev]');
    const next = root.querySelector<HTMLButtonElement>('[data-trace-next]');
    const play = root.querySelector<HTMLButtonElement>('[data-trace-play]');
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let step: Step = LAST;
    let timer = 0;

    const svg = root.querySelector<SVGSVGElement>('.trace__web');
    const show = (s: Step, announce: boolean): void => {
        if (svg && !reduced && s === step + 1) sparkle(svg, s);
        step = s;
        root.dataset.step = String(s);
        if (range) range.value = String(s);
        if (hop) hop.textContent = HOP_LABELS[s];
        if (prev) prev.disabled = s === 0;
        if (next) next.disabled = s === LAST;
        if (live && announce) live.textContent = plain(captions[s]);
    };
    const stop = (): void => {
        if (timer) window.clearTimeout(timer);
        timer = 0;
    };
    const run = (): void => {
        stop();
        show(0, false);
        const advance = (): void => {
            if (step >= LAST) {
                timer = 0;
                return;
            }
            show((step + 1) as Step, false);
            timer = window.setTimeout(advance, STEP_MS);
        };
        timer = window.setTimeout(advance, STEP_MS * 0.6);
    };

    prev?.addEventListener('click', () => {
        stop();
        show(Math.max(0, step - 1) as Step, true);
    });
    next?.addEventListener('click', () => {
        stop();
        show(Math.min(LAST, step + 1) as Step, true);
    });
    range?.addEventListener('input', () => {
        stop();
        show(Number(range.value) as Step, true);
    });
    play?.addEventListener('click', () => {
        if (reduced) show(0, true);
        else run();
    });

    show(LAST, false);
    if (reduced || !('IntersectionObserver' in window)) return;
    const io = new IntersectionObserver(
        (entries) => {
            if (!entries.some((e) => e.isIntersecting)) return;
            io.disconnect();
            run();
        },
        { threshold: 0.45 },
    );
    io.observe(root);
}
