/* Draws pencil strokes when they scroll into view.

   Each stroke is hidden with a dash as long as itself and revealed by sliding
   the dash offset to zero. The length has to be measured in *screen* space:
   several primitives stretch their SVG with `preserveAspectRatio="none"` and
   keep the pen width constant with `non-scaling-stroke`, and Chrome then lays
   the dash pattern out in screen pixels, ignoring `pathLength`. So this walks
   each path through its screen CTM and writes the result to `--len`.

   The stylesheet only hides strokes under `html.js` and inside
   `prefers-reduced-motion: no-preference`, so with no script or with reduced
   motion every line is simply there. */

type Stroke = SVGGeometryElement;

function screenLength(el: Stroke): number {
    const ctm = el.getScreenCTM();
    const total = el.getTotalLength();
    if (!ctm || total === 0) return 0;
    const steps = Math.max(8, Math.min(120, Math.ceil(total / 4)));
    let length = 0;
    let prev = el.getPointAtLength(0).matrixTransform(ctm);
    for (let i = 1; i <= steps; i++) {
        const point = el.getPointAtLength((total * i) / steps).matrixTransform(ctm);
        length += Math.hypot(point.x - prev.x, point.y - prev.y);
        prev = point;
    }
    return length;
}

const STROKES = 'path, line, polyline, circle, ellipse';

/** Measure every stroke of `svgs`, then write every length, then commit once.

   Reads first and writes after: interleaving them made the browser recompute
   style once per stroke, and a section with dozens of strokes entering the
   viewport cost a frame or two each time (14-21 ms tasks on a slow scroll).
   The one style read at the end commits the hidden state, so a drawing that
   follows transitions from the measured length -- not from the 4000 fallback,
   which made the dash cycle on its way to 0. */
function measure(svgs: Iterable<Element>): void {
    const lengths: [Stroke, number][] = [];
    for (const svg of svgs) {
        for (const stroke of svg.querySelectorAll<Stroke>(STROKES)) {
            lengths.push([stroke, Math.ceil(screenLength(stroke) * 1.04) + 2]);
        }
    }
    for (const [stroke, len] of lengths) stroke.style.setProperty('--len', String(len));
    const first = lengths[0];
    if (first) void getComputedStyle(first[0]).strokeDashoffset;
}

// Longest `--delay` a stroke is given (the echo line of a box), plus a margin: when
// no `transitionend` arrives (a stroke with nothing left to animate), the drawing
// still settles.
const SETTLE_SLACK_MS = 400;

/** Drop the dash once every stroke has finished drawing (see global.css). */
function settle(svg: Element, drawMs: number): void {
    const strokes = svg.querySelectorAll(STROKES).length;
    let pending = strokes;
    const done = (): void => {
        svg.classList.add('is-settled');
        svg.removeEventListener('transitionend', onEnd);
    };
    const onEnd = (event: Event): void => {
        if ((event as TransitionEvent).propertyName !== 'stroke-dashoffset') return;
        pending -= 1;
        if (pending <= 0) done();
    };
    svg.addEventListener('transitionend', onEnd);
    // A drawing with its own pace says so in `data-draw-ms` (SketchBox), so the
    // fallback never cuts it short -- and no style has to be read to find out.
    const own = Number((svg as SVGElement).dataset.drawMs);
    window.setTimeout(done, (own || drawMs) + SETTLE_SLACK_MS);
}

export function initStrokeDraw(root: ParentNode = document): void {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const targets = Array.from(root.querySelectorAll<SVGElement>('[data-draw]'));
    if (targets.length === 0) return;

    measure(targets);
    // Read once: a style read per drawing would force a style pass per drawing.
    const drawMs =
        parseFloat(
            getComputedStyle(document.documentElement).getPropertyValue('--draw-duration'),
        ) || 900;
    // Once more when the web fonts land, for the strokes not drawn yet: an
    // underline stretches with its word, and a box with the text it frames.
    // Never while scrolling; a stroke that enters the viewport only draws.
    document.fonts?.ready.then(() => {
        measure(targets.filter((svg) => !svg.classList.contains('is-drawn')));
    });

    if (!('IntersectionObserver' in window)) {
        for (const svg of targets) svg.classList.add('is-drawn', 'is-settled');
        return;
    }
    const observer = new IntersectionObserver(
        (entries) => {
            for (const entry of entries) {
                if (!entry.isIntersecting) continue;
                requestAnimationFrame(() => {
                    entry.target.classList.add('is-drawn');
                    settle(entry.target, drawMs);
                });
                observer.unobserve(entry.target);
            }
        },
        { threshold: 0.3, rootMargin: '0px 0px -5% 0px' },
    );
    for (const svg of targets) observer.observe(svg);
}
