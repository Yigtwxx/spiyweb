/* The page's entrance choreography, in one place.

   1. Reveal: blocks rise out of a frost blur as they enter the viewport, siblings
      a beat apart. Elements are tagged here by selector rather than in every
      component, so the rhythm is set once for the whole page.
   2. Develop: each halftone plate comes up from a point outward, the way energy
      spreads from a seed (a registered `--reveal` radius on its mask, global.css).
   3. Count: the measured figures run up from zero to their value once.

   All of it is gated: without the script, under reduced motion or without
   IntersectionObserver, every element is simply there. */

const REVEAL_GROUPS: readonly string[] = [
    '.section__head > div > *',
    '.section__head > .section__margin',
    '.chapter__text > *',
    '.chapter__side > *',
    '.honesty__cards > li',
    '.rules__list > li',
    '.rules__plate',
    '.limits__list > li',
    '.limits__aside > *',
    '.measured__list > li',
    '.measured__side > *',
    '.compare__row',
    '.compare__table thead tr',
    '.trace__figure',
    '.trace__side > *',
    '.ci__main > *',
    '.ci__side > *',
    '.run__notes > .run__lede',
    '.footer__col',
];
const STAGGER_MS = 90;
const MAX_STAGGER = 6;

function tagReveals(): HTMLElement[] {
    const tagged: HTMLElement[] = [];
    for (const selector of REVEAL_GROUPS) {
        for (const el of document.querySelectorAll<HTMLElement>(selector)) {
            if (el.hasAttribute('data-reveal')) continue;
            // A beat per sibling, capped, so a long list does not keep the reader waiting.
            const index = el.parentElement ? [...el.parentElement.children].indexOf(el) : 0;
            el.setAttribute('data-reveal', '');
            el.style.setProperty(
                '--reveal-delay',
                `${Math.min(index, MAX_STAGGER) * STAGGER_MS}ms`,
            );
            tagged.push(el);
        }
    }
    return tagged;
}

/** ".5094" → 0.5094 with 4 decimals and no leading zero; "~2.3" keeps its tilde. */
function parseFigure(
    text: string,
): { prefix: string; value: number; decimals: number; bare: boolean } | null {
    const match = /^([^\d.]*)(\d*)(?:\.(\d+))?$/.exec(text.trim());
    if (!match) return null;
    const [, prefix = '', whole = '', frac = ''] = match;
    if (!whole && !frac) return null;
    return {
        prefix,
        value: Number(`${whole || '0'}.${frac || '0'}`),
        decimals: frac.length,
        bare: whole === '',
    };
}

function countUp(el: HTMLElement): void {
    const original = el.textContent ?? '';
    const figure = parseFigure(original);
    if (!figure) return;
    const duration = 1500;
    const start = performance.now();
    const step = (now: number): void => {
        const t = Math.min(1, (now - start) / duration);
        const eased = 1 - Math.pow(2, -10 * t); // ease-out-expo: fast, then it settles
        let text = (figure.value * (t === 1 ? 1 : eased)).toFixed(figure.decimals);
        if (figure.bare) text = text.replace(/^0(?=\.)/, '');
        el.textContent = figure.prefix + text;
        if (t < 1) requestAnimationFrame(step);
        else el.textContent = original;
    };
    requestAnimationFrame(step);
}

export function initReveal(): void {
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduced || !('IntersectionObserver' in window)) return;
    document.documentElement.classList.add('motion');

    const io = new IntersectionObserver(
        (entries) => {
            for (const entry of entries) {
                if (!entry.isIntersecting) continue;
                const el = entry.target as HTMLElement;
                el.classList.add('is-in');
                if (el.matches('[data-count]')) countUp(el);
                io.unobserve(el);
            }
        },
        { rootMargin: '0px 0px -8% 0px', threshold: 0.12 },
    );

    for (const el of tagReveals()) io.observe(el);
    for (const el of document.querySelectorAll<HTMLElement>('.halftone')) io.observe(el);
    for (const el of document.querySelectorAll<HTMLElement>('.measured__figure b')) {
        el.setAttribute('data-count', '');
        io.observe(el);
    }
}
