/* Loads a plate's hover tone the first time a pointer rests on it.

   The second tone is only ever seen under a hovering pointer, so it carries its
   sources in `data-src`/`data-srcset` (Halftone.astro) and gets them here, on the
   first hover; a touch screen never downloads it. `has-hover` turns the
   cross-fade on once it has loaded, so a slow network never fades in an empty
   frame. */

export function initHalftoneHover(): void {
    if (!window.matchMedia('(hover: hover)').matches) return;
    document.addEventListener(
        'pointerover',
        (event) => {
            const frame = (event.target as Element | null)?.closest?.('.halftone--hover');
            if (!frame || frame.classList.contains('has-hover')) return;
            const second = frame.querySelector<HTMLImageElement>('.halftone__img--hover');
            if (!second || second.getAttribute('srcset')) return;
            second.addEventListener('load', () => frame.classList.add('has-hover'), {
                once: true,
            });
            second.srcset = second.dataset.srcset ?? '';
            second.src = second.dataset.src ?? '';
        },
        { passive: true },
    );
}
