/* Marks the nav once the hero has scrolled out from under it. The stylesheet does
   the rest: the wordmark slides to the centre and the links fade. At the top of
   the page nothing is touched. */

const SCROLLED = 'data-scrolled';

export function initNav(): void {
    const nav = document.querySelector<HTMLElement>('[data-nav-bar]');
    const hero = document.querySelector<HTMLElement>('[data-nav-hero]');
    if (!nav || !hero) return;
    if (!('IntersectionObserver' in window)) return;
    // The hero counts as gone once its bottom edge passes the nav's bottom edge.
    const observer = new IntersectionObserver(
        ([entry]) => {
            if (!entry) return;
            nav.toggleAttribute(SCROLLED, !entry.isIntersecting);
        },
        { rootMargin: `-${Math.ceil(nav.offsetHeight)}px 0px 0px 0px`, threshold: 0 },
    );
    observer.observe(hero);
}
