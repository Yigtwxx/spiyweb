/* Pins each margin note beside the run to the terminal lines it talks about.

   Each note owns a span of lines: `data-anchor="name"` opens it and
   `data-anchor="name-end"` closes it (data/demo.ts). The run is typed out line by
   line, so a note stays hidden until its first line exists, then fades in with its
   arrow somewhere on its span. Notes are taller than the lines they explain, so
   they are laid out like a small constraint problem: first downwards, each as high
   on its span as the note above allows; then upwards, lifting a note (never above
   its own span) so the one below can stay on its span. Only in two columns: in
   one, the notes sit under the terminal, where there is nothing beside them to
   point at. Without the script the notes keep their plain stacked layout. */

// Matches the two-column breakpoint of `.run` (pages/index.astro).
const WIDE = '(min-width: 80rem)';
// The curve-left arrow's head sits at y=14 of its 64-unit box (SketchArrow).
const ARROW_TIP = 14 / 64;
// Space kept between two notes.
const NOTE_GAP = 8;

interface Placed {
    note: HTMLElement;
    first: number; // highest allowed top (arrow on the span's first line)
    last: number; // lowest allowed top (arrow on the span's last line)
    height: number;
    top: number;
}

function middle(line: HTMLElement, origin: number): number {
    const box = line.getBoundingClientRect();
    return box.top - origin + box.height / 2;
}

export function initRunNotes(): void {
    const run = document.querySelector<HTMLElement>('[data-run]');
    const out = run?.querySelector<HTMLElement>('[data-tui-out]');
    const column = run?.querySelector<HTMLElement>('[data-run-notes]');
    if (!run || !out || !column) return;
    const notes = [...column.querySelectorAll<HTMLElement>('[data-note-for]')];
    const wide = window.matchMedia(WIDE);
    let pending = 0;

    const layout = (): void => {
        pending = 0;
        run.classList.toggle('run--anchored', wide.matches);
        if (!wide.matches) return;
        const origin = column.getBoundingClientRect().top;
        // The run's opening note stays in flow at the top of the column; the pinned
        // notes start below it.
        const lede = column.querySelector<HTMLElement>('[data-run-lede]');
        const floor = lede ? lede.getBoundingClientRect().bottom - origin + NOTE_GAP : 0;
        const placed: Placed[] = [];
        for (const note of notes) {
            const name = note.dataset.noteFor;
            const start = out.querySelector<HTMLElement>(`[data-anchor="${name}"]`);
            note.classList.toggle('is-shown', start !== null);
            if (!start) continue;
            const end = out.querySelector<HTMLElement>(`[data-anchor="${name}-end"]`) ?? start;
            const arrow = note.querySelector('svg')?.getBoundingClientRect().height ?? 0;
            const lift = arrow * ARROW_TIP;
            const first = Math.max(middle(start, origin) - lift, floor);
            placed.push({
                note,
                first,
                last: Math.max(first, middle(end, origin) - lift),
                height: note.offsetHeight,
                top: first,
            });
        }
        // Downwards: below the note above, but no lower than the span allows.
        for (let i = 1; i < placed.length; i++) {
            const above = placed[i - 1];
            const item = placed[i];
            if (!above || !item) continue;
            item.top = Math.min(
                Math.max(item.first, above.top + above.height + NOTE_GAP),
                item.last,
            );
        }
        // Upwards: lift a note that still overlaps the one below, within its own span.
        for (let i = placed.length - 2; i >= 0; i--) {
            const item = placed[i];
            const below = placed[i + 1];
            if (!item || !below) continue;
            const room = below.top - NOTE_GAP - item.height;
            if (item.top > room) item.top = Math.max(item.first, room);
        }
        for (const item of placed) {
            item.note.style.setProperty('--note-top', `${Math.round(item.top)}px`);
        }
    };
    const schedule = (): void => {
        if (!pending) pending = requestAnimationFrame(layout);
    };

    // A line is added, replaced (a stage row finishing) or cleared (replay).
    new MutationObserver(schedule).observe(out, { childList: true });
    new ResizeObserver(schedule).observe(run);
    wide.addEventListener('change', schedule);
    schedule();
}
