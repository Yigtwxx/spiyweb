/* Replays a scripted run inside a <pre>, one frame at a time.
   Under reduced motion, or when asked to `finish`, it renders the final state
   at once. The replay button re-runs it. */
import { frames, type Frame, type Span } from '../data/demo';
import { brailleSvg, cellStyle } from '../lib/braille';

function spanEl(s: Span): HTMLSpanElement {
    const el = document.createElement('span');
    if (s.braille) el.innerHTML = brailleSvg(s.text);
    else el.textContent = s.text;
    if (s.tone) el.dataset.tone = s.tone;
    if (s.cells) el.style.cssText = cellStyle(s);
    return el;
}

function lineEl(spans: Span[], anchor?: string): HTMLDivElement {
    const line = document.createElement('div');
    line.className = 'tui__line';
    // What a margin note beside the run finds this line by (scripts/run-notes.ts).
    if (anchor) line.dataset.anchor = anchor;
    for (const s of spans) line.append(spanEl(s));
    return line;
}

export class TerminalPlayer {
    private out: HTMLElement;
    private lines = new Map<string, HTMLElement>();
    private timer: number | null = null;
    private reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    constructor(root: HTMLElement) {
        const out = root.querySelector<HTMLElement>('[data-tui-out]');
        if (!out) throw new Error('terminal: missing [data-tui-out]');
        this.out = out;
    }

    private place(frame: Frame): HTMLElement {
        const el = lineEl(frame.spans, frame.anchor);
        const prev = frame.id ? this.lines.get(frame.id) : undefined;
        if (prev) prev.replaceWith(el);
        else this.out.append(el);
        if (frame.id) this.lines.set(frame.id, el);
        return el;
    }

    reset(): void {
        if (this.timer) window.clearTimeout(this.timer);
        this.timer = null;
        this.out.replaceChildren();
        this.lines.clear();
    }

    finish(): void {
        this.reset();
        for (const f of frames) this.place(f);
    }

    play(): void {
        this.reset();
        if (this.reduced) {
            this.finish();
            return;
        }
        let i = 0;
        const step = () => {
            const frame = frames[i++];
            if (!frame) {
                this.timer = null;
                return;
            }
            if (frame.type) {
                // Type the last span character by character, the earlier spans at once.
                const head = frame.spans.slice(0, -1);
                const tail = frame.spans[frame.spans.length - 1];
                const line = lineEl(head, frame.anchor);
                const typed = spanEl({ text: '', tone: tail?.tone });
                line.append(typed);
                this.out.append(line);
                if (frame.id) this.lines.set(frame.id, line);
                const text = tail?.text ?? '';
                let c = 0;
                const tick = () => {
                    typed.textContent = text.slice(0, ++c);
                    if (c < text.length)
                        this.timer = window.setTimeout(tick, 28 + Math.random() * 40);
                    else this.timer = window.setTimeout(step, frames[i]?.wait ?? 0);
                };
                this.timer = window.setTimeout(tick, frame.wait);
                return;
            }
            this.place(frame);
            this.timer = window.setTimeout(step, frames[i]?.wait ?? 0);
        };
        this.timer = window.setTimeout(step, frames[0]?.wait ?? 0);
    }
}

export function initTerminals(): void {
    for (const root of document.querySelectorAll<HTMLElement>('[data-tui]')) {
        let player: TerminalPlayer;
        try {
            player = new TerminalPlayer(root);
        } catch {
            continue; // no output pane: the static transcript stays as it is
        }
        const replay = root.querySelector<HTMLButtonElement>('[data-tui-replay]');
        replay?.addEventListener('click', () => player.play());
        if (!('IntersectionObserver' in window)) {
            player.finish();
            continue;
        }
        let started = false;
        const io = new IntersectionObserver(
            (entries) => {
                if (!entries.some((e) => e.isIntersecting) || started) return;
                started = true;
                player.play();
                io.disconnect();
            },
            { threshold: 0.4 },
        );
        io.observe(root);
    }
}
