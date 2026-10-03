/* Braille cells as shapes. A browser font draws braille as small, unevenly spaced
   dots (or borrows them from a fallback font), so the welcome-box spider fell
   apart on the page while it stays solid in a terminal. Each U+2800 cell here is
   decoded into its 2×4 dot grid and drawn as squares, one SVG per text line,
   exactly one text cell wide per character. */

// Dot bit → (column, row) inside the 2×4 cell, in Unicode's dot numbering.
const DOTS: readonly (readonly [number, number])[] = [
    [0, 0], // dot 1
    [0, 1], // dot 2
    [0, 2], // dot 3
    [1, 0], // dot 4
    [1, 1], // dot 5
    [1, 2], // dot 6
    [0, 3], // dot 7
    [1, 3], // dot 8
];

/** One line of braille as an inline SVG string, `currentColor` dots. */
export function brailleSvg(line: string): string {
    const cells = Array.from(line);
    const rects: string[] = [];
    cells.forEach((ch, x) => {
        const code = ch.codePointAt(0) ?? 0;
        if (code < 0x2800 || code > 0x28ff) return;
        const bits = code - 0x2800;
        DOTS.forEach(([dx, dy], bit) => {
            if (bits & (1 << bit))
                rects.push(
                    `<rect x="${x * 2 + dx + 0.14}" y="${dy + 0.12}" width="0.72" height="0.76"/>`,
                );
        });
    });
    return `<svg viewBox="0 0 ${cells.length * 2} 4" preserveAspectRatio="none" aria-hidden="true" focusable="false" style="display:block;width:100%;height:100%;fill:currentColor">${rects.join('')}</svg>`;
}

/** A span pinned to its text cells; braille fills the line's full height. */
export function cellStyle(s: { cells?: number; braille?: boolean }): string {
    const base = `display:inline-block;width:${s.cells ?? 0}ch;overflow:hidden;vertical-align:top`;
    return s.braille ? `${base};height:1.5em` : base;
}
