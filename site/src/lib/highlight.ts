/* A small build-time Python highlighter for the page's code samples: comments,
   strings, keywords and called names, as spans the stylesheet colours. Enough
   for a dozen lines of example code; not a parser. */
const KEYWORDS = new Set([
    'import',
    'from',
    'with',
    'as',
    'for',
    'in',
    'if',
    'else',
    'return',
    'def',
    'not',
    'and',
    'or',
    'None',
    'True',
    'False',
]);

const escape = (s: string): string =>
    s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

const TOKEN = /(#[^\n]*)|([rf]?"(?:[^"\\\n]|\\.)*")|\b([A-Za-z_]\w*)\b(\s*\()?/g;

export function highlightPython(code: string): string {
    let out = '';
    let last = 0;
    for (const m of code.matchAll(TOKEN)) {
        const at = m.index ?? 0;
        out += escape(code.slice(last, at));
        const [whole, comment, string, word, call] = m;
        if (comment) out += `<span class="hl-c">${escape(comment)}</span>`;
        else if (string) out += `<span class="hl-s">${escape(string)}</span>`;
        else if (word && KEYWORDS.has(word))
            out += `<span class="hl-k">${word}</span>${escape(call ?? '')}`;
        else if (word && call) out += `<span class="hl-f">${word}</span>${escape(call)}`;
        else out += escape(whole);
        last = at + whole.length;
    }
    return out + escape(code.slice(last));
}

/* Shell lines: the command name and its flags; comments and blanks pass. */
export function highlightShell(code: string): string {
    return code
        .split('\n')
        .map((line) => {
            // A blank line separates; a comment explains. Neither is typed.
            if (line.trim() === '') return '';
            if (line.startsWith('#')) return `<span class="hl-c">${escape(line)}</span>`;
            const [cmd = '', ...rest] = line.split(' ');
            const tail = rest
                .map((w) =>
                    w.startsWith('-')
                        ? `<span class="hl-k">${escape(w)}</span>`
                        : w.startsWith('"')
                          ? `<span class="hl-s">${escape(w)}</span>`
                          : escape(w),
                )
                .join(' ');
            return `<span class="hl-p">$ </span><span class="hl-f">${escape(cmd)}</span>${tail ? ' ' + tail : ''}`;
        })
        .join('\n');
}
