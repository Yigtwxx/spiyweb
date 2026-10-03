/* Captions kept in data files: `*Title*` becomes <cite>Title</cite>.
   Everything else is escaped, so data stays plain text. */
const escape = (s: string): string =>
    s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

export function cite(text: string): string {
    return escape(text).replace(/\*([^*]+)\*/g, '<cite>$1</cite>');
}
