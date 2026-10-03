/* The install line, the CLI surface and the links, as the README shows them. */
export interface Command {
    cmd: string;
    comment: string;
}

export const install: readonly { label: string; cmd: string }[] = [
    { label: 'pip', cmd: 'pip install spiyweb' },
    { label: 'uv', cmd: 'uv tool install spiyweb' },
    { label: 'pipx', cmd: 'pipx install spiyweb' },
];

/* The same verbs without the screen, for scripts and pipes. */
export const cliCommands: readonly Command[] = [
    { cmd: 'pip install "spiyweb[index]"', comment: 'the index-time extras, once' },
    { cmd: 'python -m spacy download en_core_web_sm', comment: 'the entity model' },
    { cmd: 'spiyweb index docs/ my-index', comment: 'a directory of .txt/.md → an index' },
    {
        cmd: 'spiyweb query my-index "what happened afterwards"',
        comment: 'what lit up, as bars',
    },
    { cmd: 'spiyweb query my-index "…" --json', comment: 'the same, machine-readable' },
    { cmd: 'spiyweb lint my-index', comment: 'what is wrong with the corpus' },
    { cmd: 'spiyweb version', comment: 'which extras are missing, and the line to fix it' },
];

/* How the bare command behaves where nobody is there to answer it. */
export const surfaces: readonly { where: string; what: string }[] = [
    { where: 'a terminal', what: 'the live monitor takes the window over' },
    { where: 'a pipe or CI', what: 'the usage text and a non-zero exit, never a hung prompt' },
    { where: 'no braille, no box lines', what: 'SPIYWEB_ASCII=1 draws the plain glyph set' },
    { where: 'Ctrl-C', what: 'exit 130, nothing half-written' },
];

/* The library call the CLI is a face of. */
export const snippet = `import spiyweb

with spiyweb.open_index("my-index") as index:
    answer = index.retrieve("who signed off on the release?")

    for passage in answer.passages:
        print(f"{passage.energy:5.2f}  {passage.votes} votes  {passage.text[:60]}")

    print(answer.confidence)   # total energy, node count, hop depth
    for path in answer.paths():  # how the energy reached each node
        print(path)`;

export const version = '0.2.2';
export const repo = 'https://github.com/Yigtwxx/spiyweb';
export const pypi = 'https://pypi.org/project/spiyweb/';

/* The sibling tools: same family, same page. */
export const siblings: readonly { name: string; href: string; tag: string }[] = [
    {
        name: 'proofpath',
        href: 'https://proofpath-yigtwx.vercel.app',
        tag: 'checks that a citation says what the claim says',
    },
    {
        name: 'reasonhound',
        href: 'https://reasonhound.vercel.app',
        tag: 'a security scanner that follows the scent, not the checklist',
    },
];

/* Using it as a library: three steps from a folder of documents to context for
   whichever model the caller already uses. No index of your own needed. */
export const libraryInstall = `pip install "spiyweb[index]"
python -m spacy download en_core_web_sm`;

export const libraryIndex = `spiyweb index docs/ my-index`;

export const librarySnippet = `import spiyweb

with spiyweb.open_index("my-index") as index:
    answer = index.retrieve("who signed off on the release?")

# The web stopped itself; hand what it found to any LLM.
context = "\\n\\n".join(p.text for p in answer.passages)

for p in answer.passages:
    print(f"{p.energy:5.2f}  {p.votes} votes  {p.source_id}")
print(answer.confidence)  # how sure it is: energy, nodes, hop depth`;
