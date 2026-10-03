"""Prompt templates, kept apart from pipeline logic.

Templates are plain `str.format` strings. They ask for machine-parseable plain
text (one item per line), never JSON - parsing stays trivial and `eval`-free.
"""

from __future__ import annotations

ENTITY_EXTRACTION_PROMPT = """\
Extract the named entities (people, organizations, places, products, events,
works, laws) mentioned in the following text.

Rules:
- Output ONE entity per line.
- Output ONLY the entity names - no commentary, no numbering, no explanations.
- If the text mentions no entities, output nothing.

Text:
{text}
"""

PROPOSITION_EXTRACTION_PROMPT = """\
Extract the atomic factual propositions stated in the following text.

Rules:
- Output ONE proposition per line.
- Each proposition is a single, self-contained factual sentence.
- Resolve pronouns: name the actual person, place or thing they refer to.
- Do not add facts the text does not state. No commentary, no numbering.
- If the text states no facts, output nothing.

Text:
{text}
"""

# Tightened 2026-08-16 after the first label audit (open question #11). The
# first version produced 142 tags over 26.058 propositions and only 46.5% of
# them held up: 76 carried no negation at all, and some INVENTED a denial the
# source never made ("The series was not canceled on January 10, 2012." for a
# passage saying the series PREMIERED that day). Recall was the other half of
# the failure - 532 untagged propositions carried a negation cue, roughly 8x
# the tagged set. A negative-polarity atom DESTROYS energy, so a wrong tag is
# worse than no tag; hence the two explicit prohibitions plus the implicit-
# negation examples that go after the misses. The measurement that rejected
# the first version is in `memory/open-questions.md` #11.
PROPOSITION_EXTRACTION_POLARITY_PROMPT = """\
Extract the atomic factual propositions stated in the following text.

Rules:
- Output ONE proposition per line.
- Each proposition is a single, self-contained factual sentence.
- Resolve pronouns: name the actual person, place or thing they refer to.
- If a proposition states that something is NOT the case, start that line with
  exactly "NEG: " and then the proposition. This covers explicit denials
  ("did not", "never", "no longer", "was not") and implicit ones ("failed to
  win", "remains unfinished", "without a successor", "the plan was abandoned").
- Only tag what the text itself denies. Never turn a positive statement into a
  denial of something else: if the text says an event happened in 1917, do NOT
  write that it did not happen in 1918. A plain positive fact never gets the
  prefix.
- If you cannot point to the words in the text that do the denying, the line
  is not a NEG line.
- Do not add facts the text does not state. No commentary, no numbering.
- If the text states no facts, output nothing.

Text:
{text}
"""

QUERY_DECOMPOSITION_PROMPT = """\
Split the question into the SMALLEST number of search queries that together
cover every fact it needs. Most questions need exactly 2; use 3 or 4 ONLY if
the question truly chains that many separate facts. Follow the examples.

Question: Who is the spouse of the performer of the song Green?
Queries:
song Green performer
performer of the song Green spouse

Question: In which country is the city where Air China's headquarters is located?
Queries:
Air China headquarters city
country of the city of Air China headquarters

Question: Who founded the company that owns the studio that released the film Avatar?
Queries:
film Avatar studio
studio that released Avatar owner company
founder of the company that owns the Avatar studio

Question: Which film has the director who was born earlier, Silver Harvest or The Glass Orchard?
Queries:
film Silver Harvest director
director of Silver Harvest birth date
film The Glass Orchard director
director of The Glass Orchard birth date

Rules:
- One search query per line, nothing else.
- Each query is self-contained keywords naming the concrete entities,
  works, places or relations it asks about.
- A comparison between facts DERIVED from two entities (their directors'
  birth dates, their countries' populations, ...) needs the full chain for
  EACH side - one query per fact, usually 4 in total.
- Do not number the lines. Do not answer the question.

Question: {question}
Queries:
"""

INTERMEDIATE_ANSWER_PROMPT = """\
Read the passage and answer the question with ONLY the exact name or phrase
it asks for - a few words, no sentence, no explanation. If the passage does
not answer it, reply exactly: NONE

Passage: {title}
{text}

Question: {subquestion}
Answer:
"""

QUERY_REWRITE_PROMPT = """\
You answer a multi-hop question step by step. You have already retrieved the
paragraphs below and written the reasoning steps so far. Write the SINGLE next
reasoning sentence. Name the fact you still need to look up, or - if the
collected paragraphs already suffice - state the conclusion in the form
"... answer is ...".

Rules:
- Output exactly ONE sentence.
- No commentary, no numbering, no restating the question.

Question:
{question}

Retrieved paragraphs:
{paragraphs}

Reasoning so far:
{reasoning}
"""

READER_PROMPT = """\
Answer the question using the passages below. Reply with ONLY the answer -
the exact name, date, number or short phrase, or "yes"/"no" - with no
sentence and no explanation. If the passages do not contain the answer, give
your single best guess.

{passages}

Question: {question}
Answer:
"""
