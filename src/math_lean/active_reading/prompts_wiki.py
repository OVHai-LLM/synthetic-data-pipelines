"""Prompts for the Wikipedia variant of Active Reading.

Mirror of `prompts.py` but tuned for general-knowledge Wikipedia
articles instead of ProofWiki theorems. Two flavours:

  * `agnostic`  : generic "study this article" strategies
  * `factual`   : task-specific — paper's SimpleWikiQA recipe, asking
                  the model to first imagine downstream trivia/QA
                  questions and then propose strategies that target
                  exactly that kind of content (tail facts, dates,
                  named entities, etc.)

Both prompts are English. The article body is fed verbatim in its
source language, and an explicit `Write the artifact in {language}.`
line is added so the multilingual model produces same-language output.

Strategy count default is 3 (vs 5 for ProofWiki) — for pretraining we
want breadth across many articles, not depth on each one.
"""

from __future__ import annotations

# Re-use the strategy parser and the apply prompts from the main module.
# These are flavour-agnostic.
from .prompts import APPLY_SYSTEM, APPLY_USER, parse_strategies  # noqa: F401

# ─────────────────────────────────────────────────────────────────
# Stage 1 — strategy generation (agnostic + factual)
#
# Both flavours now share a common SYSTEM prompt and a common USER
# preamble (article-first), with the flavour-specific guidance
# appended at the END of the user message. This makes the bytes
# preceding the flavour-specific tail byte-identical across both
# flavours so vLLM's prefix cache shares the article's KV blocks
# between the two Stage-1 calls per article.
# ─────────────────────────────────────────────────────────────────

STRATEGY_SYSTEM_COMMON = (
    "You are a study coach who designs reading plans for Wikipedia "
    "articles. Given a single article (in markdown, possibly with "
    "tables and an infobox), you will propose 3 diverse, document-"
    "specific study strategies tailored to *this* article. The "
    "strategies should help a learner internalise the article's "
    "content, recall named entities and dates, place those facts in "
    "context, and connect them to neighbouring topics."
)

# Backwards-compat aliases — wiki_main may import these names.
STRATEGY_AGNOSTIC_SYSTEM = STRATEGY_SYSTEM_COMMON
STRATEGY_FACTUAL_SYSTEM = STRATEGY_SYSTEM_COMMON

# Common preamble: article first so its tokens land in a fixed prefix
# position across both flavours.
_STRATEGY_USER_PREAMBLE = """<article title="{title}" lang="{language}">
{chunk}
</article>

---

"""

# Format-tail used by both flavours.
_STRATEGY_USER_FORMAT_TAIL = """

Format strictly:
## Strategy 1: <short name>
<one paragraph describing exactly what to do>

## Strategy 2: <short name>
...

(Three strategies. No preamble, no closing remarks.)"""

STRATEGY_AGNOSTIC_USER = _STRATEGY_USER_PREAMBLE + (
    "Consider the article above. Propose 3 diverse, document-specific "
    "study strategies that I can use to learn and remember the content.\n\n"
    "Cover a range of cognitive modes — for example: active recall of "
    "named entities and dates, paraphrase of key paragraphs, building a "
    "timeline of events, mapping the article's outbound topical links, "
    "contrasting the subject with a related one, summarising into a fact "
    "list, working an analogy. Pick whatever genuinely fits this article; "
    "do not force every mode.\n\n"
    "Each strategy must be self-contained enough that someone reading it "
    "(with the article) could carry it out without further instructions."
) + _STRATEGY_USER_FORMAT_TAIL

STRATEGY_FACTUAL_USER = _STRATEGY_USER_PREAMBLE + (
    "Consider the article above. First, imagine a few trivia / short-form "
    "factual questions one could be asked about it (e.g. \"When was X "
    "founded?\", \"Who was X's first leader?\", \"How many Y did X "
    "produce?\"). Do NOT write the questions out — just use them mentally.\n\n"
    "Then propose 3 diverse, document-specific study strategies aimed at "
    "mastering exactly that kind of recall. Strategies might include, for "
    "example: building a Q&A flashcard sheet covering every named entity "
    "in the infobox, drafting a chronological timeline of every dated "
    "event, listing every numerical claim with its source sentence, "
    "mapping each external link / outbound topic to its role in the "
    "article, contrasting tail-fact details against the article's lead "
    "summary. Pick whatever genuinely fits this article."
) + _STRATEGY_USER_FORMAT_TAIL


# ─────────────────────────────────────────────────────────────────
# Stage 2 — strategy application (Wikipedia variant)
# ─────────────────────────────────────────────────────────────────

# Same APPLY_SYSTEM as ProofWiki, but with a language-instruction
# suffix appended at runtime so the model produces the artifact in
# the source language rather than drifting into English.
APPLY_LANGUAGE_SYSTEM_SUFFIX = """

The source article is in {language}. Write the entire artifact in {language}, matching the article's register. Preserve all named entities, numbers, and dates verbatim. Use markdown structure (headings, lists, tables) where it helps."""

# Article-first layout: the article body comes BEFORE the strategy so
# that, across the 6 apply calls per article (3 strategies × 2 flavours),
# the bytes preceding the strategy text are byte-identical and vLLM's
# prefix cache shares the article's KV blocks. Saves ~5/6 of article
# prefill per article. Smoke-validated on a 5168-tok uk article: calls
# 2-6 ran 17% faster than the cold call.
APPLY_USER_WIKI = """<article title="{title}" lang="{language}">
{chunk}
</article>

# Strategy
{strategy}

# Output
Apply the strategy above to the article. Write the resulting study artifact directly — no preamble like "Here is the artifact", no closing remarks. Use markdown with section headings as appropriate. Write in {language}."""


# ─────────────────────────────────────────────────────────────────
# Flavour registry (used by wiki_main / pipeline)
# ─────────────────────────────────────────────────────────────────

STRATEGY_FLAVOURS_WIKI = {
    "agnostic": (STRATEGY_AGNOSTIC_SYSTEM, STRATEGY_AGNOSTIC_USER),
    "factual":  (STRATEGY_FACTUAL_SYSTEM,  STRATEGY_FACTUAL_USER),
}
