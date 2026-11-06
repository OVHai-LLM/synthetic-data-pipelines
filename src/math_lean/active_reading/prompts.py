"""Prompts for the two-stage Active Reading pipeline.

Stage 1 — strategy generation (two flavours, used in parallel):
  * `STRATEGY_AGNOSTIC`  : generic study strategies for *this* document
  * `STRATEGY_LEAN`      : task-specific (Lean formalisation) strategies

Stage 2 — strategy application:
  * `APPLY_STRATEGY`     : apply one strategy to the source document and
                           emit a study artifact

Conventions follow Lin et al. 2025: each strategy is a markdown block
prefixed by `## ` so we can split them out reliably with a regex.
"""

from __future__ import annotations

import re

# ─────────────────────────────────────────────────────────────────
# Stage 1a — task-agnostic strategy generation
# ─────────────────────────────────────────────────────────────────

STRATEGY_AGNOSTIC_SYSTEM = (
    "You are a mathematics tutor who designs study plans for advanced "
    "self-learners. Given a single source document (a theorem, definition, "
    "axiom or corollary from ProofWiki, in raw wikitext with embedded "
    "LaTeX), you will propose a diverse set of concrete study strategies "
    "that are tailored to *this* document. The strategies should help a "
    "learner internalise the content so they can reproduce it from memory, "
    "explain it, manipulate it, and connect it to neighbouring results."
)

STRATEGY_AGNOSTIC_USER = """Consider the following ProofWiki document. Propose 5 diverse, document-specific study strategies that I can use to learn and remember everything in it.

Cover a range of cognitive modes — for example: active recall, paraphrase / retell, working a concrete example, varying or weakening a hypothesis, mapping the proof's lemma dependencies, comparing alternative proof routes, contrasting with neighbouring theorems, building a counter-example, formalising a key step. Pick whatever fits the document; do not force every mode.

Each strategy must be self-contained enough that someone reading it (with the source document) could carry it out without further instructions.

Format strictly:
## Strategy 1: <short name>
<one paragraph describing exactly what to do>

## Strategy 2: <short name>
...

(Five strategies. No preamble, no closing remarks.)

<document title="{title}" kind="{kind}">
{chunk}
</document>"""


# ─────────────────────────────────────────────────────────────────
# Stage 1b — task-specific (Lean formalisation) strategy generation
# ─────────────────────────────────────────────────────────────────

STRATEGY_LEAN_SYSTEM = (
    "You are training a model whose downstream task is to take an "
    "informal mathematical statement (with a prose proof) and produce "
    "a Lean 4 / mathlib4 formalisation: a `theorem` declaration with "
    "a tactic-mode proof. You design study strategies that prepare a "
    "learner — and indirectly, the model trained on the resulting "
    "data — for that specific task."
)

STRATEGY_LEAN_USER = """Consider the following ProofWiki document. First, imagine a few Lean-formalisation exam questions one could be asked about it (e.g. \"state this in Lean 4 using mathlib\", \"prove this lemma by induction in tactic mode\", \"identify which mathlib lemma corresponds to step X\"). Do NOT write the questions out — just use them mentally.

Then propose 5 diverse, document-specific study strategies aimed at mastering exactly that kind of task. Strategies might include, for example: drafting a Lean 4 theorem signature with appropriate types and typeclasses, identifying the mathlib lemma for each natural-language proof step, restating the proof goal-first using `apply` / `refine`, decomposing the proof into named sub-lemmas, predicting which tactics (`induction`, `linarith`, `ring`, `rw`, `simp`, `omega`, …) each step would use, building a minimal `import` list. Pick whatever genuinely fits this document.

Format strictly:
## Strategy 1: <short name>
<one paragraph describing exactly what to do; reference Lean / mathlib idioms by name where relevant>

## Strategy 2: <short name>
...

(Five strategies. No preamble, no closing remarks.)

<document title="{title}" kind="{kind}">
{chunk}
</document>"""


# ─────────────────────────────────────────────────────────────────
# Stage 2 — strategy application
# ─────────────────────────────────────────────────────────────────

APPLY_SYSTEM = (
    "You are a mathematics learner executing a study strategy. "
    "Given a source document and one strategy, you produce a self-study "
    "artifact: the concrete written output of carrying that strategy "
    "out on this specific document. Be substantive and detailed; this "
    "artifact will be used as training data, so it must stand on its "
    "own and contain real mathematical content drawn from the source. "
    "Preserve LaTeX in `$...$` form. Do not invent facts beyond what "
    "the source plus standard background mathematics support."
)

# Appended to APPLY_SYSTEM only when flavour == "lean". Concrete house
# rules for the Lean snippets the model writes — derived from the
# failure modes we observed in the v1/v2 lean_check runs.
APPLY_LEAN_SYSTEM_SUFFIX = """

When this artifact contains Lean 4 code, follow these house rules:
* Toolchain: Lean 4 + mathlib4 master (post-mathlib3 rename).
* Wrap every Lean snippet in a ```lean ... ``` fence.
* Imports: write `import Mathlib` at the top of any snippet that needs
  mathlib. Do NOT write sub-imports like `import Analysis.X` — anything
  not starting with `Mathlib.` will be stripped before compilation.
* Names are mathlib4-cased: `Nat.succ`, `Real.log`, `MeasurableSpace`,
  `IsProbabilityMeasure` — never the snake_case mathlib3 forms
  (`nat.succ`, `real.log`, `measurable_space`).
* Common tactics you can rely on: `rfl`, `simp`, `simp_all`, `linarith`,
  `nlinarith`, `omega`, `ring`, `ring_nf`, `field_simp`, `norm_num`,
  `decide`, `tauto`, `exact_mod_cast`, `push_neg`. For lemma lookup
  use `exact?` / `apply?` mentally — but in the snippet itself, write
  the actual lemma name you believe exists.
* `sorry` is acceptable only when the strategy is signature-only
  (typeclass drafting, signature design) or when filling the proof
  is genuinely out of scope. Never use `sorry` to skip an error.
* Don't redeclare names already in mathlib (will shadow and confuse)."""

APPLY_USER = """Apply the following study strategy to the document below. Write the resulting study artifact directly — no preamble like "Here is the artifact", no closing remarks. Use markdown with section headings as appropriate.

# Strategy
{strategy}

# Source document
<document title="{title}" kind="{kind}">
{chunk}
</document>

# Output
Write the study artifact now."""


# ─────────────────────────────────────────────────────────────────
# Strategy parsing
# ─────────────────────────────────────────────────────────────────

# Match `## Strategy N: name\n<body until next `## ` or EOF>`
_STRAT_RE = re.compile(r"^##\s+(?:Strategy\s+\d+\s*:\s*)?(.+?)\n(.*?)(?=^##\s|\Z)",
                       re.MULTILINE | re.DOTALL)


def parse_strategies(text: str) -> list[dict]:
    """Split a strategy-generation response into individual strategies.

    Returns a list of {"name": str, "body": str, "raw": str} where `raw`
    is the full `## ...` block (suitable to feed back into APPLY_USER).
    """
    out = []
    for m in _STRAT_RE.finditer(text):
        name = m.group(1).strip()
        body = m.group(2).strip()
        if not body:
            continue
        out.append({
            "name": name,
            "body": body,
            "raw": m.group(0).strip(),
        })
    return out