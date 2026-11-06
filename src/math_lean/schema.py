from dataclasses import asdict, dataclass, field
from typing import Literal


Kind = Literal["theorem", "lemma", "def", "example", "instance"]
AutoformSource = Literal[
    "mathlib-docstring",
    "textbook",
    "synthetic-from-mathlib",
    "synthetic-from-informal",
]
ProofWikiKind = Literal["theorem", "definition", "axiom", "corollary"]


@dataclass
class TheoremRecord:
    id: str
    module: str
    category: str
    kind: Kind
    source_file: str
    source_line: int
    # All text-content fields are nullable: LeanDojo Benchmark 4 gives
    # locators only (file_path + line range + name); the actual statement
    # and proof text live in mathlib4 source at the pinned commit and are
    # backfilled by a separate pass.
    formal_statement: str | None = None
    formal_proof: str | None = None
    informal_statement: str | None = None
    imports: list[str] | None = None
    dependencies: list[str] | None = None
    proof_length_tokens: int | None = None

    def to_jsonl_dict(self) -> dict:
        return asdict(self)


@dataclass
class TacticTraceRecord:
    theorem_id: str
    step_idx: int
    goal_before: str
    tactic: str
    goal_after: str
    n_goals_before: int
    n_goals_after: int
    is_terminal: bool

    def to_jsonl_dict(self) -> dict:
        return asdict(self)


@dataclass
class AutoformPairRecord:
    informal: str
    formal: str
    source: AutoformSource
    theorem_id: str | None = None

    def to_jsonl_dict(self) -> dict:
        return asdict(self)


@dataclass
class Proof:
    name: str
    text: str


@dataclass
class ProofWikiRecord:
    id: str
    title: str
    kind: ProofWikiKind
    source_url: str
    statement: str | None = None
    proofs: list[Proof] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    also_known_as: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    wikilinks: list[str] = field(default_factory=list)

    def to_jsonl_dict(self) -> dict:
        return asdict(self)
