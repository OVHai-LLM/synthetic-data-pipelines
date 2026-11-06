"""Logic puzzles backed by the Z3 SMT solver.

Each generator builds a fully randomised puzzle, asks Z3 for a model, and
returns either the unique assignment, "Multiple solutions", or "Impossible".
All three are level-0 — output diversity is bounded only by the parameter
spaces, not by curated templates."""

import random

import z3

from .core import StepBuilder

NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi"]


# ─────────────────────────── Knights & Knaves (N people) ───────────────────────────

def _model_assignment(model, knight_vars):
    parts = []
    for name, v in knight_vars.items():
        is_k = bool(model.eval(v, model_completion=True))
        parts.append(f"{name} is a {'knight' if is_k else 'knave'}")
    return "; ".join(parts)


def gen_knights_knaves_z3(dif):
    """N-person knights/knaves with random statements solved by Z3. RANDOMIZED."""
    n = random.choice([2, 3, 3, 4]) if dif != "simple" else 2
    names = random.sample(NAMES, n)
    knight = {nm: z3.Bool(f"k_{nm}") for nm in names}

    statements = []
    statement_text = []

    for nm in names:
        kind = random.choice([
            "self_knight", "self_knave",
            "other_knight", "other_knave",
            "all_knights", "at_least_knave",
            "exactly_one_knight",
        ])
        if kind == "self_knight":
            stmt = knight[nm]
            text = f"{nm} says: 'I am a knight.'"
        elif kind == "self_knave":
            stmt = z3.Not(knight[nm])
            text = f"{nm} says: 'I am a knave.'"
        elif kind == "other_knight":
            other = random.choice([x for x in names if x != nm])
            stmt = knight[other]
            text = f"{nm} says: '{other} is a knight.'"
        elif kind == "other_knave":
            other = random.choice([x for x in names if x != nm])
            stmt = z3.Not(knight[other])
            text = f"{nm} says: '{other} is a knave.'"
        elif kind == "all_knights":
            stmt = z3.And(*knight.values())
            text = f"{nm} says: 'All of us are knights.'"
        elif kind == "at_least_knave":
            stmt = z3.Or(*[z3.Not(v) for v in knight.values()])
            text = f"{nm} says: 'At least one of us is a knave.'"
        else:
            stmt = z3.Sum([z3.If(v, 1, 0) for v in knight.values()]) == 1
            text = f"{nm} says: 'Exactly one of us is a knight.'"
        statements.append(knight[nm] == stmt)
        statement_text.append(text)

    s = z3.Solver()
    for c in statements:
        s.add(c)

    res = s.check()
    if res == z3.unsat:
        ans = "Impossible (no consistent assignment)"
    else:
        m = s.model()
        # uniqueness check
        s.add(z3.Or(*[v != m.eval(v, model_completion=True) for v in knight.values()]))
        if s.check() == z3.sat:
            ans = "Multiple solutions; the puzzle is under-determined."
        else:
            # re-solve to recover full model (s now has the negation constraint)
            s2 = z3.Solver()
            for c in statements: s2.add(c)
            s2.check()
            ans = _model_assignment(s2.model(), knight)

    sb = StepBuilder()
    sb.add(f"For each person X, encode 'X is a knight ↔ what X said'.")
    sb.add("Solve the resulting Boolean system; if no model exists, the puzzle is impossible.")
    if "Impossible" not in ans and "Multiple" not in ans:
        sb.add("Verify uniqueness by negating the found model and re-solving.")
    prob = ("Knights always tell the truth and knaves always lie. "
            + " ".join(statement_text)
            + " Determine each person's type.")
    return prob, ans, {"category": "knights_knaves_z3", "step_text": sb.text()}


# ─────────────────────────── Zebra-style mini puzzle ───────────────────────────

def gen_zebra_z3(dif):
    """Random 3-house puzzle with two attributes (colour and drink). Z3 solves.
    The number of clues is grown until the puzzle is uniquely determined."""
    colors = ["red", "blue", "green"]
    drinks = ["tea", "coffee", "water"]
    random.shuffle(colors)
    random.shuffle(drinks)

    # Build a pool of clues consistent with the random ground truth.
    pool = []
    for i, c in enumerate(colors):
        pool.append((f"The {c} house is at position {i + 1}.",
                     lambda cp, dp, c=c, i=i: cp[c] == i))
    for i, d in enumerate(drinks):
        pool.append((f"The house at position {i + 1} drinks {d}.",
                     lambda cp, dp, d=d, i=i: dp[d] == i))
    for i in range(3):
        pool.append((f"The {colors[i]} house drinks {drinks[i]}.",
                     lambda cp, dp, c=colors[i], d=drinks[i]: cp[c] == dp[d]))
    for i in range(3):
        for j in range(3):
            if i != j and i < j and colors.index(colors[i]) < colors.index(colors[j]):
                pool.append(
                    (f"The {colors[i]} house is to the left of the {colors[j]} house.",
                     lambda cp, dp, ci=colors[i], cj=colors[j]: cp[ci] < cp[cj]))

    def build(clues):
        cp = {c: z3.Int(f"c_{c}") for c in colors}
        dp = {d: z3.Int(f"d_{d}") for d in drinks}
        s = z3.Solver()
        for v in list(cp.values()) + list(dp.values()):
            s.add(v >= 0, v < 3)
        s.add(z3.Distinct(*cp.values()))
        s.add(z3.Distinct(*dp.values()))
        for _, mk in clues:
            s.add(mk(cp, dp))
        return s, cp, dp

    def unique_model(clues):
        s, cp, dp = build(clues)
        if s.check() != z3.sat: return None
        m = s.model()
        diff = []
        for v in list(cp.values()) + list(dp.values()):
            diff.append(v != m.eval(v, model_completion=True))
        s.add(z3.Or(*diff))
        if s.check() == z3.sat:
            return None
        return m, cp, dp

    random.shuffle(pool)
    selected = []
    result = None
    for clue in pool:
        selected.append(clue)
        if len(selected) < 2:
            continue
        result = unique_model(selected)
        if result is not None:
            break

    if result is None:
        return ("3-house puzzle (degenerate)", "indeterminate",
                {"category": "zebra_z3", "step_text": "Step: insufficient clues for uniqueness."})

    m, cp, dp = result
    rows = {i: {} for i in range(3)}
    for c in colors:
        rows[m.eval(cp[c]).as_long()]["color"] = c
    for d in drinks:
        rows[m.eval(dp[d]).as_long()]["drink"] = d
    ans = "; ".join(f"House {i + 1}: {rows[i]['color']}, drinks {rows[i]['drink']}"
                    for i in range(3))

    sb = StepBuilder()
    sb.add("Encode each attribute value as an integer position in [0, 3).")
    sb.add("Add Distinct(...) to enforce 'one value per house' for each attribute.")
    sb.add("Add the clue constraints; Z3 reports the unique assignment.")
    prob = ("Three houses sit in a row at positions 1, 2, 3, each with a colour and a drink. "
            + " ".join(t for t, _ in selected)
            + " Determine each house's colour and drink.")
    return prob, ans, {"category": "zebra_z3", "step_text": sb.text()}


# ─────────────────────────── SAT with witness ───────────────────────────

def gen_sat_witness_z3(dif):
    """Random CNF formula; ask Z3 for a satisfying assignment or UNSAT verdict."""
    n_vars = random.choice([3, 4]) if dif != "hard" else random.choice([4, 5])
    var_names = [f"x{i + 1}" for i in range(n_vars)]
    z3_vars = {nm: z3.Bool(nm) for nm in var_names}

    n_clauses = random.randint(3, 6) if dif != "hard" else random.randint(5, 9)
    text_clauses = []
    s = z3.Solver()
    for _ in range(n_clauses):
        k = random.choice([2, 3]) if n_vars >= 3 else 2
        chosen = random.sample(var_names, k)
        lits_text = []
        lits_z3 = []
        for v in chosen:
            if random.random() < 0.5:
                lits_text.append(f"¬{v}"); lits_z3.append(z3.Not(z3_vars[v]))
            else:
                lits_text.append(v); lits_z3.append(z3_vars[v])
        text_clauses.append("(" + " ∨ ".join(lits_text) + ")")
        s.add(z3.Or(*lits_z3))

    formula = " ∧ ".join(text_clauses)
    if s.check() == z3.sat:
        m = s.model()
        witness = ", ".join(
            f"{v}={'T' if bool(m.eval(z3_vars[v], model_completion=True)) else 'F'}"
            for v in var_names
        )
        ans = f"SAT; witness: {witness}"
    else:
        ans = "UNSAT"

    sb = StepBuilder()
    sb.add("Convert each clause into a Z3 disjunction over Boolean literals.")
    sb.add("If the conjunction is SAT, Z3 returns a model — read off variable values.")
    sb.add("If UNSAT, no assignment satisfies all clauses.")
    prob = f"Determine whether the formula is satisfiable, and if so produce a witness assignment: {formula}"
    return prob, ans, {"category": "sat_witness_z3", "step_text": sb.text()}
