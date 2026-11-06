"""Boolean algebra: truth tables, simplification (SymPy), SAT, tautology, DNF."""

import itertools
import random

import sympy as sp

from .core import bool_eval, bool_to_unicode


def _rand_bool_atom(vars_):
    v = random.choice(vars_)
    return ("¬" + v) if random.random() < 0.4 else v


def _rand_bool_expr(vars_, depth=2):
    if depth == 0 or random.random() < 0.3:
        return _rand_bool_atom(vars_)
    op = random.choice(["∧", "∨"])
    a = _rand_bool_expr(vars_, depth - 1)
    b = _rand_bool_expr(vars_, depth - 1)
    return f"({a} {op} {b})"


def gen_boolean_truth_table(dif):
    """Evaluate a randomly built 2–3 variable boolean expression at a random assignment."""
    vars_ = ["p", "q", "r"][: random.choice([2, 3])]
    expr = _rand_bool_expr(vars_, depth=2)
    env = {v: random.choice([True, False]) for v in vars_}
    val = bool_eval(expr, env)
    assignments = ", ".join(f"{v}={env[v]}" for v in vars_)
    prob = f"Evaluate the boolean expression {expr} given {assignments}."
    ans = "True" if val else "False"
    return prob, ans, {
        "category": "boolean_truth_table",
        "step_text": "Step: substitute the truth values; apply ¬ first, then ∧, then ∨.",
    }


def gen_boolean_simplify_sympy(dif):
    """Random Boolean expression simplified with SymPy (Quine–McCluskey)."""
    A_, B_, C_ = sp.symbols("A B C")
    pool = [A_, B_, C_]

    def rand_lit():
        v = random.choice(pool)
        return ~v if random.random() < 0.4 else v

    def rand_clause(depth):
        if depth == 0 or random.random() < 0.3:
            return rand_lit()
        op = random.choice([sp.And, sp.Or])
        return op(rand_clause(depth - 1), rand_clause(depth - 1))

    expr = simp = None
    for _ in range(15):
        expr = rand_clause(random.choice([2, 3]))
        try:
            simp = sp.simplify_logic(expr, form="dnf")
        except Exception:
            continue
        if simp != expr:
            break
    if simp is None:
        simp = expr

    prob = f"Simplify the Boolean expression to its minimal DNF: {bool_to_unicode(expr)}"
    ans = bool_to_unicode(simp)
    step = "Step: apply Quine–McCluskey minimisation (drop covered minterms, combine adjacent ones)."
    return prob, ans, {"category": "boolean_simplify", "step_text": step}


def gen_boolean_satisfiability(dif):
    """Is the random CNF-style formula satisfiable?"""
    vars_ = ["p", "q", "r"]
    n_clauses = random.randint(2, 4)
    clauses = []
    for _ in range(n_clauses):
        k = random.randint(1, 3)
        lits = []
        for v in random.sample(vars_, k):
            lits.append(("¬" + v) if random.random() < 0.5 else v)
        clauses.append("(" + " ∨ ".join(lits) + ")")
    formula = " ∧ ".join(clauses)
    sat = False
    for asst in itertools.product([False, True], repeat=len(vars_)):
        if bool_eval(formula, dict(zip(vars_, asst))):
            sat = True
            break
    prob = f"Is the formula satisfiable (does some assignment make it True)? {formula}"
    ans = "Yes" if sat else "No"
    return prob, ans, {
        "category": "boolean_sat",
        "step_text": "Step: enumerate all 2^n assignments; satisfiable iff at least one is True.",
    }


def gen_boolean_tautology(dif):
    """Is the formula always True?"""
    vars_ = ["p", "q"]
    if random.random() < 0.4:
        forms = ["p ∨ ¬p", "p → p", "(p ∧ q) → p", "p → (p ∨ q)",
                 "¬(p ∧ ¬p)", "(p → q) ↔ (¬q → ¬p)"]
        formula = random.choice(forms)
    else:
        a = _rand_bool_atom(vars_)
        b = _rand_bool_atom(vars_)
        op = random.choice(["∧", "∨", "→", "↔"])
        formula = f"({a} {op} {b})"
    is_taut = all(bool_eval(formula, dict(zip(vars_, asst)))
                  for asst in itertools.product([False, True], repeat=len(vars_)))
    prob = f"Is the formula a tautology (always True)? {formula}"
    ans = "Yes" if is_taut else "No"
    return prob, ans, {"category": "boolean_tautology",
                       "step_text": "Step: a formula is a tautology iff every truth-table row is True."}


def gen_boolean_dnf(dif):
    """Build a random truth table over 2 vars; ask for an equivalent DNF expression."""
    vars_ = ["p", "q"]
    table = {asst: random.choice([True, False])
             for asst in itertools.product([False, True], repeat=2)}
    if not any(table.values()):
        table[random.choice(list(table.keys()))] = True
    if all(table.values()):
        table[random.choice(list(table.keys()))] = False
    minterms = []
    for asst, v in table.items():
        if v:
            lits = [var if b else f"¬{var}" for var, b in zip(vars_, asst)]
            minterms.append("(" + " ∧ ".join(lits) + ")")
    dnf = " ∨ ".join(minterms)
    rows = "; ".join(f"({a[0]},{a[1]})→{table[a]}" for a in table)
    prob = f"Given truth table for f(p,q) with rows {rows}, write f in disjunctive normal form (DNF)."
    return prob, dnf, {"category": "boolean_dnf",
                       "step_text": "Step: take the disjunction of one minterm per True row."}
