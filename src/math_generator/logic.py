"""Propositional / quantifier logic, deduction, knights/knaves, latex simplify, proofs."""

import math
import random

import sympy as sp

from .core import canon_num, rint


# ─────────────────────────── PROPOSITIONAL ───────────────────────────

def gen_logic_propositional(dif):
    """Truth value of a propositional logic statement."""
    p, q = random.choice([True, False]), random.choice([True, False])
    pstr = "True" if p else "False"
    qstr = "True" if q else "False"
    connectives = [
        (f"p ∧ q (AND), given p = {pstr}, q = {qstr}",            p and q,            "AND is True only when both are True"),
        (f"p ∨ q (OR), given p = {pstr}, q = {qstr}",             p or q,             "OR is True when at least one is True"),
        (f"¬p (NOT p), given p = {pstr}",                         not p,              "NOT flips the truth value"),
        (f"p → q (p IMPLIES q), given p = {pstr}, q = {qstr}",    (not p) or q,       "Implication is False only when p=True, q=False"),
        (f"p ↔ q (p IFF q), given p = {pstr}, q = {qstr}",        p == q,             "Biconditional is True iff p and q have the same truth value"),
        (f"¬(p ∧ q) (NAND), given p = {pstr}, q = {qstr}",        not (p and q),      "NAND = NOT(AND)"),
        (f"¬p ∨ ¬q, given p = {pstr}, q = {qstr}",                (not p) or (not q), "By De Morgan: ¬(p∧q)"),
        (f"p ∧ ¬q, given p = {pstr}, q = {qstr}",                 p and (not q),      "p AND NOT q"),
    ]
    stmt, val, reason = random.choice(connectives)
    ans = "True" if val else "False"
    return f"Determine the truth value of: {stmt}", ans, {
        "category": "logic_propositional",
        "step_text": f"Step: {reason}. Result: {ans}",
    }


# ─────────────────────────── QUANTIFIER (SymPy-verified) ───────────────────────────

def gen_logic_quantifier_sympy(dif):
    """Random universally/existentially quantified arithmetic statement, verified by SymPy."""
    x = sp.Symbol("x", real=True)
    kind = random.choice(["exists_root", "forall_nonneg", "exists_int_sq", "forall_inequality"])

    if kind == "exists_root":
        a = random.randint(1, 4)
        b = random.randint(-5, 5)
        c = random.randint(-10, 10)
        target = random.randint(-15, 15)
        expr = a * x ** 2 + b * x + c
        sols = sp.solve(sp.Eq(expr, target), x)
        real_sols = [s for s in sols if s.is_real]
        true = bool(real_sols)
        prob = f"True or False: ∃x ∈ ℝ such that ${sp.latex(expr)} = {target}$."
        ans = "True" if true else "False"
        if true:
            step = f"Step: SymPy returns real solution(s); e.g. x = {sp.latex(real_sols[0])}."
        else:
            disc = b * b - 4 * a * (c - target)
            step = f"Step: discriminant = {disc} < 0, so no real x."
        return prob, ans, {"category": "logic_quantifier", "step_text": step}

    if kind == "forall_nonneg":
        a = random.randint(1, 5)
        b = random.randint(-3, 3)
        c = random.randint(0, 10)
        expr = a * x ** 2 + b * x + c
        disc = b * b - 4 * a * c
        true = (disc <= 0)
        prob = f"True or False: ∀x ∈ ℝ, ${sp.latex(expr)} \\geq 0$."
        ans = "True" if true else "False"
        if true:
            step = f"Step: discriminant = {disc} ≤ 0 and a > 0, so the quadratic stays non-negative."
        else:
            step = f"Step: discriminant = {disc} > 0, so the quadratic dips below 0 between its real roots."
        return prob, ans, {"category": "logic_quantifier", "step_text": step}

    if kind == "exists_int_sq":
        n = random.randint(0, 200)
        true = (math.isqrt(n) ** 2 == n)
        prob = f"True or False: ∃x ∈ ℤ such that x² = {n}."
        ans = "True" if true else "False"
        if true:
            step = f"Step: {n} = {math.isqrt(n)}², so x = ±{math.isqrt(n)} works."
        else:
            step = f"Step: √{n} is not an integer (≈ {math.sqrt(n):.3f})."
        return prob, ans, {"category": "logic_quantifier", "step_text": step}

    # forall_inequality: ∀x ∈ ℝ, ax² + bx + c ≥ d (or > d)
    a = random.randint(1, 5)
    b = random.randint(-4, 4)
    c = random.randint(-3, 6)
    d = random.randint(-3, 3)
    expr = a * x ** 2 + b * x + c
    diff = expr - d
    coeffs = sp.Poly(diff, x).all_coeffs()
    A, B, C = coeffs
    disc = B * B - 4 * A * C
    true = (A > 0 and disc <= 0)
    prob = f"True or False: ∀x ∈ ℝ, ${sp.latex(expr)} \\geq {d}$."
    ans = "True" if true else "False"
    if true:
        step = f"Step: ({sp.latex(expr)}) − {d} has discriminant {disc} ≤ 0 with leading coef {A} > 0."
    else:
        step = f"Step: ({sp.latex(expr)}) − {d} has discriminant {disc} > 0, so it goes below 0 somewhere."
    return prob, ans, {"category": "logic_quantifier", "step_text": step}


# ─────────────────────────── DEDUCTION (RANDOMIZED) ───────────────────────────

def gen_deduction_chain(dif):
    """Multi-step implication chain; ask for the conclusion."""
    n = random.randint(2, 4)
    letters = random.sample(["p", "q", "r", "s", "t", "u"], n + 1)
    premises = [f"{letters[i]} → {letters[i + 1]}" for i in range(n)]
    style = random.choice(["forward", "backward"])
    if style == "forward":
        premises.append(letters[0])
        ans = letters[-1]
        step = f"Step: apply modus ponens repeatedly along {' → '.join(letters)}."
    else:
        premises.append(f"¬{letters[-1]}")
        ans = f"¬{letters[0]}"
        step = f"Step: apply modus tollens backward through the chain to get ¬{letters[0]}."
    prob = ("Premises: " + "; ".join(f"({i + 1}) {p}" for i, p in enumerate(premises))
            + ". What is the conclusion?")
    return prob, ans, {"category": "deduction_chain", "step_text": step}


def gen_deduction_validity(dif):
    """Random argument: valid or fallacy?"""
    A, B = random.sample(["P", "Q", "R", "S", "T"], 2)
    args = [
        ([f"{A} → {B}", A],          B,            True,  "Modus ponens"),
        ([f"{A} → {B}", f"¬{B}"],    f"¬{A}",      True,  "Modus tollens"),
        ([f"{A} → {B}", B],          A,            False, "Affirming the consequent (fallacy)"),
        ([f"{A} → {B}", f"¬{A}"],    f"¬{B}",      False, "Denying the antecedent (fallacy)"),
        ([f"{A} ∨ {B}", f"¬{A}"],    B,            True,  "Disjunctive syllogism"),
        ([f"{A} ∧ {B}"],             A,            True,  "Conjunction elimination"),
        ([A, B],                     f"{A} ∧ {B}", True,  "Conjunction introduction"),
    ]
    prems, conc, valid, why = random.choice(args)
    prob = ("Is this argument valid? Premises: "
            + "; ".join(f"({i + 1}) {p}" for i, p in enumerate(prems))
            + f". Conclusion: {conc}.")
    ans = "Yes" if valid else "No"
    return prob, ans, {"category": "deduction_validity", "step_text": f"Step: {why}."}


# ─────────────────────────── LATEX (SymPy-backed) ───────────────────────────

def gen_latex_simplify_sympy(dif):
    """Random algebraic simplification via SymPy.
    Tasks: factor / expand / simplify rational / partial fractions / trig simplify."""
    x = sp.Symbol("x")
    task = random.choice(["factor", "expand", "rational", "partial_fractions", "trig"])

    if task == "factor":
        deg = random.choice([2, 2, 3])
        roots = [random.randint(-6, 6) for _ in range(deg)]
        leading = random.choice([1, 1, 1, 2, -1])
        poly = sp.expand(leading * sp.Mul(*[(x - r) for r in roots]))
        ans = sp.factor(poly)
        prob = f"Factor the polynomial: ${sp.latex(poly)}$"
        return prob, f"${sp.latex(ans)}$", {"category": "latex_simplify_factor",
                                            "step_text": "Step: factor over the integers."}

    if task == "expand":
        n = random.choice([2, 3])
        terms = []
        for _ in range(n):
            c = random.randint(-5, 5)
            if c == 0: c = 1
            terms.append(x + c)
        product_unevaluated = sp.Mul(*terms, evaluate=False)
        ans = sp.expand(sp.Mul(*terms))
        prob = f"Expand and simplify: ${sp.latex(product_unevaluated)}$"
        return prob, f"${sp.latex(ans)}$", {"category": "latex_simplify_expand",
                                            "step_text": "Step: distribute and combine like terms."}

    if task == "rational":
        c = random.randint(-4, 4)
        extra = random.randint(-5, 5)
        if extra == c: extra += 1
        num = sp.expand((x - c) * (x + extra))
        den = (x - c)
        ans = sp.simplify(num / den)
        prob = f"Simplify $\\dfrac{{{sp.latex(num)}}}{{{sp.latex(den)}}}$ for $x \\neq {c}$."
        return prob, f"${sp.latex(ans)}$", {"category": "latex_simplify_rational",
                                            "step_text": f"Step: factor and cancel $(x - {c})$."}

    if task == "partial_fractions":
        a = random.randint(-4, 4)
        b = random.randint(-4, 4)
        if a == b: b += 1
        expr = 1 / ((x - a) * (x - b))
        ans = sp.apart(expr, x)
        prob = f"Decompose into partial fractions: ${sp.latex(expr)}$"
        return prob, f"${sp.latex(ans)}$", {"category": "latex_simplify_partial_fractions",
                                            "step_text": "Step: write as A/(x-a) + B/(x-b) and solve for A, B."}

    a = random.choice([1, 2, 3])
    candidates = [
        (sp.sin(a * x) ** 2 + sp.cos(a * x) ** 2,                "Pythagorean identity"),
        (1 - sp.cos(a * x) ** 2,                                 "Pythagorean identity"),
        (sp.sin(a * x) / sp.cos(a * x),                          "Definition of tangent"),
        (1 + sp.tan(a * x) ** 2,                                 "Pythagorean (sec form)"),
        (sp.sin(-a * x),                                          "Sine is odd"),
        (sp.cos(-a * x),                                          "Cosine is even"),
        (2 * sp.sin(a * x) * sp.cos(a * x),                       "Double-angle: 2 sin·cos = sin(2θ)"),
        (sp.cos(a * x) ** 2 - sp.sin(a * x) ** 2,                 "Double-angle for cosine"),
    ]
    e, why = random.choice(candidates)
    ans = sp.trigsimp(e)
    prob = f"Simplify the trig expression: ${sp.latex(e)}$"
    return prob, f"${sp.latex(ans)}$", {"category": "latex_simplify_trig",
                                        "step_text": f"Step: {why}."}


def gen_latex_computation(dif):
    """Multi-step LaTeX computation problems."""
    n1, n2 = rint(2, 6), rint(2, 6)
    a, b = rint(1, 8), rint(1, 8)
    problems = [
        (f"Compute $\\binom{{{n1 + n2}}}{{{n1}}}$.",
         str(math.comb(n1 + n2, n1)),
         f"C({n1 + n2},{n1}) = ({n1 + n2})! / ({n1}! · {n2}!)"),
        (f"Evaluate $\\sum_{{k=1}}^{{{a}}} k$.",
         str(a * (a + 1) // 2),
         "Sum = a(a+1)/2"),
        (f"Evaluate $\\sum_{{k=1}}^{{{b}}} k^2$.",
         str(b * (b + 1) * (2 * b + 1) // 6),
         "Sum of squares = n(n+1)(2n+1)/6"),
        (f"Evaluate $\\sum_{{k=0}}^{{{a - 1}}} 2^k$.",
         str(2 ** a - 1),
         f"Geometric: 2^{a} − 1"),
    ]
    prob, ans, step = random.choice(problems)
    return prob, ans, {"category": "latex_computation", "step_text": f"Step: {step}."}


# ─────────────────────────── PROOF ───────────────────────────

def gen_proof_concept(dif):
    """Random parameterised induction problem. The identity is sampled from a
    family; SymPy verifies it; the answer is a complete induction proof
    structure with the correctly-parameterised RHS."""
    import sympy as sp
    n = sp.Symbol("n", integer=True, positive=True)
    k = sp.Symbol("k", integer=True, positive=True)

    family = random.choice(["sum_k", "sum_k2", "sum_k3",
                            "sum_2k", "sum_geom_2",
                            "divides_p2_minus_p"])

    if family == "sum_k":
        lhs = sp.Sum(k, (k, 1, n))
        rhs = n * (n + 1) / 2
        statement = "$\\sum_{k=1}^{n} k = \\frac{n(n+1)}{2}$ for all $n \\geq 1$"
    elif family == "sum_k2":
        lhs = sp.Sum(k ** 2, (k, 1, n))
        rhs = n * (n + 1) * (2 * n + 1) / 6
        statement = "$\\sum_{k=1}^{n} k^2 = \\frac{n(n+1)(2n+1)}{6}$ for all $n \\geq 1$"
    elif family == "sum_k3":
        lhs = sp.Sum(k ** 3, (k, 1, n))
        rhs = (n * (n + 1) / 2) ** 2
        statement = "$\\sum_{k=1}^{n} k^3 = \\left(\\frac{n(n+1)}{2}\\right)^2$ for all $n \\geq 1$"
    elif family == "sum_2k":
        # 1 + 2 + 4 + ... + 2^(n-1) = 2^n - 1; index from k=0 to n-1
        lhs = sp.Sum(2 ** k, (k, 0, n - 1))
        rhs = 2 ** n - 1
        statement = "$\\sum_{k=0}^{n-1} 2^k = 2^n - 1$ for all $n \\geq 1$"
    elif family == "sum_geom_2":
        # 2 + 4 + 8 + ... + 2^n = 2^(n+1) - 2
        lhs = sp.Sum(2 ** k, (k, 1, n))
        rhs = 2 ** (n + 1) - 2
        statement = "$\\sum_{k=1}^{n} 2^k = 2^{n+1} - 2$ for all $n \\geq 1$"
    else:
        # n*(n+1) is even (or n^3 - n divisible by 6)
        statement = "$n^3 - n$ is divisible by 6 for all $n \\geq 1$"
        prob = f"Outline a proof by induction that {statement}."
        ans = ("Base n = 1: 1³ − 1 = 0, divisible by 6. ✓  "
               "Inductive step: assume k³ − k = 6m for some integer m. Then "
               "(k+1)³ − (k+1) = k³ + 3k² + 3k + 1 − k − 1 = (k³ − k) + 3k(k+1). "
               "k(k+1) is a product of consecutive integers, hence even, so 3k(k+1) is divisible by 6. "
               "Thus (k+1)³ − (k+1) = 6m + 6·(k(k+1)/2), divisible by 6.  "
               "By induction the claim holds for all n ≥ 1.")
        return prob, ans, {"category": "proof_concept",
                           "step_text": "Step: induction on n; verify base, then use (k+1)³ − (k+1) = (k³ − k) + 3k(k+1)."}

    # Verify the identity SymPy-side and produce a structured proof.
    if not sp.simplify(lhs.doit() - rhs) == 0:
        # Should not happen; fall through with an apology.
        return ("Verify the induction step for an unspecified summation identity.",
                "Verification failed — please consult standard induction templates.",
                {"category": "proof_concept", "step_text": "Step: verification failure; skip."})

    rhs_n_eq_1 = sp.simplify(rhs.subs(n, 1))
    rhs_n_plus_1 = sp.simplify(rhs.subs(n, n + 1))
    increment = sp.simplify(rhs_n_plus_1 - rhs)

    prob = f"Outline a proof by induction that {statement}."
    ans = (f"Base case (n=1): LHS = 1, RHS = {sp.latex(rhs_n_eq_1)}. ✓  "
           f"Inductive step: assume the identity holds for some n ≥ 1, i.e. "
           f"$\\sum = {sp.latex(rhs)}$. Adding the (n+1)-st term to both sides "
           f"increases the LHS by exactly that term, and the RHS by "
           f"${sp.latex(increment)}$ — these are equal by elementary algebra. "
           f"So the identity holds for n+1, completing the induction.")
    step = ("Step: induction. Verify base; assume P(n); add the (n+1)-st term; "
            "show RHS(n+1) − RHS(n) equals that term.")
    return prob, ans, {"category": "proof_concept", "step_text": step}
