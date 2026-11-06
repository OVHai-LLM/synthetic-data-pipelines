"""Algebra generators: linear systems, polynomials, inequalities, complex, vectors, sets."""

import math
import random
from fractions import Fraction

from .core import (canon_num, det3, fmt_poly, mat2str, rint)


def gen_system_2x2(dif):
    """Solve a 2×2 linear system with a clean integer solution."""
    x0, y0 = rint(-15, 15), rint(-15, 15)
    for _ in range(50):
        a1, b1 = rint(-8, 8, exclude_zero=True), rint(-8, 8)
        a2, b2 = rint(-8, 8), rint(-8, 8, exclude_zero=True)
        if a1 * b2 != a2 * b1:
            break
    c1 = a1 * x0 + b1 * y0
    c2 = a2 * x0 + b2 * y0

    def fmt_eq(a, b, c):
        lhs = ""
        if a == 1: lhs += "x"
        elif a == -1: lhs += "-x"
        else: lhs += f"{a}x"
        if b > 0: lhs += f" + {b}y"
        elif b < 0: lhs += f" - {abs(b)}y"
        return f"{lhs} = {c}"

    prob = f"Solve the system:\n  {fmt_eq(a1, b1, c1)}\n  {fmt_eq(a2, b2, c2)}"
    step = f"Step: use substitution or elimination. x = {x0}, y = {y0}"
    return prob, f"x = {x0}, y = {y0}", {"category": "system_2x2", "step_text": step}


def gen_system_3x3(dif):
    """Solve a 3×3 linear system."""
    x0, y0, z0 = rint(-6, 6), rint(-6, 6), rint(-6, 6)
    rows = []
    for _ in range(50):
        candidates = []
        for _ in range(3):
            a, b, c = rint(-5, 5, exclude_zero=True), rint(-5, 5), rint(-5, 5)
            d = a * x0 + b * y0 + c * z0
            candidates.append((a, b, c, d))
        if det3([[r[0], r[1], r[2]] for r in candidates]) != 0:
            rows = candidates
            break
    if not rows:
        rows = [(1, 0, 0, x0), (0, 1, 0, y0), (0, 0, 1, z0)]

    def fmt3(a, b, c, d):
        parts = []
        for coef, var in [(a, "x"), (b, "y"), (c, "z")]:
            if coef == 0: continue
            if coef == 1: parts.append(("+" if parts else "") + var)
            elif coef == -1: parts.append("-" + var)
            else: parts.append(("+" if coef > 0 and parts else "") + f"{coef}{var}")
        lhs = "".join(parts) or "0"
        return f"{lhs} = {d}"

    eqs = "\n  ".join(fmt3(*r) for r in rows)
    prob = f"Solve the system:\n  {eqs}"
    ans = f"x = {x0}, y = {y0}, z = {z0}"
    step = f"Step: Gaussian elimination yields {ans}."
    return prob, ans, {"category": "system_3x3", "step_text": step}


def gen_quadratic_complete(dif):
    """Quadratic with integer roots, factored form."""
    if dif == "medium":
        r1, r2 = rint(-8, 8), rint(-8, 8)
        a = 1
    else:
        a = rint(1, 4)
        r1, r2 = rint(-6, 6), rint(-6, 6)
    b = -a * (r1 + r2)
    c = a * r1 * r2
    poly = fmt_poly([a, b, c])
    prob = f"Solve for x: {poly} = 0"
    if r1 == r2:
        ans = f"x = {r1} (double root)"
        step = f"Step: discriminant = 0, x = {r1}"
    else:
        rs = sorted([r1, r2])
        ans = f"x = {rs[0]} or x = {rs[1]}"
        step = f"Step: factor as {a}(x - {r1})(x - {r2}) = 0"
    return prob, ans, {"category": "quadratic", "step_text": step}


def gen_polynomial_eval(dif):
    """Evaluate a polynomial at a point."""
    degree = 2 if dif == "medium" else random.choice([2, 3])
    coeffs = [rint(-5, 5) for _ in range(degree + 1)]
    if coeffs[0] == 0: coeffs[0] = 1
    x0 = rint(-5, 5)
    val = sum(c * (x0 ** (degree - i)) for i, c in enumerate(coeffs))
    poly = fmt_poly(coeffs)
    prob = f"Evaluate f(x) = {poly} at x = {x0}."
    step = "Step: substitute x = " + str(x0) + " term by term and sum."
    return prob, canon_num(val), {"category": "polynomial_eval", "step_text": step}


def gen_polynomial_cubic(dif):
    """Cubic with three integer roots."""
    r1, r2, r3 = rint(-4, 4), rint(-4, 4), rint(-4, 4)
    b = -(r1 + r2 + r3)
    c = r1 * r2 + r1 * r3 + r2 * r3
    d = -(r1 * r2 * r3)
    poly = fmt_poly([1, b, c, d])
    prob = f"Find all roots of: {poly} = 0"
    roots = sorted(set([r1, r2, r3]))
    ans = "x = " + ", ".join(str(r) for r in roots)
    step = f"Step: factor as (x - {r1})(x - {r2})(x - {r3}) = 0."
    return prob, ans, {"category": "cubic_roots", "step_text": step}


def gen_inequality_linear(dif):
    """Solve a linear inequality."""
    a = rint(1, 10, exclude_zero=True)
    b, c = rint(-20, 20), rint(-20, 20)
    op = random.choice(["<", "≤", ">", "≥"])
    rhs = Fraction(c - b, a)
    if a < 0:
        flip = {"<": ">", "≤": "≥", ">": "<", "≥": "≤"}
        ans_op = flip[op]
    else:
        ans_op = op
    prob = f"Solve: {a}x + {b} {op} {c}"
    if rhs.denominator == 1:
        ans = f"x {ans_op} {rhs.numerator}"
    else:
        ans = f"x {ans_op} {rhs.numerator}/{rhs.denominator}"
    step = f"Step: {a}x {op} {c - b}, then divide by {a}."
    return prob, ans, {"category": "inequality_linear", "step_text": step}


def gen_absolute_value(dif):
    """Solve |ax + b| = c."""
    a = rint(1, 8, exclude_zero=True)
    b = rint(-10, 10)
    c = rint(0, 20)
    prob = f"Solve: |{a}x + {b}| = {c}"
    if c < 0:
        c = abs(c)
    x1 = Fraction(c - b, a)
    x2 = Fraction(-c - b, a)
    if x1 == x2 or c == 0:
        ans = f"x = {canon_num(x1)}"
    else:
        xs = sorted([x1, x2])
        ans = f"x = {canon_num(xs[0])} or x = {canon_num(xs[1])}"
    step = f"Step: split |·| into {a}x + {b} = {c} or {a}x + {b} = -{c}."
    return prob, ans, {"category": "absolute_value", "step_text": step}


# ─────────────────────────── COMPLEX NUMBERS ───────────────────────────

def gen_complex_arithmetic(dif):
    a, b, c, d = rint(-6, 6), rint(-6, 6), rint(-6, 6), rint(-6, 6)
    op = random.choice(["add", "sub", "mul", "mod", "conj"] if dif == "medium"
                       else ["add", "sub", "mul", "div", "mod", "conj", "power"])

    def fmt(r, i):
        if i == 0: return str(r)
        if r == 0: return f"{i}i" if i != 1 else "i"
        sign = "+" if i > 0 else "-"
        ai = abs(i)
        return f"{r} {sign} {ai}i" if ai != 1 else f"{r} {sign} i"

    z1, z2 = fmt(a, b), fmt(c, d)

    if op == "add":
        r, i = a + c, b + d
        prob = f"Compute $({z1}) + ({z2})$."
        ans = fmt(r, i)
        step = "Step: add real and imaginary parts."
    elif op == "sub":
        r, i = a - c, b - d
        prob = f"Compute $({z1}) - ({z2})$."
        ans = fmt(r, i)
        step = "Step: subtract real and imaginary parts."
    elif op == "mul":
        r, i = a * c - b * d, a * d + b * c
        prob = f"Compute $({z1}) \\cdot ({z2})$."
        ans = fmt(r, i)
        step = "Step: (a+bi)(c+di) = (ac-bd) + (ad+bc)i."
    elif op == "div" and (c != 0 or d != 0):
        denom = c * c + d * d
        if denom == 0: c, d, denom = 1, 0, 1
        rf = Fraction(a * c + b * d, denom)
        i_f = Fraction(b * c - a * d, denom)
        prob = f"Compute $\\frac{{{z1}}}{{{z2}}}$."
        ans = f"${canon_num(rf)} + {canon_num(i_f)}i$"
        step = "Step: multiply numerator and denominator by conjugate of the denominator."
    elif op == "mod":
        mag_sq = a * a + b * b
        prob = f"Find $|{z1}|$."
        if math.isqrt(mag_sq) ** 2 == mag_sq:
            ans = str(math.isqrt(mag_sq))
        else:
            ans = f"√{mag_sq}"
        step = f"Step: |a+bi| = √(a²+b²) = √({a}² + {b}²) = √{mag_sq}."
    elif op == "conj":
        prob = f"Find the complex conjugate of ${z1}$."
        ans = fmt(a, -b)
        step = "Step: flip the sign of the imaginary part."
    elif op == "power":
        r, i = a * a - b * b, 2 * a * b
        prob = f"Compute $({z1})^2$."
        ans = fmt(r, i)
        step = "Step: (a+bi)² = a²−b² + 2abi."
    else:
        r, i = a + c, b + d
        prob = f"Compute $({z1}) + ({z2})$."
        ans = fmt(r, i)
        step = "Step: add real and imaginary parts."
    return prob, ans, {"category": "complex_arithmetic", "step_text": step}


# ─────────────────────────── VECTORS ───────────────────────────

def gen_vector_ops(dif):
    op = random.choice(["add", "scale", "dot", "magnitude"])
    n = 2 if dif == "medium" else random.choice([2, 3])
    lo, hi = -6, 6

    if op == "add":
        u = [rint(lo, hi) for _ in range(n)]
        v = [rint(lo, hi) for _ in range(n)]
        result = [u[i] + v[i] for i in range(n)]
        prob = f"Compute $\\mathbf{{u}} + \\mathbf{{v}}$ where $\\mathbf{{u}} = {mat2str([[x] for x in u])}$ and $\\mathbf{{v}} = {mat2str([[x] for x in v])}$."
        ans = f"${mat2str([[x] for x in result])}$"
        step = "Step: add corresponding components."
        return prob, ans, {"category": "vector_add", "step_text": step}

    if op == "scale":
        u = [rint(lo, hi) for _ in range(n)]
        k = rint(-5, 5, exclude_zero=True)
        result = [k * x for x in u]
        prob = f"Compute ${k}\\mathbf{{u}}$ where $\\mathbf{{u}} = {mat2str([[x] for x in u])}$."
        ans = f"${mat2str([[x] for x in result])}$"
        step = f"Step: multiply each component by {k}."
        return prob, ans, {"category": "vector_scale", "step_text": step}

    if op == "dot":
        u = [rint(lo, hi) for _ in range(n)]
        v = [rint(lo, hi) for _ in range(n)]
        result = sum(u[i] * v[i] for i in range(n))
        prob = f"Compute the dot product of $\\mathbf{{u}} = {mat2str([[x] for x in u])}$ and $\\mathbf{{v}} = {mat2str([[x] for x in v])}$."
        ans = str(result)
        step = "Step: u·v = Σ uᵢ vᵢ."
        return prob, ans, {"category": "dot_product", "step_text": step}

    # magnitude
    u = [rint(lo, hi) for _ in range(n)]
    sq = sum(x * x for x in u)
    prob = f"Find $\\|\\mathbf{{u}}\\|$ where $\\mathbf{{u}} = {mat2str([[x] for x in u])}$."
    if math.isqrt(sq) ** 2 == sq:
        ans = str(math.isqrt(sq))
    else:
        ans = f"√{sq}"
    step = f"Step: ‖u‖ = √(Σ uᵢ²) = √{sq}."
    return prob, ans, {"category": "vector_magnitude", "step_text": step}


# ─────────────────────────── SETS ───────────────────────────

def gen_set_theory(dif):
    """Set operations on small integer sets."""
    universe = list(range(1, 16))
    A_list = sorted(random.sample(universe, rint(3, 7)))
    B_list = sorted(random.sample(universe, rint(3, 7)))
    A, B = set(A_list), set(B_list)

    op = random.choice(["union", "intersection", "difference", "sym_diff", "subset"])
    if op == "union":
        res = sorted(A | B)
        prob = f"Let A = {{{', '.join(map(str, A_list))}}} and B = {{{', '.join(map(str, B_list))}}}. Find A ∪ B."
        ans = "{" + ", ".join(map(str, res)) + "}"
        step = "Step: A ∪ B contains every element appearing in A or B."
    elif op == "intersection":
        res = sorted(A & B)
        prob = f"Let A = {{{', '.join(map(str, A_list))}}} and B = {{{', '.join(map(str, B_list))}}}. Find A ∩ B."
        ans = "{" + ", ".join(map(str, res)) + "}" if res else "∅"
        step = "Step: A ∩ B contains the elements in both A and B."
    elif op == "difference":
        res = sorted(A - B)
        prob = f"Let A = {{{', '.join(map(str, A_list))}}} and B = {{{', '.join(map(str, B_list))}}}. Find A \\ B (A minus B)."
        ans = "{" + ", ".join(map(str, res)) + "}" if res else "∅"
        step = "Step: A \\ B contains elements in A but not B."
    elif op == "sym_diff":
        res = sorted(A ^ B)
        prob = f"Let A = {{{', '.join(map(str, A_list))}}} and B = {{{', '.join(map(str, B_list))}}}. Find A △ B (symmetric difference)."
        ans = "{" + ", ".join(map(str, res)) + "}" if res else "∅"
        step = "Step: A △ B = (A \\ B) ∪ (B \\ A)."
    else:
        is_sub = A.issubset(B)
        prob = f"Let A = {{{', '.join(map(str, A_list))}}} and B = {{{', '.join(map(str, B_list))}}}. Is A ⊆ B?"
        ans = "Yes" if is_sub else "No"
        step = f"Step: check whether every element of A is in B ({'all are' if is_sub else 'not all are'})."
    return prob, ans, {"category": "set_theory", "step_text": step}
