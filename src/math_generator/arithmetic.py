"""Arithmetic, fractions, comparisons, sequences, modular, prime test, geometric sequence."""

import math
import random
from decimal import Decimal
from fractions import Fraction

from .core import (DEC_P, StepBuilder, canon_dec, canon_num, is_prime,
                   rand_div, rand_mul, rdec, rfraction, rint)


# ─────────────────────────── step-text helpers ───────────────────────────
# Each helper renders one of two styles, picked 50/50 per sample so the
# corpus mixes textbook substitution (A) and natural-language prose (B).

def _fmt_addition_chain(signs, terms, render, style):
    """signs[i] in {+1,-1}; terms[i] is a non-negative number; render(x) -> str."""
    if style == "A":
        lines = []
        cur = signs[0] * terms[0]
        for i in range(1, len(terms)):
            cur = cur + signs[i] * terms[i]
            parts = [render(cur)]
            for j in range(i + 1, len(terms)):
                op = "+" if signs[j] > 0 else "-"
                parts.append(f"{op} {render(terms[j])}")
            lines.append("= " + " ".join(parts))
        return "\n".join(lines)
    # style B
    pieces = []
    cur = signs[0] * terms[0]
    for i in range(1, len(terms)):
        prev = cur
        op = "+" if signs[i] > 0 else "-"
        cur = cur + signs[i] * terms[i]
        pieces.append(f"{render(prev)} {op} {render(terms[i])} = {render(cur)}")
    return "Going left to right: " + "; ".join(pieces) + "."


def _fmt_mul_big2(a, b, style):
    """Long multiplication of two integers (any sign) via place-value decomposition."""
    abs_a, abs_b = abs(a), abs(b)
    n = len(str(abs_b))
    pieces = [(int(d), 10 ** (n - 1 - i))
              for i, d in enumerate(str(abs_b)) if int(d) != 0]
    total = abs_a * abs_b
    signed_total = a * b
    sign_note = ""
    if (a < 0) ^ (b < 0):
        sign_note = (f" Since one factor is negative, {a} × {b} = {signed_total}.")
    elif a < 0 and b < 0:
        sign_note = (f" Both factors are negative, so {a} × {b} = {signed_total}.")

    if style == "A":
        lines = []
        if len(pieces) > 1:
            decomp = " + ".join(str(d * p) for d, p in pieces)
            dist = " + ".join(f"{abs_a} × {d * p}" for d, p in pieces)
            partials = " + ".join(str(abs_a * d * p) for d, p in pieces)
            lines.append(f"= {abs_a} × ({decomp})")
            lines.append(f"= {dist}")
            lines.append(f"= {partials}")
        lines.append(f"= {total}")
        if sign_note:
            lines.append(sign_note.strip())
        return "\n".join(lines)
    # style B
    if len(pieces) > 1:
        decomp = " + ".join(str(d * p) for d, p in pieces)
        dist = " + ".join(f"{abs_a} × {d * p}" for d, p in pieces)
        partials = " + ".join(str(abs_a * d * p) for d, p in pieces)
        prose = (f"Decompose {abs_b} = {decomp}. "
                 f"Then {abs_a} × {abs_b} = {dist} = {partials} = {total}.")
    else:
        prose = f"{abs_a} × {abs_b} = {total}."
    return prose + sign_note


def _fmt_mul_multi(factors, style):
    """Multi-factor product (3-5 small factors)."""
    if style == "A":
        lines = []
        cur = factors[0] * factors[1]
        remaining = list(factors[2:])
        while True:
            parts = [str(cur) if cur >= 0 else f"({cur})"]
            for f in remaining:
                parts.append(f"× {f}" if f >= 0 else f"× ({f})")
            lines.append("= " + " ".join(parts))
            if not remaining:
                break
            cur *= remaining.pop(0)
        return "\n".join(lines)
    # style B
    pieces = []
    cur = factors[0]
    for f in factors[1:]:
        prev = cur
        cur *= f
        prev_s = str(prev) if prev >= 0 else f"({prev})"
        f_s = str(f) if f >= 0 else f"({f})"
        pieces.append(f"{prev_s} × {f_s} = {cur}")
    return "Multiplying left to right: " + ", then ".join(pieces) + "."


def _fmt_long_chain(ops, terms, style):
    """Long arithmetic chain (5-12 operands over + - × ÷, integer)."""
    def paren(v):
        return str(v) if v >= 0 else f"({v})"

    def render_expr(o_list, t_list):
        parts = [str(t_list[0])]  # first term: no parens (matches problem render)
        for o, t in zip(o_list, t_list[1:]):
            parts.append(f"{o} {paren(t)}")
        return " ".join(parts)

    if style == "A":
        lines = []
        cur_ops, cur_terms = list(ops), list(terms)
        # collapse × ÷ leftmost first
        while True:
            mul_idx = None
            for i, o in enumerate(cur_ops):
                if o in ("×", "÷"):
                    mul_idx = i
                    break
            if mul_idx is None:
                break
            left, right = cur_terms[mul_idx], cur_terms[mul_idx + 1]
            result = left * right if cur_ops[mul_idx] == "×" else left // right
            cur_terms = cur_terms[:mul_idx] + [result] + cur_terms[mul_idx + 2:]
            cur_ops = cur_ops[:mul_idx] + cur_ops[mul_idx + 1:]
            lines.append("= " + render_expr(cur_ops, cur_terms))
        # collapse + - leftmost first
        while len(cur_terms) > 1:
            left, right = cur_terms[0], cur_terms[1]
            op = cur_ops[0]
            result = left + right if op == "+" else left - right
            cur_terms = [result] + cur_terms[2:]
            cur_ops = cur_ops[1:]
            lines.append("= " + render_expr(cur_ops, cur_terms))
        return "\n".join(lines)

    # style B
    muls = []
    cur_ops, cur_terms = list(ops), list(terms)
    while True:
        mul_idx = None
        for i, o in enumerate(cur_ops):
            if o in ("×", "÷"):
                mul_idx = i
                break
        if mul_idx is None:
            break
        left, right = cur_terms[mul_idx], cur_terms[mul_idx + 1]
        op = cur_ops[mul_idx]
        result = left * right if op == "×" else left // right
        muls.append(f"{paren(left)} {op} {paren(right)} = {result}")
        cur_terms = cur_terms[:mul_idx] + [result] + cur_terms[mul_idx + 2:]
        cur_ops = cur_ops[:mul_idx] + cur_ops[mul_idx + 1:]
    # remaining +/- expression
    final_expr = render_expr(cur_ops, cur_terms)
    cur = cur_terms[0]
    for o, t in zip(cur_ops, cur_terms[1:]):
        cur = cur + t if o == "+" else cur - t

    if muls:
        line1 = "Apply × and ÷ first: " + "; ".join(muls) + "."
        if len(cur_terms) > 1:
            return line1 + f"\nThen sum left to right: {final_expr} = {cur}."
        return line1
    # no × ÷ — just running sum
    pieces = []
    running = cur_terms[0]
    for o, t in zip(cur_ops, cur_terms[1:]):
        prev = running
        running = running + t if o == "+" else running - t
        pieces.append(f"{prev} {o} {t} = {running}")
    return "Going left to right: " + "; ".join(pieces) + "."


def gen_addition(dif):
    if random.random() < DEC_P:
        n_terms = 2 if dif == "simple" else random.choice([2, 3])
        allow_neg = (dif != "simple")
        terms = [rdec(0, 100, allow_neg=allow_neg) for _ in range(n_terms)]
        prob = " + ".join(canon_dec(t) for t in terms)
        ans = sum(terms, Decimal(0))
        return prob, canon_dec(ans, 2), {"category": "addition_dec"}
    if dif == "simple":
        terms = [rint(0, 100) for _ in range(random.choice([2, 3]))]
    else:
        terms = [rint(-200, 200) for _ in range(random.choice([3, 4]))]
    prob = " + ".join(map(str, terms))
    return prob, canon_num(sum(terms)), {"category": "addition"}


def gen_subtraction(dif):
    if random.random() < DEC_P:
        allow_neg = (dif != "simple")
        a, b = rdec(0, 200, allow_neg=allow_neg), rdec(0, 200, allow_neg=allow_neg)
        return f"{canon_dec(a)} - {canon_dec(b)}", canon_dec(a - b, 2), {"category": "subtraction_dec"}
    lo, hi = (0, 200) if dif == "simple" else (-300, 300)
    a, b = rint(lo, hi), rint(lo, hi)
    return f"{a} - {b}", canon_num(a - b), {"category": "subtraction"}


def gen_multiplication(dif):
    lo, hi = (0, 12) if dif == "simple" else (-25, 25)
    a, b = rint(lo, hi), rint(lo, hi)
    return f"{a} {rand_mul()} {b}", canon_num(a * b), {"category": "multiplication"}


def gen_division_exact(dif):
    if random.random() < DEC_P:
        divisors = [2, 4, 5, 8, 10, 20, 25, 40, 50]
        b = random.choice(divisors)
        q = rdec(0, 60, allow_neg=(dif != "simple"))
        a = Decimal(b) * q
        return f"{canon_dec(a)} {rand_div()} {b}", canon_dec(q, 2), {"category": "division_dec"}
    lo, hi = (1, 12) if dif == "simple" else (1, 25)
    b = rint(lo, hi, exclude_zero=True)
    q = rint(0, 50) if dif == "simple" else rint(-60, 60)
    return f"{b * q} {rand_div()} {b}", canon_num(q), {"category": "division_exact"}


def gen_mixed_expression(dif):
    ops = ["+", "-", "×"]
    if dif != "simple":
        ops.append("÷")
    a, b, c = rint(-30, 30), rint(-30, 30), rint(-30, 30)
    op1, op2 = random.choice(ops), random.choice(ops)

    def apply(x, op, y):
        if op == "+": return x + y
        if op == "-": return x - y
        if op == "×": return x * y
        if op == "÷":
            y = y if y != 0 else 1
            q = rint(-12, 12)
            return y * q, "÷", y, q
        raise ValueError

    if random.random() < 0.5:
        if op1 == "÷":
            a, o1, b, q1 = apply(a, "÷", b); left_val = q1
        else:
            o1 = op1; left_val = apply(a, op1, b)
        if op2 == "÷":
            _b = c if c != 0 else 1; q2 = rint(-12, 12)
            expr = f"({a} {o1} {b}) ÷ {_b}"; ans = q2
        else:
            expr = f"({a} {o1} {b}) {op2} {c}"; ans = apply(left_val, op2, c)
    else:
        if op2 == "÷":
            b, o2, c, q2 = apply(b, "÷", c); right_val = q2
        else:
            o2 = op2; right_val = apply(b, op2, c)
        if op1 == "÷":
            _b = right_val if right_val != 0 else 1; q1 = rint(-12, 12)
            expr = f"{_b * q1} ÷ ({b} {o2} {c})"; ans = q1
        else:
            expr = f"{a} {op1} ({b} {o2} {c})"; ans = apply(a, op1, right_val)

    expr = expr.replace("×", rand_mul()).replace("÷", rand_div())
    return expr, canon_num(ans), {"category": "mixed_integer"}


def gen_fraction_arith(dif):
    ops = ["+", "-"] if dif == "simple" else ["+", "-", "×", "÷"]
    op = random.choice(ops)
    f1 = rfraction(2, 12, allow_neg=(dif != "simple"))
    f2 = rfraction(2, 12, allow_neg=(dif != "simple"))
    if op == "+":
        res = f1 + f2
    elif op == "-":
        res = f1 - f2
    elif op == "×":
        res = f1 * f2
    else:
        if f2 == 0:
            f2 = Fraction(1, 3)
        res = f1 / f2
    render_op = rand_mul() if op == "×" else (rand_div() if op == "÷" else op)
    return f"{canon_num(f1)} {render_op} {canon_num(f2)}", canon_num(res), {"category": "fractions"}


def gen_linear_equation(dif):
    a = rint(1, 12, exclude_zero=True)
    x_true = rint(-50, 50)
    b = rint(-40, 40)
    c = a * x_true + b
    if dif != "simple" and random.random() < 0.5:
        p = rint(-10, 10)
        c = a * (x_true + p) + b
        prob = f"Solve for x: {a}(x + {p}) + {b} = {c}"
        step = f"Step: x = ({c} - {b} - {a}×{p}) / {a}"
    else:
        prob = f"Solve for x: {a}x + {b} = {c}"
        step = f"Step: x = ({c} - {b}) / {a}"
    return prob, canon_num(x_true), {"category": "linear_equation", "step_text": step}


def gen_gcd_lcm(dif):
    a, b = rint(1, 200), rint(1, 200)
    kind = random.choice(["gcd", "lcm"])
    if kind == "gcd":
        ans = math.gcd(a, b)
        prob = f"GCD({a}, {b}) = ?"
    else:
        ans = abs(a * b) // math.gcd(a, b)
        prob = f"LCM({a}, {b}) = ?"
    return prob, canon_num(ans), {"category": kind}


def gen_sequence_next(dif):
    start = rint(-50, 50)
    step = rint(-10, 10, exclude_zero=True)
    length = 4 if dif == "simple" else 5
    seq = [start + i * step for i in range(length)]
    prob = f"Next term in sequence: {', '.join(map(str, seq))}, ?"
    return prob, canon_num(start + length * step), {"category": "sequence_arith"}


def gen_compare(dif):
    if random.random() < DEC_P:
        allow_neg = (dif != "simple")
        d1, d2 = rdec(0, 200, allow_neg=allow_neg), rdec(0, 200, allow_neg=allow_neg)
        prob = f"Which is larger: {canon_dec(d1)} or {canon_dec(d2)}?"
        return prob, canon_dec(d1 if d1 >= d2 else d2), {"category": "compare_dec"}
    if random.random() < 0.5:
        a, b = rint(-200, 200), rint(-200, 200)
        prob = f"Which is larger: {a} or {b}?"
        ans = str(a) if a >= b else str(b)
    else:
        f1 = rfraction(2, 20, allow_neg=(dif != "simple"))
        f2 = rfraction(2, 20, allow_neg=(dif != "simple"))
        prob = f"Which is larger: {canon_num(f1)} or {canon_num(f2)}?"
        ans = canon_num(f1) if f1 >= f2 else canon_num(f2)
    return prob, ans, {"category": "compare"}


def gen_rounding(dif):
    places = 1 if dif == "simple" else 2
    x = random.uniform(-100, 100)
    prob = f"Round {x:.4f} to {places} decimal place{'s' if places > 1 else ''}."
    return prob, f"{round(x, places):.{places}f}", {"category": "rounding"}


def gen_modular(dif):
    a, b = rint(0, 10_000), rint(2, 97)
    return f"{a} mod {b} = ?", canon_num(a % b), {"category": "mod"}


def gen_prime_test(dif):
    """Identify whether n is prime."""
    n = rint(10, 200)
    prob = f"Is {n} a prime number?"
    ans = "Yes" if is_prime(n) else "No"
    if not is_prime(n):
        for d in range(2, int(n ** 0.5) + 1):
            if n % d == 0:
                step = f"Step: {n} = {d} × {n // d}, so it's composite"
                break
        else:
            step = f"Step: Check divisibility up to √{n}"
    else:
        step = f"Step: {n} has no divisors other than 1 and itself"
    return prob, ans, {"category": "prime_test", "step_text": step}


def gen_addition_chain(dif):
    """Long chains of additions / subtractions over ints or decimals.

    Closes the coverage gap left by gen_addition (max 4 terms, |x|≤200).
    Here: 5 to 20 terms, magnitudes up to 4 digits.
    """
    if dif == "simple":
        n_terms = random.choice([5, 6, 7, 8])
        magnitude = random.choice([100, 500, 1000])
        signs = [+1] * n_terms
    else:
        n_terms = random.choice([6, 8, 10, 12, 15, 20])
        magnitude = random.choice([500, 1000, 5000, 9999])
        signs = [random.choice([+1, -1]) for _ in range(n_terms)]

    use_decimal = (random.random() < DEC_P) and dif != "simple"
    style = random.choice(["A", "B"])

    if use_decimal:
        terms = [rdec(0, magnitude, places=(1, 2),
                      allow_neg=False) for _ in range(n_terms)]
        signed = [s * t for s, t in zip(signs, terms)]
        ans = sum(signed, Decimal(0))
        parts = [canon_dec(terms[0]) if signs[0] > 0 else f"-{canon_dec(terms[0])}"]
        for s, t in zip(signs[1:], terms[1:]):
            op = "+" if s > 0 else "-"
            parts.append(f"{op} {canon_dec(t)}")
        prob = " ".join(parts)
        step_text = _fmt_addition_chain(signs, terms, lambda x: canon_dec(x, 2), style)
        return prob, canon_dec(ans, 2), {
            "category": "addition_chain_dec",
            "step_text": step_text,
        }

    terms = [rint(0, magnitude) for _ in range(n_terms)]
    signed = [s * t for s, t in zip(signs, terms)]
    ans = sum(signed)
    parts = [str(terms[0]) if signs[0] > 0 else f"-{terms[0]}"]
    for s, t in zip(signs[1:], terms[1:]):
        op = "+" if s > 0 else "-"
        parts.append(f"{op} {t}")
    prob = " ".join(parts)
    step_text = _fmt_addition_chain(signs, terms, str, style)
    return prob, canon_num(ans), {
        "category": "addition_chain",
        "step_text": step_text,
    }


def gen_multiplication_long(dif):
    """Larger-number products: 3-4 digit two-factor, or 3-4 factor 2-digit.

    Closes the coverage gap left by gen_multiplication (|x|≤25, two-factor only).
    Place-value decomposition for big2; left-to-right associativity for multi.
    """
    kind = random.choice(["big2", "multi"])
    style = random.choice(["A", "B"])

    if kind == "big2":
        if dif == "simple":
            digits_a = random.choice([3, 3, 4])
            digits_b = random.choice([2, 3])
        else:
            digits_a = random.choice([3, 4, 4, 5])
            digits_b = random.choice([3, 4])
        lo_a, hi_a = 10 ** (digits_a - 1), 10 ** digits_a - 1
        lo_b, hi_b = 10 ** (digits_b - 1), 10 ** digits_b - 1
        a = rint(lo_a, hi_a)
        b = rint(lo_b, hi_b)
        if (dif != "simple") and random.random() < 0.4:
            a = -a
        if (dif != "simple") and random.random() < 0.4:
            b = -b
        ans = a * b
        prob = f"{a} {rand_mul()} {b}"
        return prob, canon_num(ans), {
            "category": "multiplication_long",
            "step_text": _fmt_mul_big2(a, b, style),
        }

    n_factors = random.choice([3, 4]) if dif == "simple" else random.choice([3, 4, 5])
    if dif == "simple":
        factors = [rint(2, 20) for _ in range(n_factors)]
    else:
        factors = []
        for _ in range(n_factors):
            f = rint(2, 99, exclude_zero=True)
            if random.random() < 0.25:
                f = -f
            factors.append(f)
    ans = 1
    for f in factors:
        ans *= f
    prob = f" {rand_mul()} ".join(str(f) for f in factors)
    return prob, canon_num(ans), {
        "category": "multiplication_multi",
        "step_text": _fmt_mul_multi(factors, style),
    }


def gen_long_arith_chain(dif):
    """5-12 operand chains over + - × ÷ with proper precedence evaluation.

    Closes the coverage gap left by gen_mixed_expression (3 operands only).
    Division operands are constructed so the *running value of the current
    × ÷ cluster* is divisible by the chosen divisor — keeping the final
    answer an exact integer regardless of associativity.
    """
    if dif == "simple":
        n_terms = random.choice([5, 6, 7])
        ops_pool = ["+", "-", "×"]
        magnitude = 50
        allow_neg_operand = False
    else:
        n_terms = random.choice([6, 8, 10, 12])
        ops_pool = ["+", "-", "×", "÷"]
        magnitude = 100
        allow_neg_operand = True

    def maybe_neg(x):
        return -x if (allow_neg_operand and random.random() < 0.3) else x

    terms = [maybe_neg(rint(1, magnitude))]
    ops = []
    cluster_val = terms[0]  # running value of the current × ÷ cluster

    for _ in range(n_terms - 1):
        op = random.choice(ops_pool)
        if op == "÷":
            divisors = [d for d in range(2, 13) if cluster_val % d == 0 and cluster_val != 0]
            if not divisors:
                op = random.choice(["+", "-", "×"])
        if op == "÷":
            d = random.choice(divisors)
            terms.append(d)
            cluster_val //= d
        elif op == "×":
            f = maybe_neg(rint(2, 12))
            terms.append(f)
            cluster_val *= f
        else:
            t = maybe_neg(rint(1, magnitude))
            terms.append(t)
            cluster_val = t
        ops.append(op)

    parts = [str(terms[0])]
    for op, t in zip(ops, terms[1:]):
        rendered_op = rand_mul() if op == "×" else (rand_div() if op == "÷" else op)
        if t < 0:
            parts.append(f"{rendered_op} ({t})")
        else:
            parts.append(f"{rendered_op} {t}")
    prob = " ".join(parts)

    # Compute answer with proper precedence (exact by construction for ÷)
    cur_ops, cur_terms = list(ops), list(terms)
    while True:
        mul_idx = None
        for i, o in enumerate(cur_ops):
            if o in ("×", "÷"):
                mul_idx = i
                break
        if mul_idx is None:
            break
        left, right = cur_terms[mul_idx], cur_terms[mul_idx + 1]
        result = left * right if cur_ops[mul_idx] == "×" else left // right
        cur_terms = cur_terms[:mul_idx] + [result] + cur_terms[mul_idx + 2:]
        cur_ops = cur_ops[:mul_idx] + cur_ops[mul_idx + 1:]
    ans = cur_terms[0]
    for o, t in zip(cur_ops, cur_terms[1:]):
        ans = ans + t if o == "+" else ans - t

    style = random.choice(["A", "B"])
    step_text = _fmt_long_chain(ops, terms, style)

    return prob, canon_num(ans), {
        "category": "long_arith_chain",
        "step_text": step_text,
    }


def gen_geometric_sequence(dif):
    """Find next term or specific term in a geometric sequence."""
    a = rint(1, 8)
    r = random.choice([2, 3, -2, Fraction(1, 2)])
    n = rint(4, 6)

    if isinstance(r, Fraction):
        seq = [Fraction(a)]
        for _ in range(n - 1):
            seq.append(seq[-1] * r)
        seq_str = ", ".join(canon_num(s) for s in seq)
    else:
        seq = [a * (r ** i) for i in range(n)]
        seq_str = ", ".join(str(s) for s in seq)

    op = random.choice(["next", "nth"])
    if op == "next":
        prob = f"Find the next term in the geometric sequence: {seq_str}, ?"
        next_term = seq[-1] * r
        ans = canon_num(next_term)
        step = f"Step: Common ratio = {canon_num(Fraction(r) if not isinstance(r, Fraction) else r)}, multiply last term by ratio"
    else:
        k = rint(7, 12)
        if isinstance(r, Fraction):
            term = Fraction(a) * (r ** (k - 1))
        else:
            term = a * (r ** (k - 1))
        prob = f"Find the {k}th term of the geometric sequence with first term {a} and common ratio {canon_num(Fraction(r) if not isinstance(r, Fraction) else r)}."
        ans = canon_num(term)
        step = f"Step: aₙ = a₁ · r^(n-1) = {a} · ({canon_num(Fraction(r) if not isinstance(r, Fraction) else r)})^{k - 1}"
    return prob, ans, {"category": "geometric_seq", "step_text": step}
