"""Adversarial / common-error problems — easy-looking questions where naive
evaluation produces the wrong answer. Each output documents the trap explicitly."""

import random

from .core import StepBuilder, rint


def gen_adversarial_arithmetic(dif):
    """Numerical 'gotchas' where the naive answer is wrong."""
    kind = random.choice([
        "unary_power", "float_eq", "mod_neg", "intdiv_neg",
        "sqrt_x2", "ineq_neg_mul", "discount_stack", "avg_of_avg",
    ])
    sb = StepBuilder()

    if kind == "unary_power":
        n = rint(2, 9)
        prob = f"Compute −{n}². (Be careful with operator precedence.)"
        ans = str(-(n ** 2))
        sb.add(f"−n² is parsed as −(n²), not (−n)².")
        sb.add(f"So −{n}² = −({n}²) = {-(n**2)}.")
        sb.add(f"If you wanted (−{n})², that would be {n*n} — a different expression.")
        return prob, ans, {"category": "adversarial_unary_power", "step_text": sb.text()}

    if kind == "float_eq":
        prob = "In IEEE-754 double precision, what is the exact stored value of 0.1 + 0.2?"
        ans = "0.30000000000000004"
        sb.add("Neither 0.1 nor 0.2 has a finite binary representation.")
        sb.add("Their stored values differ slightly from the decimals you typed.")
        sb.add("Adding them yields 0.30000000000000004, which is why 0.1 + 0.2 == 0.3 returns False.")
        return prob, ans, {"category": "adversarial_float_eq", "step_text": sb.text()}

    if kind == "mod_neg":
        a = -rint(1, 30)
        b = rint(2, 12)
        py = a % b
        c_style = a - (a // b) * b if (a < 0 and a % b != 0) else py
        # In mathematics / Python, result lies in [0, b). In C/Java, it can be negative.
        prob = (f"Compute {a} mod {b} using the mathematician's convention "
                f"(result in [0, {b})).")
        ans = str(py)
        sb.add(f"Math convention: pick the unique integer q so that {a} − q·{b} ∈ [0, {b}).")
        sb.add(f"That gives {a} − ({a // b})·{b} = {py}, in [0, {b}). So {a} mod {b} = {py}.")
        sb.add(f"(C / Java would return {a - (-(-a // b)) * b}, which keeps the sign of the dividend.)")
        return prob, ans, {"category": "adversarial_mod_neg", "step_text": sb.text()}

    if kind == "intdiv_neg":
        a = -rint(7, 25)
        b = rint(2, 6)
        py_div = a // b
        c_div = -((-a) // b)
        prob = (f"Compute the integer quotient of {a} divided by {b}. "
                f"Specify both Python's floor-division result and C-style truncation toward zero.")
        ans = f"Python ({a} // {b}) = {py_div}; C/Java truncation = {c_div}"
        sb.add(f"Python's // floors toward −∞: ⌊{a}/{b}⌋ = {py_div}.")
        sb.add(f"C/Java truncate toward 0: trunc({a}/{b}) = {c_div}.")
        sb.add(f"Difference: when the true quotient is non-integer and negative, the two diverge by 1.")
        return prob, ans, {"category": "adversarial_intdiv_neg", "step_text": sb.text()}

    if kind == "sqrt_x2":
        x = -rint(2, 10)
        prob = f"What is √(x²) when x = {x}?"
        ans = str(abs(x))
        sb.add("√(x²) = |x|, not x.")
        sb.add(f"Here x = {x} < 0, so √(x²) = |{x}| = {abs(x)}.")
        sb.add("It is *not* −{0}; the square root function returns the non-negative root by definition.".format(abs(x)))
        return prob, ans, {"category": "adversarial_sqrt_x2", "step_text": sb.text()}

    if kind == "ineq_neg_mul":
        a = rint(2, 9)
        b = rint(-15, -3)
        c = rint(-20, 20)
        prob = f"Solve the inequality −{a}x + {b} > {c}."
        ans_op = "<"
        sb.add(f"Subtract {b} from both sides: −{a}x > {c - b}.")
        sb.add(f"Divide both sides by −{a}; this **flips** the inequality.")
        from fractions import Fraction
        rhs = Fraction(c - b, -a)
        sb.add(f"x {ans_op} {rhs.numerator}/{rhs.denominator if rhs.denominator != 1 else ''}".rstrip("/"))
        ans = f"x {ans_op} {rhs}"
        return prob, ans, {"category": "adversarial_ineq", "step_text": sb.text()}

    if kind == "discount_stack":
        d1 = random.choice([10, 20, 25, 30])
        d2 = random.choice([10, 15, 20])
        base = rint(50, 500)
        from fractions import Fraction
        new = Fraction(base) * Fraction(100 - d1, 100) * Fraction(100 - d2, 100)
        naive_pct = d1 + d2
        actual_pct = 100 - 100 * (1 - d1 / 100) * (1 - d2 / 100)
        prob = (f"An item costs ${base}. It is marked down by {d1}%, then a further {d2}% off "
                f"the discounted price. What is the final price? "
                f"Why is the total discount NOT {naive_pct}%?")
        ans = f"${float(new):.2f}"
        sb.add(f"After first discount: {base}·{(100-d1)}% = {float(Fraction(base)*Fraction(100-d1,100)):.2f}.")
        sb.add(f"After second discount: that · {(100-d2)}% = {float(new):.2f}.")
        sb.add(f"Effective discount: 1 − (1−{d1}/100)(1−{d2}/100) = {actual_pct:.1f}%, not {naive_pct}%.")
        return prob, ans, {"category": "adversarial_discount_stack", "step_text": sb.text()}

    # avg_of_avg
    k1 = rint(20, 60)
    k2 = rint(20, 60)
    avg1 = rint(60, 90)
    avg2 = rint(60, 90)
    naive_avg = (avg1 + avg2) / 2
    weighted = (k1 * avg1 + k2 * avg2) / (k1 + k2)
    prob = (f"Class A has {k1} students with average score {avg1}; "
            f"Class B has {k2} students with average score {avg2}. "
            f"What is the combined average? "
            f"Why is it generally not {naive_avg}?")
    ans = f"{weighted:.2f}"
    sb.add(f"Total points = {k1}·{avg1} + {k2}·{avg2} = {k1*avg1 + k2*avg2}.")
    sb.add(f"Total students = {k1 + k2}.")
    sb.add(f"Combined average = {k1*avg1 + k2*avg2}/{k1+k2} = {weighted:.2f}.")
    sb.add(f"Simple average of averages ({naive_avg}) is correct only when class sizes are equal.")
    return prob, ans, {"category": "adversarial_avg_of_avg", "step_text": sb.text()}
