"""Trick / 'negative knowledge' problems — the answer is that the question
is ill-posed, undefined, or has no solution within the given domain.

These are valuable training data: most synthetic math datasets only show
problems with a clean numeric answer, and models trained on them tend to
hallucinate answers for ill-posed questions. Each generator here provides
the reason explicitly so the chain of thought stays informative."""

import math
import random

from .core import StepBuilder, rint


def gen_trick_undefined(dif):
    """A computation that is undefined/indeterminate. RANDOMIZED kind."""
    kind = random.choice([
        "log_nonpos", "div_zero", "tan_pi_over_2", "zero_to_zero",
        "sqrt_neg_real", "arcsin_outofrange", "limit_indeterminate",
    ])
    sb = StepBuilder()

    if kind == "log_nonpos":
        x = -rint(1, 50)
        prob = f"Compute $\\ln({x})$."
        ans = "undefined (ln is not defined for non-positive reals)"
        sb.add(f"The natural log is defined only on (0, ∞).")
        sb.add(f"Since {x} ≤ 0, ln({x}) is undefined in the real numbers.")
        sb.add("(Over ℂ it equals ln|x| + iπ, but that is a different domain.)")
        return prob, ans, {"category": "trick_log_nonpos", "step_text": sb.text()}

    if kind == "div_zero":
        a = rint(-30, 30, exclude_zero=True)
        prob = f"Evaluate {a} / 0."
        ans = "undefined (division by zero)"
        sb.add("Division by zero has no value: there is no x with 0·x = a (a ≠ 0).")
        sb.add("It is *not* a number, *not* infinity, and *not* the same in all systems.")
        return prob, ans, {"category": "trick_div_zero", "step_text": sb.text()}

    if kind == "tan_pi_over_2":
        prob = "Compute $\\tan(\\pi/2)$."
        ans = "undefined (vertical asymptote of tan)"
        sb.add("tan(x) = sin(x)/cos(x); cos(π/2) = 0, so tan(π/2) is undefined.")
        sb.add("The function has a vertical asymptote at π/2 + kπ for integer k.")
        return prob, ans, {"category": "trick_tan_pi2", "step_text": sb.text()}

    if kind == "zero_to_zero":
        prob = "What is $0^0$?"
        ans = "indeterminate (no universally agreed-upon value)"
        sb.add("0^x = 0 for x > 0, but x^0 = 1 for x > 0; the two limits disagree at (0,0).")
        sb.add("In combinatorics 0^0 is conventionally taken as 1; in analysis it is left undefined.")
        sb.add("So the answer depends on context — there is no single correct numeric value.")
        return prob, ans, {"category": "trick_zero_zero", "step_text": sb.text()}

    if kind == "sqrt_neg_real":
        n = -rint(1, 50)
        prob = f"Find a real number x with x² = {n}."
        ans = "no real solution"
        sb.add("x² ≥ 0 for every real x.")
        sb.add(f"Since {n} < 0, no real x satisfies x² = {n}.")
        sb.add(f"In ℂ the solutions are ±i·√{abs(n)}, but that requires extending to complex numbers.")
        return prob, ans, {"category": "trick_sqrt_neg", "step_text": sb.text()}

    if kind == "arcsin_outofrange":
        v = random.choice([2, 3, -2, -3, 5, -4])
        prob = f"Compute $\\arcsin({v})$."
        ans = "undefined (input outside [−1, 1])"
        sb.add("arcsin's natural domain is [−1, 1], because sin(x) ∈ [−1, 1].")
        sb.add(f"Since {v} ∉ [−1, 1], arcsin({v}) is undefined as a real-valued function.")
        return prob, ans, {"category": "trick_arcsin_oor", "step_text": sb.text()}

    # limit_indeterminate
    prob = "What is $\\lim_{x \\to 0} \\frac{0}{x}$ — and what is $\\lim_{x \\to 0} \\frac{x}{x}$? Are they the same kind of expression?"
    ans = "First limit = 0; second limit = 1; both write as '0/0' but are not the same."
    sb.add("Both expressions evaluate to the indeterminate form 0/0 by direct substitution.")
    sb.add("Limit of 0/x as x→0: numerator stays 0, so the quotient is 0; limit = 0.")
    sb.add("Limit of x/x as x→0: simplifies to 1 for x ≠ 0; limit = 1.")
    sb.add("Moral: '0/0' is indeterminate — its value depends on the rate at which the parts approach 0.")
    return prob, ans, {"category": "trick_indeterminate", "step_text": sb.text()}


def gen_trick_no_solution(dif):
    """An equation/system with no solution in the implied domain."""
    kind = random.choice(["quad_no_real", "linear_inconsistent",
                          "abs_negative", "int_no_solution"])
    sb = StepBuilder()

    if kind == "quad_no_real":
        a = rint(1, 5)
        b = rint(-3, 3)
        # Make discriminant strictly negative
        c = b * b // (4 * a) + rint(2, 8)
        prob = f"Find all real x with {a}x² + {b}x + {c} = 0."
        disc = b * b - 4 * a * c
        ans = f"no real solution (discriminant = {disc} < 0)"
        sb.add(f"Discriminant Δ = b² − 4ac = {b}² − 4·{a}·{c} = {disc}.")
        sb.add(f"Δ < 0, so the quadratic has no real roots (only complex ones).")
        return prob, ans, {"category": "trick_quad_no_real", "step_text": sb.text()}

    if kind == "linear_inconsistent":
        a, b = rint(1, 5), rint(1, 5)
        c1 = rint(-10, 10)
        c2 = c1 + rint(1, 8)  # different RHS
        prob = (f"Solve the system:  {a}x + {b}y = {c1};  {a}x + {b}y = {c2}.")
        ans = "no solution (inconsistent system)"
        sb.add(f"The two equations have identical left-hand sides but different right-hand sides ({c1} ≠ {c2}).")
        sb.add("Geometrically, the lines are parallel and distinct — they never intersect.")
        return prob, ans, {"category": "trick_lin_inconsistent", "step_text": sb.text()}

    if kind == "abs_negative":
        a = rint(1, 6)
        b = rint(-10, 10)
        c = -rint(1, 12)
        prob = f"Solve |{a}x + {b}| = {c}."
        ans = "no solution"
        sb.add("Absolute value is always ≥ 0.")
        sb.add(f"The equation requires |…| = {c} < 0, which is impossible.")
        return prob, ans, {"category": "trick_abs_negative", "step_text": sb.text()}

    # int_no_solution: 2x = odd
    odd = 2 * rint(-10, 10) + 1
    prob = f"Find an integer x with 2x = {odd}."
    ans = "no integer solution"
    sb.add(f"2x is even for every integer x.")
    sb.add(f"{odd} is odd, so no integer x satisfies 2x = {odd}.")
    sb.add(f"(Over ℝ the unique solution is x = {odd}/2, but that is not an integer.)")
    return prob, ans, {"category": "trick_int_no_solution", "step_text": sb.text()}


def gen_trick_ambiguous(dif):
    """A wording trap or ambiguous-by-design problem. The 'answer' explains the ambiguity."""
    kind = random.choice([
        "famous_ambig",
        "implicit_assumption",
        "convention_dependent",
        "principal_root",
        "natural_number_zero",
        "monty_hall",
        "two_child",
        "real_vs_complex_roots",
        "negative_zero",
        "point_nine_repeating",
    ])
    sb = StepBuilder()

    if kind == "famous_ambig":
        n1, n2 = rint(2, 9), rint(2, 9)
        prob = (f"Evaluate {n1} ÷ 2({n2}). Be explicit about whether your answer assumes "
                f"implicit-multiplication binds tighter than '÷'.")
        v1 = (n1 / 2) * n2
        v2 = n1 / (2 * n2)
        ans = f"{v1} (left-to-right) or {v2} (implicit-mul binds tighter) — the expression is ambiguous"
        sb.add("The expression mixes ÷ and implicit multiplication; common conventions disagree.")
        sb.add(f"Strict left-to-right: {n1} ÷ 2 · {n2} = {n1/2}·{n2} = {v1}.")
        sb.add(f"Implicit-multiplication first: {n1} ÷ (2·{n2}) = {n1}/{2*n2} = {v2}.")
        sb.add("A well-formed problem would parenthesise to remove the ambiguity.")
        return prob, ans, {"category": "trick_famous_ambig", "step_text": sb.text()}

    if kind == "implicit_assumption":
        n_flips = rint(8, 15)
        prob = (f"A coin was flipped {n_flips} times and came up heads each time. "
                f"What is the probability of heads on the next flip?")
        ans = "1/2 if the coin is fair (independent flips); cannot be inferred without more info"
        sb.add("If the coin is *known* fair, flips are independent ⇒ P(H) = 1/2.")
        sb.add("If fairness is not given, the streak is evidence of bias and the posterior P(H) > 1/2.")
        sb.add("The gambler's fallacy is to expect P(H) < 1/2 to 'balance things out'.")
        return prob, ans, {"category": "trick_implicit_assumption", "step_text": sb.text()}

    if kind == "convention_dependent":
        prob = "Is 1 a prime number? Justify your answer using the standard modern definition."
        ans = "No — 1 is not prime by the modern definition (a prime has exactly two distinct positive divisors)."
        sb.add("Modern definition: p > 1, divisors are exactly 1 and p.")
        sb.add("Excluding 1 preserves the uniqueness of prime factorisation.")
        sb.add("(Some 19th-century texts called 1 prime; that convention has been abandoned.)")
        return prob, ans, {"category": "trick_convention", "step_text": sb.text()}

    if kind == "principal_root":
        n = rint(2, 12)
        prob = f"What is $\\sqrt{{{n*n}}}$? (Note: ask about the radical, not the equation x² = {n*n}.)"
        ans = f"{n} (the principal — i.e. non-negative — square root)"
        sb.add(f"The radical √· denotes the *principal* (non-negative) square root by definition.")
        sb.add(f"So √{n*n} = {n}, not ±{n}.")
        sb.add(f"The equation x² = {n*n} is different — it has both x = {n} and x = −{n} as solutions.")
        return prob, ans, {"category": "trick_principal_root", "step_text": sb.text()}

    if kind == "natural_number_zero":
        prob = "What is the smallest natural number?"
        ans = "0 or 1 — depends on the convention (ISO 80000-2 includes 0; many number-theory texts start at 1)."
        sb.add("ℕ is defined either as {0, 1, 2, …} or as {1, 2, …} depending on the source.")
        sb.add("ISO 80000-2 and most computer-science / set-theory contexts include 0.")
        sb.add("Most number-theory and analysis texts (and Peano's original axioms) exclude 0.")
        sb.add("A well-posed problem must specify which convention.")
        return prob, ans, {"category": "trick_natural_zero", "step_text": sb.text()}

    if kind == "monty_hall":
        prob = ("In a 3-door Monty Hall game, the contestant picks door 1; the host (knowing where the prize is) "
                "opens door 3 to reveal a goat. Should the contestant switch to door 2?")
        ans = "Yes — switching wins with probability 2/3; staying wins with probability 1/3."
        sb.add("Initial choice is right with probability 1/3, so the prize is *not* behind door 1 with probability 2/3.")
        sb.add("In that 2/3 case, the host's reveal is forced (he must open the only remaining goat door), pointing at the prize.")
        sb.add("So switching wins with probability 2/3 — counterintuitive but consistent with conditional probability.")
        return prob, ans, {"category": "trick_monty_hall", "step_text": sb.text()}

    if kind == "two_child":
        prob = ("A family has two children. You learn that at least one of them is a boy. "
                "What is the probability that both are boys?")
        ans = "1/3 under the 'at least one boy' interpretation; 1/2 under 'a specific child is a boy' — phrasing-dependent"
        sb.add("Sample space {BB, BG, GB, GG}, each with probability 1/4.")
        sb.add("'At least one is a boy' rules out only GG, leaving {BB, BG, GB}; P(BB | ≥1 boy) = 1/3.")
        sb.add("If instead you learn 'the elder child is a boy', the space is {BB, BG} and P = 1/2.")
        sb.add("The English statement is ambiguous about which conditioning event is meant.")
        return prob, ans, {"category": "trick_two_child", "step_text": sb.text()}

    if kind == "real_vs_complex_roots":
        n = random.choice([3, 5, 7])
        prob = f"How many roots does the polynomial $x^{n} - 1 = 0$ have?"
        ans = f"1 real root (x = 1); {n} complex roots in total (the {n}-th roots of unity)"
        sb.add(f"Over ℝ: x^{n} − 1 = (x − 1)·(x^{n-1} + x^{n-2} + … + 1); the second factor has no real roots when n is odd > 1.")
        sb.add(f"So x^{n} − 1 has exactly 1 real root, namely x = 1.")
        sb.add(f"Over ℂ: by the fundamental theorem of algebra, there are {n} roots — the {n}-th roots of unity.")
        sb.add("'How many roots' is ambiguous unless the field is specified.")
        return prob, ans, {"category": "trick_real_vs_complex_roots", "step_text": sb.text()}

    if kind == "negative_zero":
        prob = "In IEEE-754 floating-point, is `-0.0 == 0.0` true? Are they the same number?"
        ans = "`==` returns True; but they are distinct bit patterns and `1/0.0 = +∞`, `1/-0.0 = -∞`."
        sb.add("IEEE-754 has two zeros: +0 (sign bit 0) and −0 (sign bit 1).")
        sb.add("The equality operator `==` returns True for `+0 == -0` to satisfy mathematical intuition.")
        sb.add("But `1/+0 = +∞` while `1/-0 = -∞`, so they're observably different in some contexts.")
        sb.add("The 'same number' question is therefore phrasing-dependent.")
        return prob, ans, {"category": "trick_negative_zero", "step_text": sb.text()}

    # point_nine_repeating
    prob = "Is $0.999\\ldots$ (with infinitely many 9s) equal to 1, or strictly less than 1?"
    ans = "Equal to 1 — the standard real-number interpretation makes them the same number, not 'just below 1'."
    sb.add("Let x = 0.999…; then 10x = 9.999… = 9 + x, so 9x = 9, hence x = 1.")
    sb.add("More rigorously: 0.999… is defined as the limit of the partial sums 0.9 + 0.09 + …, "
           "a convergent geometric series with sum 9/(10−1)·(1/10)/(1) = 1.")
    sb.add("So in the standard real numbers, 0.999… = 1 exactly. (Non-standard analysis can give a different answer, "
           "but only under non-standard conventions about what real numbers are.)")
    return prob, ans, {"category": "trick_point_nine", "step_text": sb.text()}
