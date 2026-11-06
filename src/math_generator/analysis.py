"""Trigonometry, calculus, series, ODE, conics, functions, log/exp.

Several generators here use SymPy for ground truth so they qualify as
fully randomized (hardcoded level 0)."""

import math
import random
from fractions import Fraction

import sympy as sp

from .core import canon_num, fmt_poly, rint


# ─────────────────────────── TRIGONOMETRY ───────────────────────────

# Lookup table of exact trig values used by gen_trig_exact.
TRIG_TABLE = {
    (0, "0"):          ("0",          "1",            "0"),
    (30, "\\pi/6"):    ("1/2",        "\\sqrt{3}/2",  "\\sqrt{3}/3"),
    (45, "\\pi/4"):    ("\\sqrt{2}/2", "\\sqrt{2}/2", "1"),
    (60, "\\pi/3"):    ("\\sqrt{3}/2", "1/2",         "\\sqrt{3}"),
    (90, "\\pi/2"):    ("1",          "0",            "undefined"),
    (120, "2\\pi/3"):  ("\\sqrt{3}/2", "-1/2",        "-\\sqrt{3}"),
    (135, "3\\pi/4"):  ("\\sqrt{2}/2", "-\\sqrt{2}/2", "-1"),
    (150, "5\\pi/6"):  ("1/2",        "-\\sqrt{3}/2", "-\\sqrt{3}/3"),
    (180, "\\pi"):     ("0",          "-1",           "0"),
    (210, "7\\pi/6"):  ("-1/2",       "-\\sqrt{3}/2", "\\sqrt{3}/3"),
    (225, "5\\pi/4"):  ("-\\sqrt{2}/2", "-\\sqrt{2}/2", "1"),
    (240, "4\\pi/3"):  ("-\\sqrt{3}/2", "-1/2",       "\\sqrt{3}"),
    (270, "3\\pi/2"):  ("-1",         "0",            "undefined"),
    (300, "5\\pi/3"):  ("-\\sqrt{3}/2", "1/2",        "-\\sqrt{3}"),
    (315, "7\\pi/4"):  ("-\\sqrt{2}/2", "\\sqrt{2}/2", "-1"),
    (330, "11\\pi/6"): ("-1/2",       "\\sqrt{3}/2",  "-\\sqrt{3}/3"),
}


def gen_trig_exact(dif):
    """Pick a (deg, rad) angle from the lookup table and ask sin/cos/tan. Level 1."""
    key = random.choice(list(TRIG_TABLE.keys()))
    deg, rad = key
    sin_v, cos_v, tan_v = TRIG_TABLE[key]
    func = random.choice(["sin", "cos"] if tan_v == "undefined" else ["sin", "cos", "tan"])
    use_rad = random.random() < 0.5
    arg = f"{rad}" if use_rad else f"{deg}°"
    prob = f"Find the exact value of $\\{func}({arg})$."
    ans = f"${ {'sin': sin_v, 'cos': cos_v, 'tan': tan_v}[func] }$"
    return prob, ans, {"category": "trig_exact", "step_text": f"Step: read the exact value off the unit circle at {arg}."}


def gen_trig_identity_sympy(dif):
    """Random trig simplification computed by SymPy. RANDOMIZED."""
    x = sp.Symbol("x")
    n = random.choice([1, 1, 2, 3])
    candidates = [
        (sp.sin(n * x) ** 2 + sp.cos(n * x) ** 2,             "Pythagorean: sin² + cos² = 1"),
        (1 - sp.cos(n * x) ** 2,                               "1 − cos² = sin²"),
        (1 - sp.sin(n * x) ** 2,                               "1 − sin² = cos²"),
        (sp.sin(n * x) / sp.cos(n * x),                        "definition of tan"),
        (1 + sp.tan(n * x) ** 2,                               "Pythagorean (sec form)"),
        (sp.sec(n * x) ** 2 - 1,                               "sec² − 1 = tan²"),
        (2 * sp.sin(n * x) * sp.cos(n * x),                    "double-angle: 2 sinθ cosθ = sin(2θ)"),
        (sp.cos(n * x) ** 2 - sp.sin(n * x) ** 2,              "double-angle: cos² − sin² = cos(2θ)"),
        (sp.sin(-n * x),                                       "sine is odd"),
        (sp.cos(-n * x),                                       "cosine is even"),
        (sp.cos(n * x) * sp.tan(n * x),                        "cosθ · tanθ = sinθ"),
    ]
    e, why = random.choice(candidates)
    ans = sp.trigsimp(e)
    prob = f"Simplify: ${sp.latex(e)}$"
    return prob, f"${sp.latex(ans)}$", {"category": "trig_identity", "step_text": f"Step: {why}."}


def gen_trig_equation_sympy(dif):
    """Solve a random trig equation in [0, 2π) via SymPy. RANDOMIZED."""
    x = sp.Symbol("x", real=True)
    fn = random.choice([sp.sin, sp.cos, sp.tan])
    if fn is sp.tan:
        rhs = random.choice([0, 1, -1, sp.sqrt(3), -sp.sqrt(3),
                             sp.sqrt(3) / 3, -sp.sqrt(3) / 3])
    else:
        rhs = random.choice([0, 1, -1,
                             sp.Rational(1, 2), -sp.Rational(1, 2),
                             sp.sqrt(2) / 2, -sp.sqrt(2) / 2,
                             sp.sqrt(3) / 2, -sp.sqrt(3) / 2])
    eq = sp.Eq(fn(x), rhs)
    sols = sp.solveset(eq, x, domain=sp.Interval(0, 2 * sp.pi, right_open=True))
    try:
        sol_list = sorted(list(sols), key=lambda s: float(s))
    except Exception:
        sol_list = list(sols.args) if hasattr(sols, "args") else [sols]
    if not sol_list:
        ans = "no solution in $[0, 2\\pi)$"
    else:
        ans = ", ".join(f"x = {sp.latex(s)}" for s in sol_list)
    prob = f"Solve for $x \\in [0, 2\\pi)$: ${sp.latex(eq.lhs)} = {sp.latex(rhs)}$"
    step = "Step: identify the reference angle, then place each solution in the correct quadrant(s)."
    return prob, ans, {"category": "trig_equation", "step_text": step}


def gen_trig_inverse_sympy(dif):
    """Evaluate a random inverse-trig at a value with a clean closed form. RANDOMIZED."""
    x = sp.Symbol("x")
    triples = [(sp.asin, [0, 1, -1, sp.Rational(1, 2), -sp.Rational(1, 2),
                          sp.sqrt(2) / 2, -sp.sqrt(2) / 2,
                          sp.sqrt(3) / 2, -sp.sqrt(3) / 2]),
               (sp.acos, [0, 1, -1, sp.Rational(1, 2), -sp.Rational(1, 2),
                          sp.sqrt(2) / 2, -sp.sqrt(2) / 2,
                          sp.sqrt(3) / 2, -sp.sqrt(3) / 2]),
               (sp.atan, [0, 1, -1, sp.sqrt(3), -sp.sqrt(3),
                          sp.sqrt(3) / 3, -sp.sqrt(3) / 3])]
    fn, choices = random.choice(triples)
    arg = random.choice(choices)
    val = sp.simplify(fn(arg))
    name = {sp.asin: "arcsin", sp.acos: "arccos", sp.atan: "arctan"}[fn]
    prob = f"Evaluate $\\{name}({sp.latex(arg)})$."
    return prob, f"${sp.latex(val)}$", {"category": "trig_inverse", "step_text": f"Step: read off from the unit circle."}


# ─────────────────────────── CALCULUS ───────────────────────────

def gen_derivative_basic_sympy(dif):
    """Random derivative; SymPy-computed. RANDOMIZED."""
    x = sp.Symbol("x")
    a = random.randint(1, 5)
    n = random.randint(1, 6)
    forms = [
        a * x ** n,
        a * sp.sin(n * x),
        a * sp.cos(n * x),
        sp.exp(a * x),
        sp.log(a * x + 1),
        a * sp.tan(x),
        x * sp.exp(x),
        x ** n * sp.sin(x),
        a * x ** n + random.randint(-5, 5) * x + random.randint(-5, 5),
        sp.sqrt(a * x + 1),
    ]
    f = random.choice(forms)
    df = sp.diff(f, x)
    prob = f"Differentiate $f(x) = {sp.latex(f)}$."
    ans = f"$f'(x) = {sp.latex(df)}$"
    step = "Step: apply power rule, chain rule, and product rule as needed."
    return prob, ans, {"category": "derivative_basic", "step_text": step}


def gen_derivative_poly(dif):
    """Differentiate a concrete random polynomial."""
    degree = 3 if dif == "medium" else random.choice([3, 4])
    coeffs = [rint(-5, 5) for _ in range(degree + 1)]
    if coeffs[0] == 0: coeffs[0] = rint(1, 5)
    d_coeffs = [coeffs[i] * (degree - i) for i in range(degree)]
    prob = f"Differentiate: $f(x) = {fmt_poly(coeffs)}$"
    ans = f"$f'(x) = {fmt_poly(d_coeffs)}$"
    step = "Step: apply the power rule term by term."
    return prob, ans, {"category": "derivative_poly", "step_text": step}


def gen_integral_indefinite_sympy(dif):
    """Random indefinite integral computed by SymPy. RANDOMIZED."""
    x = sp.Symbol("x")
    a = random.randint(1, 5)
    n = random.randint(1, 5)
    forms = [
        a * x ** n,
        a * sp.sin(n * x),
        a * sp.cos(n * x),
        sp.exp(a * x),
        a / x,
        sp.exp(a * x) * a,
        a * x ** n + random.randint(-4, 4),
        1 / (1 + x ** 2),
        1 / sp.sqrt(1 - x ** 2),
    ]
    f = random.choice(forms)
    F = sp.integrate(f, x)
    prob = f"Compute $\\int {sp.latex(f)} \\, dx$."
    ans = f"${sp.latex(F)} + C$"
    step = "Step: apply reverse power rule / known antiderivatives / u-substitution."
    return prob, ans, {"category": "integral_indefinite", "step_text": step}


def gen_integral_definite(dif):
    """Definite integral with a clean rational answer."""
    n = rint(1, 4)
    a, b = sorted([rint(-4, 4), rint(-4, 4)])
    if a == b:
        b = a + rint(1, 4)
    coef = rint(1, 5)
    val = Fraction(coef * (b ** (n + 1) - a ** (n + 1)), n + 1)
    prob = f"Evaluate: $\\int_{{{a}}}^{{{b}}} {coef if coef != 1 else ''}x^{{{n}}} \\, dx$"
    ans = canon_num(val)
    step = (f"Step: antiderivative is $\\frac{{{coef}}}{{{n + 1}}} x^{{{n + 1}}}$; "
            f"evaluate at the endpoints and subtract.")
    return prob, ans, {"category": "integral_definite", "step_text": step}


def gen_limit_sympy(dif):
    """Random limit problem evaluated by SymPy. RANDOMIZED."""
    x = sp.Symbol("x")
    kind = random.choice(["polynomial_factor", "trig_zero", "exp_zero",
                          "infinity", "log_zero", "lhopital"])
    if kind == "polynomial_factor":
        a = random.randint(1, 5)
        n = random.randint(2, 4)
        f = (x ** n - a ** n) / (x - a)
        point = a
    elif kind == "trig_zero":
        a = random.randint(1, 4)
        f = random.choice([sp.sin(a * x) / x, (1 - sp.cos(a * x)) / x,
                           (1 - sp.cos(a * x)) / x ** 2, sp.tan(a * x) / x])
        point = 0
    elif kind == "exp_zero":
        a = random.randint(1, 4)
        f = (sp.exp(a * x) - 1) / x
        point = 0
    elif kind == "infinity":
        deg_num = random.randint(1, 3)
        deg_den = random.randint(1, 3)
        c1 = random.randint(1, 5)
        c2 = random.randint(1, 5)
        f = (c1 * x ** deg_num + random.randint(-3, 3)) / (c2 * x ** deg_den + random.randint(-3, 3))
        point = sp.oo
    elif kind == "log_zero":
        a = random.randint(1, 4)
        f = sp.log(1 + a * x) / x
        point = 0
    else:
        a = random.randint(1, 3)
        b = random.randint(1, 3)
        f = (sp.exp(a * x) - 1) / sp.sin(b * x)
        point = 0
    val = sp.limit(f, x, point)
    point_latex = sp.latex(point)
    prob = f"Compute $\\lim_{{x \\to {point_latex}}} {sp.latex(f)}$."
    return prob, f"${sp.latex(val)}$", {"category": "limit", "step_text": "Step: apply L'Hôpital, factoring, or a known limit form."}


# ─────────────────────────── LOG / EXP ───────────────────────────

def gen_logarithm(dif):
    """Compute exact logarithm values, change of base, log equations and properties."""
    if dif == "medium":
        entries = [
            (2, 8, 3), (2, 16, 4), (2, 32, 5), (2, 1, 0), (2, 2, 1),
            (3, 9, 2), (3, 27, 3), (3, 1, 0),
            (10, 100, 2), (10, 1000, 3), (10, 1, 0), (10, 10, 1),
            (5, 25, 2), (5, 125, 3),
        ]
        base, arg, ans = random.choice(entries)
        prob = f"Compute $\\log_{{{base}}}({arg})$."
        return prob, str(ans), {"category": "logarithm",
                                "step_text": f"Step: {base}^{ans} = {arg}."}
    sub = random.choice(["change_of_base", "log_equation", "log_property"])
    if sub == "change_of_base":
        base, arg = random.choice([(4, 32), (8, 64), (9, 27), (4, 8), (8, 4)])
        val = math.log(arg) / math.log(base)
        val_f = Fraction(val).limit_denominator(10)
        prob = f"Compute $\\log_{{{base}}}({arg})$ using change of base."
        return prob, canon_num(val_f), {"category": "log_change_base",
                                        "step_text": f"Step: log_{base}({arg}) = ln({arg})/ln({base})."}
    if sub == "log_equation":
        b = random.choice([2, 3, 10])
        k = rint(1, 4)
        x = b ** k
        prob = f"Solve for $x$: $\\log_{{{b}}}(x) = {k}$"
        return prob, str(x), {"category": "log_equation",
                              "step_text": f"Step: x = {b}^{k} = {x}."}
    props = [
        ("$\\log_b(xy) = ?$", "$\\log_b(x) + \\log_b(y)$", "Product rule"),
        ("$\\log_b(x/y) = ?$", "$\\log_b(x) - \\log_b(y)$", "Quotient rule"),
        ("$\\log_b(x^n) = ?$", "$n \\log_b(x)$", "Power rule"),
        ("$\\log_b(b) = ?$", "$1$", "log of base = 1"),
        ("$\\log_b(1) = ?$", "$0$", "log of 1 = 0"),
    ]
    prob, ans, step = random.choice(props)
    return prob, ans, {"category": "log_property", "step_text": f"Step: {step}."}


def gen_exponential(dif):
    """Exponential equations / growth / decay."""
    sub = random.choice(["solve_exp", "growth", "decay"] if dif == "hard" else ["solve_exp"])
    if sub == "solve_exp":
        b = random.choice([2, 3, 5, 10])
        k = rint(1, 6)
        c = b ** k
        prob = f"Solve for $x$: ${b}^x = {c}$"
        return prob, str(k), {"category": "exponential_eq",
                              "step_text": f"Step: {c} = {b}^{k}, so x = {k}."}
    if sub == "growth":
        P0 = rint(100, 1000) // 100 * 100
        r = random.choice([0.05, 0.10, 0.20])
        t = rint(1, 5)
        A = P0 * ((1 + r) ** t)
        prob = (f"A population starts at {P0} and grows at {int(r * 100)}% per year. "
                f"What is it after {t} year{'s' if t > 1 else ''}?")
        ans = f"{A:.2f}".rstrip("0").rstrip(".")
        return prob, ans, {"category": "exp_growth", "step_text": f"Step: A = {P0}(1+{r})^{t}."}
    P0 = rint(100, 1000) // 100 * 100
    r = random.choice([0.05, 0.10, 0.20])
    t = rint(1, 5)
    A = P0 * ((1 - r) ** t)
    prob = (f"A quantity starts at {P0} and decays at {int(r * 100)}% per period. "
            f"What is the value after {t} period{'s' if t > 1 else ''}?")
    ans = f"{A:.4f}".rstrip("0").rstrip(".")
    return prob, ans, {"category": "exp_decay", "step_text": f"Step: A = {P0}(1−{r})^{t}."}


# ─────────────────────────── SERIES ───────────────────────────

def gen_arith_series_sum(dif):
    a = rint(-10, 20)
    d = rint(1, 6)
    n = rint(5, 25)
    last = a + (n - 1) * d
    s = n * (a + last) // 2
    prob = (f"Find the sum of the first {n} terms of the arithmetic sequence "
            f"with first term {a} and common difference {d}.")
    return prob, str(s), {"category": "arith_series_sum",
                          "step_text": f"Step: S_n = n(a₁ + aₙ)/2 = {n}·({a}+{last})/2 = {s}."}


def gen_geom_series_sum(dif):
    a = rint(1, 10)
    if random.random() < 0.5:
        r = random.choice([2, 3, Fraction(1, 2), Fraction(1, 3)])
        n = rint(3, 8)
        s = Fraction(a) * (1 - Fraction(r) ** n) / (1 - Fraction(r))
        prob = (f"Find the sum of the first {n} terms of a geometric series "
                f"with first term {a} and common ratio {canon_num(Fraction(r))}.")
        return prob, canon_num(s), {"category": "geom_series_sum",
                                    "step_text": "Step: S_n = a(1 − rⁿ)/(1 − r)."}
    r = random.choice([Fraction(1, 2), Fraction(1, 3), Fraction(1, 4),
                       Fraction(2, 3), Fraction(-1, 2)])
    s = Fraction(a) / (1 - r)
    prob = (f"Find the sum of the infinite geometric series with first term {a} "
            f"and common ratio {canon_num(r)}.")
    return prob, canon_num(s), {"category": "geom_series_sum",
                                "step_text": "Step: |r|<1 so S = a/(1−r)."}


def gen_taylor_series_sympy(dif):
    """Random Maclaurin / Taylor series via SymPy. RANDOMIZED."""
    x = sp.Symbol("x")
    a = random.choice([1, 1, 2, 3])
    fns = [
        sp.exp(a * x), sp.sin(a * x), sp.cos(a * x), sp.log(1 + a * x),
        1 / (1 - a * x), sp.sqrt(1 + a * x), sp.atan(a * x),
        sp.sinh(a * x), sp.cosh(a * x),
    ]
    if dif == "hard" and random.random() < 0.4:
        f1 = random.choice(fns)
        if random.random() < 0.5:
            f = f1 * random.choice([sp.exp(x), sp.cos(x), 1 + x])
        else:
            f = f1 + random.choice([sp.sin(x), x ** 2, sp.exp(-x)])
    else:
        f = random.choice(fns)
    n = random.randint(4, 7)
    point = random.choice([0, 0, 0, 1])
    try:
        s = sp.series(f, x, point, n).removeO()
    except Exception:
        s = sp.series(f, x, 0, n).removeO()
    if point == 0:
        prob = f"Find the Maclaurin series of $f(x) = {sp.latex(f)}$ up to $O(x^{n})$."
        step = f"Step: Taylor expansion at 0 to order {n - 1}."
    else:
        prob = f"Find the Taylor series of $f(x) = {sp.latex(f)}$ around $x = {point}$ up to order {n - 1}."
        step = f"Step: Taylor expansion around {point}."
    return prob, f"${sp.latex(s)}$", {"category": "taylor_series", "step_text": step}


# ─────────────────────────── FUNCTIONS ───────────────────────────

def gen_function_domain(dif):
    """Domain of a function. Hardcoded templates parameterised by a constant."""
    a = rint(1, 9)
    forms = [
        (f"f(x) = √(x - {a})",     f"x ≥ {a}",                "Need radicand ≥ 0"),
        (f"f(x) = √({a} - x)",     f"x ≤ {a}",                "Need radicand ≥ 0"),
        (f"f(x) = 1/(x - {a})",    f"x ≠ {a}",                "Denominator can't be zero"),
        (f"f(x) = ln(x - {a})",    f"x > {a}",                "Need argument > 0"),
        (f"f(x) = √(x² - {a*a})",  f"x ≤ -{a} or x ≥ {a}",    "Need x² ≥ a²"),
        (f"f(x) = 1/√(x - {a})",   f"x > {a}",                "Need radicand > 0 (also denom)"),
    ]
    fn, dom, why = random.choice(forms)
    prob = f"Find the domain of {fn}."
    return prob, dom, {"category": "function_domain", "step_text": f"Step: {why}."}


def gen_function_composition(dif):
    """(f∘g)(x) for two random linear functions."""
    a, b = rint(-4, 4, exclude_zero=True), rint(-6, 6)
    c, d = rint(-4, 4, exclude_zero=True), rint(-6, 6)
    p = a * c
    q = a * d + b
    f = f"f(x) = {a}x + {b}" if b >= 0 else f"f(x) = {a}x - {-b}"
    g = f"g(x) = {c}x + {d}" if d >= 0 else f"g(x) = {c}x - {-d}"
    if q >= 0:
        ans = f"{p}x + {q}"
    else:
        ans = f"{p}x - {-q}"
    prob = f"Given {f} and {g}, find $(f \\circ g)(x) = f(g(x))$ and simplify."
    step = f"Step: f(g(x)) = {a}·({c}x+{d})+{b} = {p}x + {q}."
    return prob, ans, {"category": "function_composition", "step_text": step}


def gen_function_inverse_linear(dif):
    """Inverse of a random linear function."""
    a = rint(-5, 5, exclude_zero=True)
    b = rint(-8, 8)
    fa = Fraction(1, a)
    fb = Fraction(-b, a)
    f = f"f(x) = {a}x + {b}" if b >= 0 else f"f(x) = {a}x - {-b}"
    if fb >= 0:
        ans = f"f⁻¹(x) = {canon_num(fa)}x + {canon_num(fb)}"
    else:
        ans = f"f⁻¹(x) = {canon_num(fa)}x - {canon_num(-fb)}"
    prob = f"Find the inverse of {f}."
    step = f"Step: swap x and y, then solve x = {a}y + {b} for y."
    return prob, ans, {"category": "function_inverse_linear", "step_text": step}


# ─────────────────────────── ODE ───────────────────────────

def gen_ode_separable(dif):
    """Solve a random separable first-order ODE."""
    if random.random() < 0.5:
        k = rint(-3, 3, exclude_zero=True)
        y0 = rint(1, 5)
        prob = f"Solve the ODE dy/dx = {k}y with y(0) = {y0}."
        ans = f"y = {y0}e^({k}x)"
        step = f"Step: separable: dy/y = {k} dx → ln|y| = {k}x + C → y = Ae^({k}x); y(0)={y0} ⇒ A={y0}."
    else:
        k = rint(1, 4)
        y0 = rint(0, 5)
        prob = f"Solve dy/dx = {k}x with y(0) = {y0}."
        coef = Fraction(k, 2)
        coef_s = "" if coef == 1 else ("-" if coef == -1 else f"{canon_num(coef)}")
        if y0 == 0:
            ans = f"y = {coef_s}x²"
        else:
            ans = f"y = {coef_s}x² + {y0}"
        step = f"Step: integrate both sides; use y(0)={y0} to fix the constant."
    return prob, ans, {"category": "ode_separable", "step_text": step}


def gen_ode_linear_first(dif):
    """First-order linear ODE: y' + ay = b, y(0) = y0."""
    a = rint(1, 4)
    b = random.choice([0, rint(1, 6)])
    y0 = rint(1, 5)
    if b == 0:
        prob = f"Solve y' + {a}y = 0 with y(0) = {y0}."
        ans = f"y = {y0}e^(-{a}x)"
        step = f"Step: y' = -{a}y → y = Ae^(-{a}x); A = {y0}."
    else:
        bp = Fraction(b, a)
        c = Fraction(y0) - bp
        prob = f"Solve y' + {a}y = {b} with y(0) = {y0}."
        ans = f"y = {canon_num(c)}e^(-{a}x) + {canon_num(bp)}"
        step = (f"Step: particular solution {canon_num(bp)}; general "
                f"y = Ae^(-{a}x) + {canon_num(bp)}; y(0)={y0} ⇒ A={canon_num(c)}.")
    return prob, ans, {"category": "ode_linear_first", "step_text": step}
