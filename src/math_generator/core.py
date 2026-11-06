"""Shared helpers used across all generator modules."""

import math
import random
import re
from decimal import Decimal
from fractions import Fraction

# ─────────────────────────── numeric / random helpers ───────────────────────────

DEC_P = 0.2  # historical: probability some generators introduce decimals

MUL_SIGNS = ["×", "*", "·"]
DIV_SIGNS = ["/", "÷"]


def rand_mul(*, allow_x=False):
    if allow_x and random.random() < 0.05:
        return "x"
    return random.choice(MUL_SIGNS)


def rand_div():
    return random.choice(DIV_SIGNS)


def rint(lo, hi, *, exclude_zero=False):
    while True:
        x = random.randint(lo, hi)
        if not (exclude_zero and x == 0):
            return x


def rfraction(den_lo=2, den_hi=12, allow_neg=False):
    num = rint(1, 19)
    den = rint(den_lo, den_hi, exclude_zero=True)
    if allow_neg and random.random() < 0.5:
        num = -num
    return Fraction(num, den)


def rdec(lo, hi, *, places=(1, 2), allow_neg=False):
    p = random.choice(places) if isinstance(places, (list, tuple)) else places
    scale = Decimal(10) ** -p
    n = rint(int(lo) * (10 ** p), int(hi) * (10 ** p))
    val = Decimal(n) * scale
    if allow_neg and random.random() < 0.5:
        val = -val
    return val.quantize(scale)


def canon_dec(x: Decimal, places=2) -> str:
    q = Decimal(10) ** -places
    x = x.quantize(q)
    s = format(x.normalize(), "f")
    return s.rstrip("0").rstrip(".") if "." in s else s


def canon_num(x):
    """Render a number consistently. Fractions reduce; integers come back plain."""
    if isinstance(x, Fraction):
        x = x.limit_denominator()
        if x.denominator == 1:
            return str(x.numerator)
        return f"{x.numerator}/{x.denominator}"
    if isinstance(x, float):
        s = str(x)
        return s.rstrip("0").rstrip(".") if "." in s else s
    return str(int(x))


def choice_weighted(items_with_w):
    total = sum(w for _, w in items_with_w)
    pick = random.uniform(0, total)
    acc = 0.0
    for item, w in items_with_w:
        acc += w
        if pick <= acc:
            return item
    return items_with_w[-1][0]


# ─────────────────────────── polynomial / matrix formatting ───────────────────────────

def fmt_coef(c, var, first=False):
    if c == 0:
        return ""
    if var == "":
        return ("+" if c > 0 and not first else "") + str(c)
    if c == 1:
        s = var
    elif c == -1:
        s = f"-{var}"
    else:
        s = f"{c}{var}"
    if not first and c > 0:
        s = "+" + s
    return s


def fmt_poly(coeffs, var="x"):
    """coeffs[i] = coefficient of var^(n-i), highest degree first."""
    n = len(coeffs) - 1
    parts = []
    for i, c in enumerate(coeffs):
        deg = n - i
        if c == 0:
            continue
        if deg == 0:
            parts.append(fmt_coef(c, "", first=(len(parts) == 0)))
        elif deg == 1:
            parts.append(fmt_coef(c, var, first=(len(parts) == 0)))
        else:
            parts.append(fmt_coef(c, f"{var}^{deg}", first=(len(parts) == 0)))
    return "".join(parts) if parts else "0"


def mat2str(M):
    """Format a 2-D list as a LaTeX pmatrix."""
    rows = " \\\\ ".join(" & ".join(str(v) for v in row) for row in M)
    return f"\\begin{{pmatrix}} {rows} \\end{{pmatrix}}"


def mat_mul(A, B):
    n, m, p = len(A), len(A[0]), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


def mat_add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def det2(M):
    return M[0][0] * M[1][1] - M[0][1] * M[1][0]


def det3(M):
    a = M[0]
    return (
        a[0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
        - a[1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
        + a[2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])
    )


def scalar_mat(k, M):
    return [[k * M[i][j] for j in range(len(M[0]))] for i in range(len(M))]


# ─────────────────────────── arithmetic helpers ───────────────────────────

def gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)


def lcm(a, b):
    return abs(a * b) // gcd(a, b)


def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def prime_factors(n):
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def euler_totient(n):
    result = n
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result


def ext_gcd(a, b):
    """Extended Euclidean: returns (g, x, y) with g = gcd(a,b) = a*x + b*y."""
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = ext_gcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)


# ─────────────────────────── boolean rendering ───────────────────────────

_BOOL_BINARY = [("∧", " and "), ("∨", " or "), ("→", " <= "), ("↔", " == ")]


def _wrap_negations(s):
    """Walk `s` and replace each `¬X` (X = identifier or balanced parenthesised
    group) with `(not X)`. Done via a manual scan because Python's `not` has
    *lower* precedence than `<=` (used as our `→`) and `==` (used as `↔`),
    so a naive string replacement of `¬` with `not` would mis-parse
    `¬p → q` as `not (p <= q)`."""
    out = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == "¬":
            j = i + 1
            while j < n and s[j].isspace():
                j += 1
            if j < n and s[j] == "(":
                depth = 1
                k = j + 1
                while k < n and depth > 0:
                    if s[k] == "(":
                        depth += 1
                    elif s[k] == ")":
                        depth -= 1
                    k += 1
                inside = _wrap_negations(s[j + 1:k - 1])
                out.append(f"(not ({inside}))")
                i = k
            elif j < n and (s[j].isalpha() or s[j] == "_"):
                k = j
                while k < n and (s[k].isalnum() or s[k] == "_"):
                    k += 1
                out.append(f"(not {s[j:k]})")
                i = k
            else:
                out.append(c); i += 1
        elif c == "(":
            depth = 1
            k = i + 1
            while k < n and depth > 0:
                if s[k] == "(":
                    depth += 1
                elif s[k] == ")":
                    depth -= 1
                k += 1
            inner = _wrap_negations(s[i + 1:k - 1])
            out.append(f"({inner})")
            i = k
        else:
            out.append(c); i += 1
    return "".join(out)


def bool_to_py(s):
    """Convert a unicode boolean expression to a Python boolean expression."""
    out = _wrap_negations(s)
    for k, v in _BOOL_BINARY:
        out = out.replace(k, v)
    return out


def bool_eval(expr_unicode, env):
    """Evaluate a unicode boolean expression in a (truth-value) environment."""
    return bool(eval(bool_to_py(expr_unicode), {"__builtins__": {}}, env))


def bool_to_unicode(e):
    """Render a SymPy Boolean as ∧ / ∨ / ¬ instead of & / | / ~."""
    return str(e).replace("&", "∧").replace("|", "∨").replace("~", "¬")


# ─────────────────────────── step-by-step reasoning helper ───────────────────────────

class StepBuilder:
    """Tiny chain-of-thought collector. Each `.add(text)` appends a numbered step;
    `.text()` returns "Step 1: …\nStep 2: …\nStep 3: …" suitable for the
    "step_text" meta field. Generators that have only one obvious step can keep
    using a plain f"Step: …" string; this class is for problems where multiple
    intermediate transformations are useful as training signal."""

    __slots__ = ("_lines",)

    def __init__(self):
        self._lines = []

    def add(self, line):
        self._lines.append(str(line))
        return self

    def text(self):
        if not self._lines:
            return ""
        return "\n".join(f"Step {i + 1}: {ln}" for i, ln in enumerate(self._lines))

    def __len__(self):
        return len(self._lines)
