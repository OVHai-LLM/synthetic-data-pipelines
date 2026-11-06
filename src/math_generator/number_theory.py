"""Number theory + numeric bases + financial math."""

import math
import random
from decimal import Decimal
from fractions import Fraction

from .core import canon_num, ext_gcd, euler_totient, prime_factors, rint


def gen_prime_factorization(dif):
    lo, hi = (10, 100) if dif == "medium" else (50, 500)
    n = rint(lo, hi)
    factors = prime_factors(n)
    parts = []
    for p, e in sorted(factors.items()):
        parts.append(f"{p}^{e}" if e > 1 else str(p))
    ans = " × ".join(parts)
    prob = f"Find the prime factorization of {n}."
    step = f"Step: divide by smallest primes repeatedly. {n} = {ans}"
    return prob, ans, {"category": "prime_factorization", "step_text": step}


def gen_euler_totient(dif):
    lo, hi = (2, 30) if dif == "medium" else (2, 60)
    n = rint(lo, hi)
    ans = euler_totient(n)
    prob = f"Compute Euler's totient function φ({n})."
    step = f"Step: count integers in [1,{n}] coprime to {n}."
    return prob, canon_num(ans), {"category": "euler_totient", "step_text": step}


def gen_modular_arithmetic(dif):
    """Modular basic / inverse / Fermat (legacy 'pickup'). RANDOMIZED parameters."""
    kind = random.choice(["basic", "inverse", "fermat"]) if dif == "hard" else "basic"
    if kind == "basic":
        a, m = rint(0, 10000), rint(2, 97)
        return f"Compute {a} mod {m}.", str(a % m), {"category": "modular_basic"}
    if kind == "inverse":
        m = random.choice([7, 11, 13, 17, 19, 23])
        a = rint(1, m - 1)
        inv = pow(a, -1, m)
        prob = f"Find the modular inverse of {a} modulo {m}."
        step = f"Step: by Fermat, x ≡ {a}^({m}−2) ≡ {inv} (mod {m})."
        return prob, str(inv), {"category": "modular_inverse", "step_text": step}
    p = random.choice([5, 7, 11, 13, 17])
    a = rint(1, p - 1)
    prob = f"By Fermat's Little Theorem, compute {a}^{p} mod {p}."
    step = f"Step: a^p ≡ a (mod p) for prime p, so {a}^{p} ≡ {a} (mod {p})."
    return prob, str(a % p), {"category": "fermat_little", "step_text": step}


def gen_extended_euclidean(dif):
    """Find Bezout coefficients gcd(a,b) = a·x + b·y."""
    a = rint(10, 200)
    b = rint(10, 200)
    if a < b:
        a, b = b, a
    g, x, y = ext_gcd(a, b)
    prob = f"Find integers x, y with gcd({a},{b}) = {a}x + {b}y. Report (gcd, x, y)."
    ans = f"({g}, {x}, {y})"
    step = f"Step: extended Euclidean ⇒ {a}·{x} + {b}·{y} = {a*x + b*y} = {g}."
    return prob, ans, {"category": "extended_euclidean", "step_text": step}


def gen_chinese_remainder(dif):
    """Solve x ≡ a (mod m), x ≡ b (mod n) with coprime m, n."""
    pairs = [(3, 5), (3, 7), (4, 7), (5, 7), (5, 9), (7, 9), (7, 11),
             (8, 9), (11, 13), (9, 10)]
    m, n = random.choice(pairs)
    a = rint(0, m - 1)
    b = rint(0, n - 1)
    M = m * n
    n_inv_mod_m = pow(n, -1, m)
    m_inv_mod_n = pow(m, -1, n)
    x = (a * n * n_inv_mod_m + b * m * m_inv_mod_n) % M
    prob = (f"Solve the system: x ≡ {a} (mod {m}), x ≡ {b} (mod {n}). "
            f"Give the smallest non-negative solution.")
    step = f"Step: by CRT with M = {M}, x = {x}."
    return prob, str(x), {"category": "chinese_remainder", "step_text": step}


def gen_fermat_little(dif):
    """Apply Fermat's little theorem: a^(p-1) ≡ 1 (mod p)."""
    p = random.choice([5, 7, 11, 13, 17, 19, 23])
    a = rint(2, p - 1)
    k = rint(2, 6)
    r = rint(0, p - 2)
    e = k * (p - 1) + r
    val = pow(a, e, p)
    prob = f"Use Fermat's little theorem to compute {a}^{e} mod {p}."
    step = (f"Step: gcd({a},{p}) = 1, so {a}^{p - 1} ≡ 1 (mod {p}); "
            f"thus {a}^{e} ≡ {a}^{r} ≡ {val} (mod {p}).")
    return prob, str(val), {"category": "fermat_little", "step_text": step}


# ─────────────────────────── BASES & SCIENTIFIC NOTATION ───────────────────────────

def gen_base_conversion(dif):
    """Convert decimal ↔ binary/octal/hex."""
    n = rint(5, 511)
    target = random.choice(["bin", "oct", "hex"])
    direction = random.choice(["to", "from"])
    if direction == "to":
        if target == "bin": s, tname = bin(n)[2:], "binary"
        elif target == "oct": s, tname = oct(n)[2:], "octal"
        else: s, tname = hex(n)[2:].upper(), "hexadecimal"
        return (f"Convert the decimal number {n} to {tname}.", s,
                {"category": "base_conversion",
                 "step_text": f"Step: divide repeatedly by the base, recording remainders bottom-up."})
    if target == "bin": s, tname = bin(n)[2:], "binary"
    elif target == "oct": s, tname = oct(n)[2:], "octal"
    else: s, tname = hex(n)[2:].upper(), "hexadecimal"
    return (f"Convert the {tname} number {s} to decimal.", str(n),
            {"category": "base_conversion",
             "step_text": "Step: weight each digit by powers of the base and sum."})


def gen_scientific_notation(dif):
    """Convert to/from scientific notation."""
    mantissa_int = rint(10, 999)
    exp = rint(-6, 8)
    mant = Decimal(mantissa_int) / Decimal(100)
    decimal_value = mant * (Decimal(10) ** exp)
    if random.random() < 0.5:
        prob = f"Express {decimal_value} in scientific notation."
        ans = f"{mant}×10^{exp}"
        step = "Step: shift the decimal point until exactly one nonzero digit remains before it; the count is the exponent."
    else:
        prob = f"Express {mant}×10^{exp} in standard decimal form."
        ans = str(decimal_value)
        step = f"Step: shift the decimal point {exp} places."
    return prob, ans, {"category": "scientific_notation", "step_text": step}


# ─────────────────────────── FINANCE ───────────────────────────

def gen_simple_interest(dif):
    P = rint(500, 5000)
    r = random.choice([3, 4, 5, 6, 7, 8])
    t = rint(1, 10)
    I = Fraction(P * r * t, 100)
    prob = f"Find the simple interest on a principal of ${P} at {r}% per year for {t} years."
    return f"{prob}", f"${canon_num(I)}", {
        "category": "simple_interest",
        "step_text": f"Step: I = P·r·t = {P}·{r}/100·{t} = {canon_num(I)}.",
    }


def gen_compound_interest(dif):
    P = rint(500, 5000)
    r = random.choice([5, 10, 20, 25])
    t = rint(2, 6)
    factor = Fraction(100 + r, 100)
    Aval = Fraction(P) * (factor ** t)
    rounded = round(Aval * 100) / 100
    ans = f"${rounded:.2f}"
    prob = f"Find the value after {t} years of ${P} invested at {r}% compounded annually (round to cents)."
    step = f"Step: A = P(1+r)^t = {P}·(1+{r}/100)^{t} ≈ {ans}."
    return prob, ans, {"category": "compound_interest", "step_text": step}
