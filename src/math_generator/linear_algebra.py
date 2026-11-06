"""Matrix arithmetic, determinants, inverse, eigen-, transpose, linear independence."""

import math
import random
from fractions import Fraction

from .core import canon_num, det2, mat2str, mat_add, mat_mul, rint


def gen_matrix_add(dif):
    size = 2 if dif == "medium" else random.choice([2, 3])
    lo, hi = (-5, 5) if dif == "medium" else (-9, 9)
    A = [[rint(lo, hi) for _ in range(size)] for _ in range(size)]
    B = [[rint(lo, hi) for _ in range(size)] for _ in range(size)]
    op = random.choice(["+", "-"])
    if op == "+":
        C = mat_add(A, B)
    else:
        C = [[A[i][j] - B[i][j] for j in range(size)] for i in range(size)]
    prob = f"Compute $A {op} B$ where $A = {mat2str(A)}$ and $B = {mat2str(B)}$."
    return prob, f"${mat2str(C)}$", {"category": "matrix_add"}


def gen_matrix_multiply(dif):
    lo, hi = -4, 4
    A = [[rint(lo, hi) for _ in range(2)] for _ in range(2)]
    B = [[rint(lo, hi) for _ in range(2)] for _ in range(2)]
    C = mat_mul(A, B)
    prob = f"Compute $AB$ where $A = {mat2str(A)}$ and $B = {mat2str(B)}$."
    return prob, f"${mat2str(C)}$", {"category": "matrix_multiply"}


def gen_determinant_2x2(dif):
    lo, hi = (-6, 6) if dif == "medium" else (-9, 9)
    M = [[rint(lo, hi) for _ in range(2)] for _ in range(2)]
    ans = det2(M)
    prob = f"Find the determinant of $M = {mat2str(M)}$."
    step = f"Step: det = {M[0][0]}·{M[1][1]} − {M[0][1]}·{M[1][0]} = {ans}."
    return prob, str(ans), {"category": "determinant_2x2", "step_text": step}


def gen_determinant_3x3(dif):
    lo, hi = -4, 4
    M = [[rint(lo, hi) for _ in range(3)] for _ in range(3)]
    a = M[0]
    ans = (a[0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
           - a[1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
           + a[2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
    prob = f"Find the determinant of $M = {mat2str(M)}$."
    step = "Step: expand along the first row using the cofactor formula."
    return prob, str(ans), {"category": "determinant_3x3", "step_text": step}


def gen_matrix_inverse_2x2(dif):
    """Generate a 2×2 matrix with integer inverse (det = ±1)."""
    for _ in range(100):
        a, b = rint(-4, 4), rint(-4, 4)
        c, d = rint(-4, 4), rint(-4, 4)
        det = a * d - b * c
        if det in (1, -1):
            M = [[a, b], [c, d]]
            inv = [[d * det, -b * det], [-c * det, a * det]]
            prob = f"Find the inverse of $M = {mat2str(M)}$."
            step = f"Step: det(M) = {det}; $M^{{-1}} = (1/{det}) \\cdot$ adj(M)."
            return prob, f"${mat2str(inv)}$", {"category": "matrix_inverse", "step_text": step}
    M = [[2, 1], [1, 1]]
    prob = f"Find the inverse of $M = {mat2str(M)}$."
    return prob, "$\\begin{pmatrix} 1 & -1 \\\\ -1 & 2 \\end{pmatrix}$", {"category": "matrix_inverse"}


def gen_eigenvalue_2x2(dif):
    """Construct a 2×2 matrix with integer eigenvalues."""
    e1, e2 = rint(-6, 6), rint(-6, 6)
    off = rint(-4, 4)
    M = [[e1, off], [0, e2]]
    if random.random() < 0.5 and e1 != e2:
        M = [[e1, e2 - e1 + off], [0, e2]]
    prob = f"Find the eigenvalues of $A = {mat2str(M)}$."
    trace = M[0][0] + M[1][1]
    det_val = det2(M)
    disc = trace * trace - 4 * det_val
    if disc >= 0 and math.isqrt(disc) ** 2 == disc:
        sd = math.isqrt(disc)
        l1 = Fraction(trace + sd, 2)
        l2 = Fraction(trace - sd, 2)
        ev = sorted([canon_num(l1), canon_num(l2)])
        ans = f"λ₁ = {ev[0]}, λ₂ = {ev[1]}"
        step = (f"Step: char. poly λ² − {trace}λ + {det_val} = 0; "
                f"discriminant {disc}; λ = ({trace} ± {sd})/2.")
    else:
        M = [[e1, off], [0, e2]]
        ans = f"λ₁ = {e1}, λ₂ = {e2}"
        step = f"Step: upper-triangular matrix; eigenvalues are diagonal entries {e1} and {e2}."
        prob = f"Find the eigenvalues of $A = {mat2str(M)}$."
    return prob, ans, {"category": "eigenvalue_2x2", "step_text": step}


def gen_eigenvector_2x2(dif):
    """Find an eigenvector for a known eigenvalue of a 2×2 matrix."""
    e1, e2 = rint(-5, 5), rint(-5, 5)
    while e1 == e2:
        e2 = rint(-5, 5)
    off = rint(1, 4)
    M = [[e1, off], [0, e2]]
    target = random.choice([e1, e2])
    if target == e1:
        ev = [1, 0]
    else:
        ev = [off, e2 - e1]
    prob = (f"For matrix $A = {mat2str(M)}$, find an eigenvector corresponding "
            f"to eigenvalue λ = {target}.")
    ans = f"$\\begin{{pmatrix}} {ev[0]} \\\\ {ev[1]} \\end{{pmatrix}}$ (up to a scalar multiple)"
    step = f"Step: solve $(A − {target}I)\\mathbf{{v}} = 0$ and read off the null-space vector."
    return prob, ans, {"category": "eigenvector_2x2", "step_text": step}


def gen_matrix_transpose(dif):
    rows, cols = random.choice([(2, 2), (2, 3), (3, 2)])
    lo, hi = -6, 6
    M = [[rint(lo, hi) for _ in range(cols)] for _ in range(rows)]
    MT = [[M[i][j] for i in range(rows)] for j in range(cols)]
    prob = f"Find the transpose of $M = {mat2str(M)}$."
    return prob, f"${mat2str(MT)}$", {"category": "matrix_transpose"}


def gen_linear_independence_2d(dif):
    """Are two 2-D vectors linearly independent? RANDOMIZED."""
    a, b = rint(-5, 5, exclude_zero=True), rint(-5, 5, exclude_zero=True)
    if random.random() < 0.5:
        k = rint(-3, 3, exclude_zero=True)
        c, d = a * k, b * k
        indep = False
    else:
        c, d = rint(-5, 5, exclude_zero=True), rint(-5, 5, exclude_zero=True)
        indep = (a * d - b * c) != 0
    prob = f"Are the vectors v₁ = ({a}, {b}) and v₂ = ({c}, {d}) linearly independent?"
    ans = "Yes" if indep else "No"
    det = a * d - b * c
    step = f"Step: in ℝ², det([v₁ v₂]) = {a}·{d} − {b}·{c} = {det}; independent ⇔ ≠ 0."
    return prob, ans, {"category": "linear_independence_2d", "step_text": step}
