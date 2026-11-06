"""Plane geometry, 3-D mensuration, conic sections."""

import math
import random
from fractions import Fraction

from .core import canon_num, rint


PYTHAG_TRIPLES = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25),
                  (20, 21, 29), (9, 40, 41), (6, 8, 10), (9, 12, 15), (12, 16, 20)]


def gen_geometry_pythagoras(dif):
    a, b, c = random.choice(PYTHAG_TRIPLES)
    k = random.choice([1, 2, 3]) if dif != "simple" else 1
    a, b, c = a * k, b * k, c * k
    missing = random.choice(["a", "b", "c"])
    if missing == "c":
        prob = f"A right triangle has legs of length {a} and {b}. Find the hypotenuse."
        ans = str(c)
        step = f"Step: c = √(a²+b²) = √{a*a+b*b} = {c}."
    elif missing == "a":
        prob = f"A right triangle has hypotenuse {c} and one leg {b}. Find the other leg."
        ans = str(a)
        step = f"Step: a = √(c²−b²) = √{c*c-b*b} = {a}."
    else:
        prob = f"A right triangle has hypotenuse {c} and one leg {a}. Find the other leg."
        ans = str(b)
        step = f"Step: b = √(c²−a²) = √{c*c-a*a} = {b}."
    return prob, ans, {"category": "geometry_pythagoras", "step_text": step}


def gen_geometry_triangle_area(dif):
    if random.random() < 0.5:
        b = rint(2, 20); h = rint(2, 20)
        area = Fraction(b * h, 2)
        prob = f"Find the area of a triangle with base {b} and height {h}."
        return prob, canon_num(area), {"category": "geometry_triangle_area",
                                       "step_text": f"Step: area = (b·h)/2 = ({b}·{h})/2 = {canon_num(area)}."}
    x1, y1 = rint(-6, 6), rint(-6, 6)
    x2, y2 = rint(-6, 6), rint(-6, 6)
    x3, y3 = rint(-6, 6), rint(-6, 6)
    det = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    if det == 0:
        x3 += 1
        det = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    area = Fraction(abs(det), 2)
    prob = (f"Find the area of the triangle with vertices "
            f"({x1},{y1}), ({x2},{y2}), ({x3},{y3}).")
    step = f"Step: area = |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|/2 = |{det}|/2."
    return prob, canon_num(area), {"category": "geometry_triangle_area", "step_text": step}


def gen_geometry_circle(dif):
    r = rint(1, 15)
    op = random.choice(["area", "circumference", "diameter_from_circ", "radius_from_area"])
    if op == "area":
        return (f"Find the area of a circle with radius {r} (leave answer in terms of π).",
                f"{r * r}π", {"category": "geometry_circle",
                              "step_text": f"Step: A = πr² = π·{r}²."})
    if op == "circumference":
        return (f"Find the circumference of a circle with radius {r} (leave answer in terms of π).",
                f"{2 * r}π", {"category": "geometry_circle",
                              "step_text": f"Step: C = 2πr = 2π·{r}."})
    if op == "diameter_from_circ":
        return (f"A circle has circumference {2 * r}π. Find its diameter.",
                str(2 * r), {"category": "geometry_circle",
                             "step_text": "Step: C = πd, so d = C/π."})
    return (f"A circle has area {r * r}π. Find its radius.",
            str(r), {"category": "geometry_circle",
                     "step_text": f"Step: A = πr², so r = √{r*r} = {r}."})


def gen_geometry_polygon_angles(dif):
    n = rint(3, 12)
    op = random.choice(["interior_sum", "interior_each", "exterior_each"])
    if op == "interior_sum":
        s = (n - 2) * 180
        return (f"Find the sum of interior angles of a polygon with {n} sides.",
                f"{s}°", {"category": "geometry_polygon_angles",
                          "step_text": f"Step: (n−2)·180° = {s}°."})
    if op == "interior_each":
        s = (n - 2) * 180
        each = Fraction(s, n)
        return (f"Find the measure of each interior angle of a regular {n}-gon.",
                f"{canon_num(each)}°", {"category": "geometry_polygon_angles",
                                        "step_text": f"Step: ((n−2)·180°)/n = {canon_num(each)}°."})
    each = Fraction(360, n)
    return (f"Find each exterior angle of a regular {n}-gon.",
            f"{canon_num(each)}°", {"category": "geometry_polygon_angles",
                                    "step_text": f"Step: 360°/n = {canon_num(each)}°."})


def gen_geometry_distance_2d(dif):
    x1, y1 = rint(-8, 8), rint(-8, 8)
    x2, y2 = rint(-8, 8), rint(-8, 8)
    if (x1, y1) == (x2, y2):
        x2 += 1
    op = random.choice(["distance", "midpoint", "slope"])
    if op == "distance":
        d2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
        sq = math.isqrt(d2)
        ans = str(sq) if sq * sq == d2 else f"√{d2}"
        return (f"Find the distance between ({x1},{y1}) and ({x2},{y2}).",
                ans, {"category": "geometry_distance_2d",
                      "step_text": f"Step: d = √((x2−x1)² + (y2−y1)²) = √{d2}."})
    if op == "midpoint":
        mx, my = Fraction(x1 + x2, 2), Fraction(y1 + y2, 2)
        return (f"Find the midpoint of segment from ({x1},{y1}) to ({x2},{y2}).",
                f"({canon_num(mx)}, {canon_num(my)})",
                {"category": "geometry_distance_2d",
                 "step_text": "Step: midpoint = ((x1+x2)/2, (y1+y2)/2)."})
    if x2 == x1:
        return (f"Find the slope of the line through ({x1},{y1}) and ({x2},{y2}).",
                "undefined", {"category": "geometry_distance_2d",
                              "step_text": "Step: vertical line ⇒ slope undefined (Δx=0)."})
    sl = Fraction(y2 - y1, x2 - x1)
    return (f"Find the slope of the line through ({x1},{y1}) and ({x2},{y2}).",
            canon_num(sl), {"category": "geometry_distance_2d",
                            "step_text": f"Step: slope = (y2−y1)/(x2−x1) = {canon_num(sl)}."})


def gen_geometry_line_equation(dif):
    x1, y1 = rint(-6, 6), rint(-6, 6)
    x2, y2 = rint(-6, 6), rint(-6, 6)
    if x2 == x1: x2 += 1
    m = Fraction(y2 - y1, x2 - x1)
    b = Fraction(y1) - m * Fraction(x1)
    if b >= 0:
        eq = f"y = {canon_num(m)}x + {canon_num(b)}"
    else:
        eq = f"y = {canon_num(m)}x − {canon_num(-b)}"
    prob = f"Find the slope-intercept equation of the line through ({x1},{y1}) and ({x2},{y2})."
    step = f"Step: m = (y2−y1)/(x2−x1) = {canon_num(m)}; b = y1 − m·x1 = {canon_num(b)}."
    return prob, eq, {"category": "geometry_line_equation", "step_text": step}


# ─────────────────────────── 3-D ───────────────────────────

def gen_geometry_3d_volume(dif):
    shape = random.choice(["cube", "box", "cylinder", "cone", "sphere"])
    if shape == "cube":
        s = rint(2, 12)
        return (f"Find the volume of a cube with side length {s}.",
                str(s ** 3), {"category": "geometry_3d_volume",
                              "step_text": f"Step: V = s³ = {s}³ = {s ** 3}."})
    if shape == "box":
        l, w, h = rint(2, 12), rint(2, 12), rint(2, 12)
        return (f"Find the volume of a rectangular prism with dimensions {l}×{w}×{h}.",
                str(l * w * h), {"category": "geometry_3d_volume",
                                 "step_text": f"Step: V = lwh = {l*w*h}."})
    if shape == "cylinder":
        r, h = rint(1, 8), rint(1, 12)
        return (f"Find the volume of a cylinder with radius {r} and height {h} (in terms of π).",
                f"{r * r * h}π", {"category": "geometry_3d_volume",
                                  "step_text": f"Step: V = πr²h."})
    if shape == "cone":
        r, h = rint(1, 8), rint(1, 12)
        v = Fraction(r * r * h, 3)
        return (f"Find the volume of a cone with radius {r} and height {h} (in terms of π).",
                f"{canon_num(v)}π", {"category": "geometry_3d_volume",
                                     "step_text": f"Step: V = (1/3)πr²h."})
    r = rint(1, 8)
    v = Fraction(4 * r ** 3, 3)
    return (f"Find the volume of a sphere with radius {r} (in terms of π).",
            f"{canon_num(v)}π", {"category": "geometry_3d_volume",
                                 "step_text": f"Step: V = (4/3)πr³."})


def gen_geometry_3d_surface(dif):
    shape = random.choice(["cube", "box", "cylinder", "sphere"])
    if shape == "cube":
        s = rint(2, 12)
        return (f"Find the surface area of a cube with side length {s}.",
                str(6 * s * s), {"category": "geometry_3d_surface",
                                 "step_text": f"Step: SA = 6s² = {6*s*s}."})
    if shape == "box":
        l, w, h = rint(2, 10), rint(2, 10), rint(2, 10)
        sa = 2 * (l * w + l * h + w * h)
        return (f"Find the surface area of a box with dimensions {l}×{w}×{h}.",
                str(sa), {"category": "geometry_3d_surface",
                          "step_text": f"Step: SA = 2(lw + lh + wh) = {sa}."})
    if shape == "cylinder":
        r, h = rint(1, 8), rint(1, 12)
        sa = 2 * r * r + 2 * r * h
        return (f"Find the surface area of a cylinder with radius {r}, height {h} (in terms of π).",
                f"{sa}π", {"category": "geometry_3d_surface",
                           "step_text": f"Step: SA = 2πr² + 2πrh."})
    r = rint(1, 10)
    return (f"Find the surface area of a sphere with radius {r} (in terms of π).",
            f"{4 * r * r}π", {"category": "geometry_3d_surface",
                              "step_text": f"Step: SA = 4πr²."})


# ─────────────────────────── CONICS ───────────────────────────

def gen_conic_circle_eq(dif):
    h = rint(-6, 6); k = rint(-6, 6); r = rint(1, 9)
    sh = f"x − {h}" if h >= 0 else f"x + {-h}"
    sk = f"y − {k}" if k >= 0 else f"y + {-k}"
    prob = f"Find the centre and radius of the circle ({sh})² + ({sk})² = {r * r}."
    ans = f"centre ({h}, {k}), radius {r}"
    step = f"Step: standard form (x−h)²+(y−k)² = r² gives centre ({h},{k}) and radius √{r*r} = {r}."
    return prob, ans, {"category": "conic_circle_eq", "step_text": step}


def gen_conic_parabola(dif):
    if random.random() < 0.5:
        a = rint(-3, 3, exclude_zero=True); h = rint(-5, 5); k = rint(-5, 5)
        sh = f"x − {h}" if h >= 0 else f"x + {-h}"
        sk_str = f"+ {k}" if k >= 0 else f"− {-k}"
        return (f"Find the vertex of the parabola y = {a}({sh})² {sk_str}.",
                f"({h}, {k})", {"category": "conic_parabola",
                                "step_text": "Step: y = a(x−h)² + k has vertex (h, k)."})
    p = rint(1, 5)
    return (f"For the parabola x² = {4 * p}y, find the focus.",
            f"(0, {p})", {"category": "conic_parabola",
                          "step_text": f"Step: x² = 4py with 4p = {4*p}; focus at (0, p) = (0,{p})."})


# ─────────────────────────── HERON & TRIANGLE INEQUALITY ───────────────────────────

def gen_geometry_heron(dif):
    """Random integer-sided triangle, area via Heron's formula. Picks sides
    that satisfy the triangle inequality and yield a clean (integer or √n) area."""
    a, b, c = 3, 4, 5  # safe default
    for _ in range(50):
        a, b, c = rint(3, 14), rint(3, 14), rint(3, 14)
        if a + b > c and a + c > b and b + c > a:
            break
    s = Fraction(a + b + c, 2)
    area_sq = s * (s - a) * (s - b) * (s - c)
    # area_sq is a Fraction. Try to express as integer or √n form.
    if area_sq.denominator == 1:
        n = area_sq.numerator
        sq = math.isqrt(n)
        if sq * sq == n:
            ans = str(sq)
        else:
            ans = f"√{n}"
    else:
        ans = f"√({canon_num(area_sq)})"
    prob = f"Find the area of a triangle with sides {a}, {b}, {c} (use Heron's formula)."
    step = (f"Step: s = (a+b+c)/2 = {canon_num(s)}; area = √(s·(s−a)·(s−b)·(s−c)) "
            f"= √({canon_num(area_sq)}).")
    return prob, ans, {"category": "geometry_heron", "step_text": step}


def gen_geometry_triangle_inequality(dif):
    """Given three lengths, can they form a triangle?"""
    a = rint(2, 15); b = rint(2, 15); c = rint(2, 15)
    valid = a + b > c and a + c > b and b + c > a
    prob = f"Can sides of length {a}, {b}, {c} form a triangle?"
    ans = "Yes" if valid else "No"
    if valid:
        step = (f"Step: triangle inequality requires each side < sum of the other two. "
                f"Checks: {a} < {b}+{c}, {b} < {a}+{c}, {c} < {a}+{b} — all hold.")
    else:
        offenders = []
        for x, y, z in [(a, b, c), (b, a, c), (c, a, b)]:
            if x >= y + z:
                offenders.append(f"{x} ≥ {y}+{z} = {y+z}")
        step = "Step: triangle inequality fails: " + "; ".join(offenders) + "."
    return prob, ans, {"category": "geometry_triangle_inequality", "step_text": step}


# ─────────────────────────── LAW OF SINES / COSINES ───────────────────────────

def gen_geometry_law_of_cosines(dif):
    """SAS triangle: given two sides and the included angle (in degrees, special
    values that yield clean cos), find the third side. RANDOMIZED."""
    # Use angles whose cosine is rational/clean: 60° (1/2), 90° (0), 120° (-1/2),
    # 45° (√2/2 — keeps √ in output).
    angle_deg, cos_val = random.choice([(60, Fraction(1, 2)),
                                        (90, Fraction(0)),
                                        (120, Fraction(-1, 2))])
    a = rint(3, 12); b = rint(3, 12)
    c_sq = Fraction(a * a + b * b) - 2 * Fraction(a * b) * cos_val
    if c_sq.denominator == 1:
        n = c_sq.numerator
        sq = math.isqrt(n)
        ans = str(sq) if sq * sq == n else f"√{n}"
    else:
        ans = f"√({canon_num(c_sq)})"
    prob = (f"In triangle ABC, sides AB = {a}, AC = {b}, and the included angle A = {angle_deg}°. "
            f"Find side BC.")
    step = (f"Step: BC² = AB² + AC² − 2·AB·AC·cos(A) = {a*a} + {b*b} − 2·{a}·{b}·{canon_num(cos_val)} "
            f"= {canon_num(c_sq)}; BC = {ans}.")
    return prob, ans, {"category": "geometry_law_of_cosines", "step_text": step}


def gen_geometry_law_of_sines(dif):
    """ASA-ish: given one side and two angles, find another side. Uses angles whose
    sines are clean."""
    A_deg, B_deg = 60, 45  # safe default; A + B = 105 < 180
    for _ in range(30):
        A_deg = random.choice([30, 45, 60, 90, 150])
        B_deg = random.choice([30, 45, 60, 90])
        if A_deg + B_deg < 180 and 180 - A_deg - B_deg > 0:
            break
    a = rint(4, 12)
    # By law of sines: a/sin(A) = b/sin(B), so b = a·sin(B)/sin(A)
    sA = math.sin(math.radians(A_deg))
    sB = math.sin(math.radians(B_deg))
    b_val = a * sB / sA
    prob = (f"In triangle ABC, ∠A = {A_deg}°, ∠B = {B_deg}°, and side a (opposite A) = {a}. "
            f"Find side b (opposite B), to 3 decimal places.")
    ans = f"{b_val:.3f}"
    step = f"Step: by the law of sines, b = a·sin(B)/sin(A) = {a}·{sB:.4f}/{sA:.4f} ≈ {b_val:.3f}."
    return prob, ans, {"category": "geometry_law_of_sines", "step_text": step}


# ─────────────────────────── SPECIAL RIGHT TRIANGLES ───────────────────────────

def gen_geometry_special_triangle(dif):
    """30-60-90 or 45-45-90 with a random scaling, ask for one of the missing sides."""
    kind = random.choice(["30-60-90", "45-45-90"])
    k = rint(2, 10)
    if kind == "30-60-90":
        # sides: k (short, opposite 30°), k√3 (opp 60°), 2k (hypotenuse opp 90°)
        sub = random.choice(["short_to_long", "short_to_hyp", "hyp_to_short"])
        if sub == "short_to_long":
            return (f"In a 30-60-90 right triangle, the side opposite the 30° angle is {k}. "
                    f"Find the side opposite the 60° angle.",
                    f"{k}√3", {"category": "geometry_special_triangle",
                               "step_text": "Step: 30-60-90 ratio is 1 : √3 : 2; "
                                            f"long leg = short·√3 = {k}√3."})
        if sub == "short_to_hyp":
            return (f"In a 30-60-90 right triangle, the side opposite the 30° angle is {k}. "
                    f"Find the hypotenuse.",
                    str(2 * k), {"category": "geometry_special_triangle",
                                 "step_text": f"Step: hypotenuse = 2·short leg = {2*k}."})
        return (f"In a 30-60-90 right triangle, the hypotenuse is {2*k}. "
                f"Find the side opposite the 30° angle.",
                str(k), {"category": "geometry_special_triangle",
                         "step_text": f"Step: short leg = hypotenuse/2 = {k}."})
    # 45-45-90: legs both k, hypotenuse k√2
    sub = random.choice(["leg_to_hyp", "hyp_to_leg"])
    if sub == "leg_to_hyp":
        return (f"In a 45-45-90 right triangle, each leg is {k}. Find the hypotenuse.",
                f"{k}√2", {"category": "geometry_special_triangle",
                           "step_text": f"Step: 45-45-90 ratio 1 : 1 : √2 ⇒ hyp = {k}√2."})
    return (f"In a 45-45-90 right triangle, the hypotenuse is {k}√2. Find each leg.",
            str(k), {"category": "geometry_special_triangle",
                     "step_text": f"Step: leg = hypotenuse/√2 = ({k}√2)/√2 = {k}."})


# ─────────────────────────── SECTORS & ARCS ───────────────────────────

def gen_geometry_sector(dif):
    """Arc length or sector area, given radius and central angle."""
    r = rint(2, 12)
    # Use angles that are nice fractions of π (in radians) so answer stays clean.
    angle_choices = [(Fraction(1, 6), "π/6"), (Fraction(1, 4), "π/4"),
                     (Fraction(1, 3), "π/3"), (Fraction(1, 2), "π/2"),
                     (Fraction(2, 3), "2π/3"), (Fraction(3, 4), "3π/4"),
                     (Fraction(1, 1), "π"), (Fraction(2, 1), "2π")]
    frac, theta_str = random.choice(angle_choices)
    if random.random() < 0.5:
        # Arc length s = rθ
        arc = Fraction(r) * frac
        return (f"Find the arc length of a circle with radius {r} subtending a central angle of {theta_str} radians.",
                f"{canon_num(arc)}π", {"category": "geometry_sector",
                                       "step_text": f"Step: s = rθ = {r}·{theta_str} = {canon_num(arc)}π."})
    # Sector area A = (1/2) r² θ
    area = Fraction(1, 2) * Fraction(r * r) * frac
    return (f"Find the area of a circular sector with radius {r} and central angle {theta_str} radians.",
            f"{canon_num(area)}π", {"category": "geometry_sector",
                                    "step_text": f"Step: A = (1/2) r² θ = (1/2)·{r*r}·{theta_str} = {canon_num(area)}π."})


# ─────────────────────────── QUADRILATERALS ───────────────────────────

def gen_geometry_quadrilateral(dif):
    """Area / perimeter of parallelogram, trapezoid, or rhombus."""
    kind = random.choice(["parallelogram", "trapezoid", "rhombus_diag", "rhombus_side_h"])
    if kind == "parallelogram":
        b = rint(3, 15); h = rint(3, 15)
        return (f"Find the area of a parallelogram with base {b} and height {h}.",
                str(b * h), {"category": "geometry_quadrilateral",
                             "step_text": f"Step: A = b·h = {b}·{h} = {b*h}."})
    if kind == "trapezoid":
        b1 = rint(3, 12); b2 = rint(3, 12); h = rint(2, 10)
        if b1 == b2: b2 += 1
        area = Fraction((b1 + b2) * h, 2)
        return (f"Find the area of a trapezoid with parallel sides {b1} and {b2} and height {h}.",
                canon_num(area), {"category": "geometry_quadrilateral",
                                  "step_text": f"Step: A = (b1+b2)·h/2 = ({b1}+{b2})·{h}/2 = {canon_num(area)}."})
    if kind == "rhombus_diag":
        d1 = rint(4, 16, exclude_zero=True); d2 = rint(4, 16, exclude_zero=True)
        area = Fraction(abs(d1 * d2), 2)
        return (f"Find the area of a rhombus with diagonals {d1} and {d2}.",
                canon_num(area), {"category": "geometry_quadrilateral",
                                  "step_text": f"Step: A = d1·d2/2 = {d1}·{d2}/2 = {canon_num(area)}."})
    s = rint(3, 12); h = rint(2, s - 1) if s > 2 else 1
    return (f"Find the area of a rhombus with side length {s} and height {h}.",
            str(s * h), {"category": "geometry_quadrilateral",
                         "step_text": f"Step: A = s·h = {s}·{h} = {s*h} (rhombus is a parallelogram)."})


# ─────────────────────────── ELLIPSE & HYPERBOLA ───────────────────────────

def gen_conic_ellipse(dif):
    """Identify centre / vertices / foci / eccentricity of an ellipse from standard form."""
    h = rint(-5, 5); k = rint(-5, 5)
    # ensure a > b for orientation
    a = rint(3, 9); b = rint(2, a - 1)
    sub = random.choice(["centre_axes", "foci", "eccentricity"])
    sh = f"x − {h}" if h >= 0 else f"x + {-h}"
    sk = f"y − {k}" if k >= 0 else f"y + {-k}"
    eq = f"({sh})²/{a*a} + ({sk})²/{b*b} = 1"
    if sub == "centre_axes":
        return (f"For the ellipse {eq}, find the centre and the lengths of the semi-major and semi-minor axes.",
                f"centre ({h}, {k}); semi-major = {a}; semi-minor = {b}",
                {"category": "conic_ellipse",
                 "step_text": "Step: standard form (x−h)²/a² + (y−k)²/b² = 1 with a > b."})
    c_sq = a * a - b * b
    sq = math.isqrt(c_sq)
    c_str = str(sq) if sq * sq == c_sq else f"√{c_sq}"
    if sub == "foci":
        return (f"For the ellipse {eq}, find the foci (assuming the major axis is horizontal).",
                f"({h} ± {c_str}, {k})",
                {"category": "conic_ellipse",
                 "step_text": f"Step: c² = a² − b² = {c_sq}; foci at (h±c, k)."})
    # eccentricity = c / a
    if sq * sq == c_sq:
        ecc = canon_num(Fraction(sq, a))
    else:
        ecc = f"√{c_sq}/{a}"
    return (f"For the ellipse {eq}, find the eccentricity.",
            ecc, {"category": "conic_ellipse",
                  "step_text": f"Step: e = c/a = {c_str}/{a}."})


def gen_conic_hyperbola(dif):
    """Identify centre, vertices, foci, asymptotes of a hyperbola."""
    h = rint(-5, 5); k = rint(-5, 5)
    a = rint(2, 8); b = rint(2, 8)
    sh = f"x − {h}" if h >= 0 else f"x + {-h}"
    sk = f"y − {k}" if k >= 0 else f"y + {-k}"
    eq = f"({sh})²/{a*a} − ({sk})²/{b*b} = 1"
    sub = random.choice(["centre_vertices", "foci", "asymptotes"])
    if sub == "centre_vertices":
        return (f"For the hyperbola {eq}, find the centre and the vertices.",
                f"centre ({h}, {k}); vertices ({h} ± {a}, {k})",
                {"category": "conic_hyperbola",
                 "step_text": "Step: (x−h)²/a² − (y−k)²/b² = 1 has horizontal transverse axis; vertices at (h±a, k)."})
    if sub == "foci":
        c_sq = a * a + b * b
        sq = math.isqrt(c_sq)
        c_str = str(sq) if sq * sq == c_sq else f"√{c_sq}"
        return (f"For the hyperbola {eq}, find the foci.",
                f"({h} ± {c_str}, {k})",
                {"category": "conic_hyperbola",
                 "step_text": f"Step: c² = a² + b² = {c_sq}; foci at (h±c, k)."})
    return (f"For the hyperbola {eq}, find the asymptotes.",
            f"y − {k} = ±({b}/{a})(x − {h})",
            {"category": "conic_hyperbola",
             "step_text": f"Step: asymptotes of (x−h)²/a² − (y−k)²/b² = 1 are y−k = ±(b/a)(x−h)."})


# ─────────────────────────── 3-D COORDINATE GEOMETRY ───────────────────────────

def gen_geometry_distance_3d(dif):
    """3-D distance or midpoint."""
    p = [rint(-6, 6) for _ in range(3)]
    q = [rint(-6, 6) for _ in range(3)]
    if p == q:
        q[0] += 1
    if random.random() < 0.5:
        d2 = sum((q[i] - p[i]) ** 2 for i in range(3))
        sq = math.isqrt(d2)
        ans = str(sq) if sq * sq == d2 else f"√{d2}"
        return (f"Find the distance between the points {tuple(p)} and {tuple(q)}.",
                ans, {"category": "geometry_distance_3d",
                      "step_text": f"Step: d = √(Σ(qᵢ−pᵢ)²) = √{d2}."})
    mid = tuple(Fraction(p[i] + q[i], 2) for i in range(3))
    return (f"Find the midpoint of the segment from {tuple(p)} to {tuple(q)}.",
            "(" + ", ".join(canon_num(x) for x in mid) + ")",
            {"category": "geometry_distance_3d",
             "step_text": "Step: midpoint = ((x₁+x₂)/2, (y₁+y₂)/2, (z₁+z₂)/2)."})


# ─────────────────────────── TRIANGLE CENTROID ───────────────────────────

def gen_geometry_centroid(dif):
    """Centroid of a triangle from three vertices."""
    p = [rint(-6, 6), rint(-6, 6)]
    q = [rint(-6, 6), rint(-6, 6)]
    r = [rint(-6, 6), rint(-6, 6)]
    cx = Fraction(p[0] + q[0] + r[0], 3)
    cy = Fraction(p[1] + q[1] + r[1], 3)
    prob = f"Find the centroid of the triangle with vertices {tuple(p)}, {tuple(q)}, {tuple(r)}."
    ans = f"({canon_num(cx)}, {canon_num(cy)})"
    step = "Step: centroid = ((x₁+x₂+x₃)/3, (y₁+y₂+y₃)/3)."
    return prob, ans, {"category": "geometry_centroid", "step_text": step}

