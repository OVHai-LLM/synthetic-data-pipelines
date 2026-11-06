"""Statistics, probability, combinatorics, graph theory."""

import math
import random
from collections import Counter
from fractions import Fraction

from .core import canon_num, rint


# ─────────────────────────── DESCRIPTIVE STATISTICS ───────────────────────────

def gen_stats_descriptive(dif):
    """Mean / median / mode of a small integer list."""
    n = rint(5, 9)
    data = [rint(1, 20) for _ in range(n)]
    op = random.choice(["mean", "median", "mode"])
    if op == "mean":
        m = Fraction(sum(data), n)
        return (f"Find the mean of {data}.", canon_num(m),
                {"category": "stats_descriptive",
                 "step_text": f"Step: mean = sum/n = {sum(data)}/{n} = {canon_num(m)}."})
    if op == "median":
        s = sorted(data)
        med = Fraction(s[n // 2]) if n % 2 else Fraction(s[n // 2 - 1] + s[n // 2], 2)
        return (f"Find the median of {data}.", canon_num(med),
                {"category": "stats_descriptive",
                 "step_text": f"Step: sort → {s}, take the middle value(s)."})
    c = Counter(data)
    max_freq = max(c.values())
    if max_freq == 1:
        extra = random.choice(data)
        data.append(extra); c[extra] += 1; max_freq = c[extra]
    modes = sorted(v for v, f in c.items() if f == max_freq)
    return (f"Find the mode(s) of {data}.", ", ".join(map(str, modes)),
            {"category": "stats_descriptive",
             "step_text": f"Step: tally frequencies; pick most common (frequency {max_freq})."})


def gen_stats_variance(dif):
    """Population variance / std dev."""
    n = rint(4, 6)
    data = [rint(1, 12) for _ in range(n)]
    mean = Fraction(sum(data), n)
    var = sum((Fraction(x) - mean) ** 2 for x in data) / n
    if random.random() < 0.5:
        return (f"Find the population variance of {data}.", canon_num(var),
                {"category": "stats_variance",
                 "step_text": f"Step: mean = {canon_num(mean)}; variance = mean of squared deviations = {canon_num(var)}."})
    sd = math.sqrt(float(var))
    return (f"Find the population standard deviation of {data} (round to 3 decimals).",
            f"{sd:.3f}",
            {"category": "stats_variance",
             "step_text": f"Step: σ = √variance = √{canon_num(var)} ≈ {sd:.3f}."})


def gen_stats_range_iqr(dif):
    """Range / IQR of a small list."""
    n = rint(6, 9)
    data = sorted(rint(1, 30) for _ in range(n))
    if random.random() < 0.5:
        r = data[-1] - data[0]
        return (f"Find the range of {data}.", str(r),
                {"category": "stats_range_iqr",
                 "step_text": f"Step: range = max − min = {data[-1]} − {data[0]} = {r}."})
    q1 = data[n // 4]; q3 = data[(3 * n) // 4]
    return (f"Estimate the IQR of {data} (use index-based quartiles q1 = data[n/4], q3 = data[3n/4]).",
            str(q3 - q1),
            {"category": "stats_range_iqr",
             "step_text": f"Step: IQR = Q3 − Q1 = {q3} − {q1} = {q3-q1}."})


# ─────────────────────────── PROBABILITY ───────────────────────────

def gen_probability_basic(dif):
    """Random dice / coin / card probability problems with directly-computed answers."""
    import math as _math
    from itertools import product
    kind = random.choice(["dice_sum", "dice_inequality", "coin_heads", "card_draw"])

    if kind == "dice_sum":
        n_dice = random.choice([2, 2, 3])
        # Pick a target sum that's actually achievable
        target = random.randint(n_dice, 6 * n_dice)
        favourable = sum(1 for outcome in product(range(1, 7), repeat=n_dice)
                         if sum(outcome) == target)
        total = 6 ** n_dice
        p = Fraction(favourable, total)
        prob = (f"{n_dice} fair dice are rolled. What is the probability the sum equals {target}?")
        step = f"Step: enumerate the {total} outcomes; {favourable} of them sum to {target}."
        return prob, canon_num(p), {"category": "probability", "step_text": step}

    if kind == "dice_inequality":
        n_dice = random.choice([1, 2])
        op = random.choice(["≥", ">", "≤", "<"])
        target = random.randint(2, 6 * n_dice - 1)
        if op == "≥":   pred = lambda s: s >= target
        elif op == ">": pred = lambda s: s >  target
        elif op == "≤": pred = lambda s: s <= target
        else:           pred = lambda s: s <  target
        outcomes = list(product(range(1, 7), repeat=n_dice))
        favourable = sum(1 for o in outcomes if pred(sum(o)))
        p = Fraction(favourable, len(outcomes))
        plural = "die is" if n_dice == 1 else "dice are"
        prob = f"{n_dice} fair {plural} rolled. What is the probability the sum is {op} {target}?"
        step = f"Step: count outcomes with sum {op} {target}: {favourable} of {len(outcomes)}."
        return prob, canon_num(p), {"category": "probability", "step_text": step}

    if kind == "coin_heads":
        n_flips = random.randint(2, 6)
        target = random.randint(0, n_flips)
        favourable = _math.comb(n_flips, target)
        total = 2 ** n_flips
        p = Fraction(favourable, total)
        prob = f"{n_flips} fair coins are flipped. What is the probability of exactly {target} heads?"
        step = f"Step: P = C({n_flips},{target}) / 2^{n_flips} = {favourable}/{total}."
        return prob, canon_num(p), {"category": "probability", "step_text": step}

    # card_draw: probability of getting r cards of a chosen rank/suit when drawing k
    feature = random.choice(["rank", "suit", "face"])
    if feature == "rank":
        good, bad_total, label = 4, 52, "of a specific rank (e.g., Aces)"
    elif feature == "suit":
        good, bad_total, label = 13, 52, "of a specific suit (e.g., hearts)"
    else:
        good, bad_total, label = 12, 52, "a face card (J, Q, or K)"
    k = random.randint(1, 3)
    r = random.randint(1, k)
    if r > good:
        r = 1
    favourable = _math.comb(good, r) * _math.comb(bad_total - good, k - r)
    total = _math.comb(bad_total, k)
    p = Fraction(favourable, total)
    prob = (f"{k} cards are drawn at random (without replacement) from a standard 52-card deck. "
            f"What is the probability that exactly {r} of them is {label}?")
    step = (f"Step: P = C({good},{r})·C({bad_total-good},{k-r}) / C({bad_total},{k}) "
            f"= {favourable}/{total}.")
    return prob, canon_num(p), {"category": "probability", "step_text": step}


def gen_prob_conditional(dif):
    """Conditional probability via a 2-way table."""
    a, b, c, d = rint(5, 30), rint(5, 30), rint(5, 30), rint(5, 30)
    total = a + b + c + d
    pAandB = Fraction(a, total)
    pB = Fraction(a + c, total)
    p = pAandB / pB
    prob = (f"In a survey of {total} people, {a} like X and Y, {b} like X but not Y, "
            f"{c} like Y but not X, {d} like neither. If a respondent likes Y, "
            f"what is the probability they also like X?")
    step = f"Step: P(X|Y) = P(X∧Y)/P(Y) = {a}/{a+c} = {canon_num(p)}."
    return prob, canon_num(p), {"category": "prob_conditional", "step_text": step}


def gen_prob_bayes(dif):
    base = random.choice([1, 2, 5, 10])
    sens = random.choice([90, 95, 99])
    spec = random.choice([90, 95, 99])
    pD = Fraction(base, 100)
    pPosGivenD = Fraction(sens, 100)
    pPosGivenNotD = 1 - Fraction(spec, 100)
    pPos = pPosGivenD * pD + pPosGivenNotD * (1 - pD)
    pDgivenPos = (pPosGivenD * pD) / pPos
    prob = (f"A disease affects {base}% of a population. A test has sensitivity {sens}% "
            f"and specificity {spec}%. Given a positive test result, what is the probability "
            f"the person has the disease?")
    step = (f"Step: P(D|+) = P(+|D)P(D) / [P(+|D)P(D) + P(+|¬D)P(¬D)] "
            f"= {canon_num(pDgivenPos)}.")
    return prob, canon_num(pDgivenPos), {"category": "prob_bayes", "step_text": step}


def gen_prob_expected_value(dif):
    k = random.choice([3, 4])
    den = random.choice([6, 8, 10, 12])
    parts = [random.randint(1, den - (k - 1)) for _ in range(k - 1)]
    while sum(parts) >= den:
        parts = [max(1, p - 1) for p in parts]
    last = den - sum(parts)
    probs = parts + [last]
    random.shuffle(probs)
    values = random.sample(range(-5, 11), k)
    ev = sum(Fraction(p, den) * Fraction(v) for p, v in zip(probs, values))
    pairs = ", ".join(f"P(X={v})={p}/{den}" for v, p in zip(values, probs))
    prob = f"A random variable X has distribution: {pairs}. Find E[X]."
    return prob, canon_num(ev), {"category": "prob_expected_value",
                                 "step_text": "Step: E[X] = Σ x·P(X=x)."}


# ─────────────────────────── COMBINATORICS ───────────────────────────

def gen_combinatorics(dif):
    lo_n, hi_n = (3, 10) if dif == "medium" else (5, 15)
    n = rint(lo_n, hi_n)
    r = rint(1, n)
    if random.random() < 0.5:
        ans = math.perm(n, r)
        prob = f"How many ways to arrange {r} items chosen from {n} distinct items (order matters)? $P({n},{r}) = ?$"
        step = f"Step: P({n},{r}) = {n}!/({n}-{r})! = {ans}."
    else:
        ans = math.comb(n, r)
        prob = f"How many ways to choose {r} items from {n} distinct items (order doesn't matter)? $C({n},{r}) = ?$"
        step = f"Step: C({n},{r}) = {n}!/({r}!·({n}-{r})!) = {ans}."
    return prob, canon_num(ans), {"category": "combinatorics", "step_text": step}


def gen_inclusion_exclusion(dif):
    if random.random() < 0.5:
        a = rint(10, 30); b = rint(10, 30); ab = rint(1, min(a, b))
        u = a + b - ab
        return (f"In a class, {a} students take math, {b} take physics, and {ab} take both. "
                f"How many take at least one?",
                str(u), {"category": "inclusion_exclusion",
                         "step_text": f"Step: |A∪B| = |A|+|B|−|A∩B| = {a}+{b}−{ab} = {u}."})
    a = rint(10, 25); b = rint(10, 25); c = rint(10, 25)
    ab = rint(1, min(a, b)); ac = rint(1, min(a, c)); bc = rint(1, min(b, c))
    abc = rint(0, min(ab, ac, bc))
    u = a + b + c - ab - ac - bc + abc
    prob = (f"|A|={a}, |B|={b}, |C|={c}, |A∩B|={ab}, |A∩C|={ac}, |B∩C|={bc}, "
            f"|A∩B∩C|={abc}. Find |A∪B∪C|.")
    return prob, str(u), {"category": "inclusion_exclusion",
                          "step_text": f"Step: PIE: {a}+{b}+{c}−{ab}−{ac}−{bc}+{abc} = {u}."}


def gen_pigeonhole(dif):
    boxes = rint(3, 10)
    target = rint(2, 5)
    n_items = boxes * (target - 1) + 1
    prob = (f"What is the minimum number of items you must place into {boxes} boxes "
            f"to guarantee that some box contains at least {target} items?")
    step = (f"Step: by pigeonhole, you can fit at most {boxes}·({target}-1) = "
            f"{boxes*(target-1)} items without forcing it; one more triggers it.")
    return prob, str(n_items), {"category": "pigeonhole", "step_text": step}


def gen_derangement(dif):
    n = rint(3, 7)
    D = [1, 0]
    for i in range(2, n + 1):
        D.append((i - 1) * (D[-1] + D[-2]))
    prob = f"How many derangements (permutations with no fixed points) are there of {n} elements?"
    return prob, str(D[n]), {"category": "derangement",
                             "step_text": f"Step: D(n) = (n−1)·(D(n−1)+D(n−2)); D({n}) = {D[n]}."}


# ─────────────────────────── GRAPH THEORY ───────────────────────────

def gen_graph_degree_sum(dif):
    n = rint(4, 8)
    e = rint(n - 1, n * (n - 1) // 2)
    prob = f"A simple graph has {n} vertices and {e} edges. What is the sum of all vertex degrees?"
    return prob, str(2 * e), {"category": "graph_degree_sum",
                              "step_text": f"Step: by the handshaking lemma, sum of degrees = 2|E| = 2·{e} = {2*e}."}


def gen_graph_complete_edges(dif):
    n = rint(3, 12)
    e = n * (n - 1) // 2
    return (f"How many edges does the complete graph K_{n} have?",
            str(e), {"category": "graph_complete_edges",
                     "step_text": f"Step: |E(K_n)| = n(n−1)/2 = {n}·{n-1}/2 = {e}."})


def gen_graph_handshake(dif):
    n = rint(5, 12)
    deg = sorted(rint(1, n - 1) for _ in range(n))
    if sum(deg) % 2:
        deg[-1] += 1 if deg[-1] < n - 1 else -1
    e = sum(deg) // 2
    return (f"A simple graph has degree sequence {deg}. How many edges does it have?",
            str(e), {"category": "graph_handshake",
                     "step_text": f"Step: sum of degrees = {sum(deg)}; edges = sum/2 = {e}."})
