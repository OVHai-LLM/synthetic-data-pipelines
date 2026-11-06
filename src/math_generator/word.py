"""Word problems — distance/rate, work-rate, mixture, age, percent, plus
a multi-step composer that chains two concepts."""

import random
from fractions import Fraction

from .core import StepBuilder, canon_num, rint


def gen_word_distance_rate(dif):
    """Distance = rate × time."""
    r = rint(20, 80); t = rint(2, 8)
    d = r * t
    op = random.choice(["distance", "rate", "time"])
    if op == "distance":
        return (f"A car travels at {r} km/h for {t} hours. How far does it go?",
                f"{d} km", {"category": "word_distance_rate",
                            "step_text": f"Step: d = r·t = {r}·{t} = {d}."})
    if op == "rate":
        return (f"A car travels {d} km in {t} hours. What is its average speed?",
                f"{r} km/h", {"category": "word_distance_rate",
                              "step_text": f"Step: r = d/t = {d}/{t} = {r}."})
    return (f"A car travels {d} km at {r} km/h. How long does the trip take?",
            f"{t} h", {"category": "word_distance_rate",
                       "step_text": f"Step: t = d/r = {d}/{r} = {t}."})


def gen_word_work_rate(dif):
    a = rint(2, 10); b = rint(2, 10)
    if a == b: b += 1
    t = Fraction(a * b, a + b)
    prob = (f"Worker A finishes a job in {a} hours; Worker B finishes the same job in {b} hours. "
            f"How long do they need to finish it together?")
    step = (f"Step: combined rate = 1/{a} + 1/{b} = {canon_num(Fraction(1,a)+Fraction(1,b))} jobs/hr; "
            f"time = 1/rate = {canon_num(t)}.")
    return prob, f"{canon_num(t)} hours", {"category": "word_work_rate", "step_text": step}


def gen_word_mixture(dif):
    p1 = random.choice([10, 20, 30, 40])
    p2 = random.choice([60, 70, 80, 90])
    L1 = rint(2, 10)
    L2 = rint(2, 10)
    target = Fraction(p1 * L1 + p2 * L2, L1 + L2)
    prob = (f"You mix {L1} liters of {p1}% solution with {L2} liters of {p2}% solution. "
            f"What is the concentration of the resulting mixture?")
    step = f"Step: ({p1}·{L1} + {p2}·{L2})/({L1}+{L2}) = {canon_num(target)}%."
    return prob, f"{canon_num(target)}%", {"category": "word_mixture", "step_text": step}


def gen_word_age(dif):
    age_now = rint(8, 30)
    diff = rint(2, 15)
    yrs = rint(2, 10)
    sib_now = age_now - diff
    if sib_now < 1:
        sib_now = age_now - 1; diff = 1
    diff_word = "year" if diff == 1 else "years"
    yrs_word = "year" if yrs == 1 else "years"
    prob = (f"Anna is {age_now} years old. Her brother is {diff} {diff_word} younger. "
            f"How old will her brother be in {yrs} {yrs_word}?")
    step = f"Step: brother now = {age_now} − {diff} = {sib_now}; in {yrs} {yrs_word} he is {sib_now+yrs}."
    return prob, str(sib_now + yrs), {"category": "word_age", "step_text": step}


def gen_word_percent(dif):
    op = random.choice(["discount", "tax", "increase"])
    base = rint(10, 500)
    pct = random.choice([5, 8, 10, 12, 15, 20, 25, 30])
    if op == "discount":
        new = Fraction(base) * Fraction(100 - pct, 100)
        prob = f"A jacket costs ${base}. With a {pct}% discount, what is the new price?"
        step = (f"Step: discount = {pct}% of {base} = {canon_num(Fraction(base*pct,100))}; "
                f"price = {base} − that = {canon_num(new)}.")
        return prob, f"${canon_num(new)}", {"category": "word_percent", "step_text": step}
    if op == "tax":
        new = Fraction(base) * Fraction(100 + pct, 100)
        return (f"An item costs ${base} before {pct}% sales tax. What is the total?",
                f"${canon_num(new)}", {"category": "word_percent",
                                       "step_text": f"Step: total = {base}·(1 + {pct}/100) = {canon_num(new)}."})
    new = Fraction(base) * Fraction(100 + pct, 100)
    return (f"A salary of ${base} is increased by {pct}%. What is the new salary?",
            f"${canon_num(new)}", {"category": "word_percent",
                                   "step_text": f"Step: new = {base}·(1+{pct}/100) = {canon_num(new)}."})


# ─────────────────────────── MULTI-STEP COMPOSER ───────────────────────────

def gen_word_multi_step(dif):
    """Compose two concepts into a multi-step word problem with a numbered chain-of-thought."""
    kind = random.choice(["dist_pct", "mix_dilute", "work_remain",
                          "ci_withdraw", "age_speed"])
    sb = StepBuilder()

    if kind == "dist_pct":
        v = rint(40, 80); t1 = rint(2, 5)
        pct = random.choice([10, 20, 25, 30])
        t2 = rint(2, 5)
        new_v = Fraction(v) * (1 + Fraction(pct, 100))
        d2 = new_v * t2
        prob = (f"A car travels at {v} km/h for {t1} hours, then increases its speed by {pct}%. "
                f"How far does it travel in the next {t2} hours at the new speed?")
        sb.add(f"Phase-1 distance: d₁ = v·t₁ = {v}·{t1} = {v*t1} km (not asked).")
        sb.add(f"Compute new speed: v' = {v}·(1 + {pct}/100) = {canon_num(new_v)} km/h.")
        sb.add(f"Phase-2 distance: d₂ = v'·t₂ = {canon_num(new_v)}·{t2} = {canon_num(d2)} km.")
        return prob, f"{canon_num(d2)} km", {"category": "word_multi_step", "step_text": sb.text()}

    if kind == "mix_dilute":
        p1 = random.choice([10, 20, 30]); p2 = random.choice([60, 70, 80])
        L1 = rint(2, 10); L2 = rint(2, 10); L3 = rint(2, 8)
        solute = p1 * L1 + p2 * L2
        total = L1 + L2 + L3
        pct = Fraction(solute, total)
        prob = (f"You mix {L1} L of {p1}% solution with {L2} L of {p2}% solution, "
                f"then add {L3} L of pure water. What is the final concentration?")
        sb.add(f"Total solute: {p1}·{L1} + {p2}·{L2} = {solute} units.")
        sb.add(f"Total volume after dilution: {L1}+{L2}+{L3} = {total} L.")
        sb.add(f"Concentration: {solute}/{total} = {canon_num(pct)}%.")
        return prob, f"{canon_num(pct)}%", {"category": "word_multi_step", "step_text": sb.text()}

    if kind == "work_remain":
        a = rint(3, 10); b = rint(3, 10)
        if a == b: b += 1
        t = rint(1, min(a, b) - 1)
        rate = Fraction(1, a) + Fraction(1, b)
        done = rate * t
        remain = 1 - done
        prob = (f"Worker A finishes a job in {a} hours; Worker B in {b} hours. "
                f"They work together for {t} hours. What fraction of the job remains?")
        sb.add(f"Combined rate: 1/{a} + 1/{b} = {canon_num(rate)} jobs/hour.")
        sb.add(f"Work completed in {t} h: {canon_num(rate)}·{t} = {canon_num(done)}.")
        sb.add(f"Remaining: 1 − {canon_num(done)} = {canon_num(remain)}.")
        return prob, canon_num(remain), {"category": "word_multi_step", "step_text": sb.text()}

    if kind == "ci_withdraw":
        P = rint(1000, 5000)
        r = random.choice([5, 10, 15, 20])
        t = rint(2, 4)
        wd = random.choice([10, 25, 30, 50])
        factor = Fraction(100 + r, 100)
        Aval = Fraction(P) * (factor ** t)
        kept = Aval * Fraction(100 - wd, 100)
        rounded = round(kept * 100) / 100
        prob = (f"You invest ${P} at {r}% compounded annually for {t} years, then withdraw "
                f"{wd}% of the resulting balance. How much remains? (round to cents)")
        sb.add(f"Balance after {t} years: ${P}·(1+{r}/100)^{t} ≈ ${round(float(Aval), 2)}.")
        sb.add(f"Withdraw {wd}%: ${round(float(Aval) * wd / 100, 2)}.")
        sb.add(f"Remaining: ${rounded:.2f}.")
        return prob, f"${rounded:.2f}", {"category": "word_multi_step", "step_text": sb.text()}

    # age_speed
    age = rint(8, 15); speed = rint(4, 10); yrs = rint(2, 6)
    factor = random.choice([Fraction(11, 10), Fraction(6, 5), Fraction(5, 4), Fraction(3, 2)])
    new_speed = Fraction(speed) * factor
    t = rint(1, 4)
    dist = new_speed * t
    prob = (f"A {age}-year-old runs at {speed} km/h. In {yrs} years she will have improved her speed "
            f"by a factor of {canon_num(factor)}. At her future speed, how far can she run in {t} hours?")
    sb.add(f"Future speed: {speed}·{canon_num(factor)} = {canon_num(new_speed)} km/h.")
    sb.add(f"Future age (incidental): {age + yrs} years.")
    sb.add(f"Distance: {canon_num(new_speed)}·{t} = {canon_num(dist)} km.")
    return prob, f"{canon_num(dist)} km", {"category": "word_multi_step", "step_text": sb.text()}
