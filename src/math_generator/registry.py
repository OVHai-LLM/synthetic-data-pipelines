"""Registry of all generators + capacity-aware picker.

Capacity-aware sampling
-----------------------
Each draw remembers how many *distinct* problems we've already gotten out of
each generator. When that count approaches the generator's known ceiling
(level-2 generators have very low ceilings — e.g. `derangement` only has 5
possible outputs), the generator's effective weight is scaled down so we
stop wasting draws on duplicates. Level-0 generators are treated as having
effectively unlimited capacity and never get throttled."""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Tuple

from .core import choice_weighted

# ─── existing modules ───
from . import (
    arithmetic, algebra, linear_algebra, analysis, number_theory,
    logic, boolean, geometry, stats, word, puzzles_z3, adversarial, trick,
)

GenFn = Callable[[str], Tuple[str, str, dict]]


# ─────────────────────────── HARDCODED LEVELS ───────────────────────────
# 0 = fully randomized, 1 = fixed prose / random params, 2 = curated list.

HARDCODED_LEVELS: Dict[str, int] = {
    # arithmetic
    "addition": 0, "subtraction": 0, "multiplication": 0, "division_exact": 0,
    "mixed_integer": 0, "fractions": 0, "linear_equation": 0, "gcd_lcm": 0,
    "compare": 0, "rounding": 0, "mod": 0, "prime_test": 0,
    "geometric_seq": 0, "sequence": 1,
    "addition_chain": 0, "multiplication_long": 0, "long_arith_chain": 0,
    # algebra
    "system_2x2": 0, "system_3x3": 0, "quadratic": 0, "cubic_roots": 0,
    "polynomial_eval": 0, "inequality": 0, "absolute_value": 0,
    "complex_arith": 0, "vector_ops": 0, "set_theory": 0,
    # matrices
    "matrix_add": 0, "matrix_mul": 0, "det_2x2": 0, "det_3x3": 0,
    "matrix_inverse": 0, "matrix_transpose": 0,
    "eigenvalue": 0, "eigenvector": 0, "lin_indep_2d": 0,
    # analysis (SymPy upgrades make these level 0)
    "trig_exact": 1, "trig_identity": 0, "trig_equation": 0, "trig_inverse": 0,
    "derivative_basic": 0, "derivative_poly": 0,
    "integral_indef": 0, "integral_def": 0, "limit": 0,
    "logarithm": 1, "exponential": 1,
    "arith_series": 1, "geom_series": 1, "taylor_series": 0,
    "function_domain": 1, "function_composition": 0, "function_inverse": 0,
    "ode_separable": 0, "ode_linear_first": 0,
    # number theory / bases / finance
    "prime_factorization": 0, "euler_totient": 0, "modular_arith": 0,
    "extended_euclidean": 0, "chinese_remainder": 0, "fermat_little": 0,
    "base_conv": 0, "sci_notation": 0,
    "simple_interest": 1, "compound_interest": 1,
    # logic
    "logic_prop": 1, "logic_quant": 0, "deduction_chain": 0,
    "deduction_validity": 1,
    "latex_simplify": 0, "latex_computation": 1, "proof_concept": 1,  # parameterised induction
    # boolean
    "bool_truth_table": 0, "bool_simplify": 0,
    "bool_sat": 0, "bool_tautology": 0, "bool_dnf": 0,
    # Z3 puzzles (NEW)
    "knights_knaves_z3": 0, "zebra_z3": 0, "sat_witness_z3": 0,
    # geometry
    "pythagoras": 1, "triangle_area": 1, "circle": 1, "polygon_angles": 1,
    "distance_2d": 0, "line_equation": 0,
    "vol_3d": 1, "surface_3d": 1,
    "conic_circle_eq": 0, "conic_parabola": 0,
    # NEW geometry
    "geom_heron": 0, "geom_triangle_ineq": 0,
    "geom_law_of_cos": 1, "geom_law_of_sines": 1,
    "geom_special_triangle": 1, "geom_sector": 1, "geom_quadrilateral": 1,
    "conic_ellipse": 0, "conic_hyperbola": 0,
    "geom_distance_3d": 0, "geom_centroid": 0,
    # stats / probability / combinatorics / graph
    "stats_descriptive": 0, "stats_variance": 0, "stats_range_iqr": 0,
    "probability": 0,  # randomised dice/coin/card scenarios with computed answers
    "prob_conditional": 1, "prob_bayes": 1, "prob_expected_value": 0,
    "combinatorics": 0, "incl_excl": 1, "pigeonhole": 1, "derangement": 2,
    "graph_degree_sum": 1, "graph_complete": 1, "graph_handshake": 0,
    # word problems
    "word_distance": 1, "word_work": 1, "word_mixture": 1,
    "word_age": 1, "word_percent": 1, "word_multi_step": 1,
    # NEW: adversarial / trick
    "adversarial_arith": 1, "trick_undefined": 1,
    "trick_no_solution": 1, "trick_ambiguous": 2,
}


# ─────────────────────────── CAPACITY ESTIMATES ───────────────────────────
# Coarse upper bound on distinct problems each generator can produce.
# Used for capacity-aware sampling. Picked by inspection — generous enough
# that level-0 generators stay effectively un-throttled.

# Estimated approximate ceiling per generator. None = effectively infinite.
GEN_CAPACITY: Dict[str, int] = {
    # Level-2 generators we actually know about:
    "derangement": 5,
    "trick_ambiguous": 10,        # expanded from 3 → 10 templates
    # Level-1 (semi-bounded prose templates):
    "limit": 200,
    "taylor_series": 200,
    "proof_concept": 60,          # 5 induction families × ~12 contexts
    "function_domain": 60,
    "graph_complete": 10,
    "pigeonhole": 30,
    "sequence": 1000,
    "deduction_validity": 140,
    "trig_exact": 80,
    "knights_knaves_z3": 100000,
    # All others: assume essentially infinite (None means unbounded).
}


# ─────────────────────────── REGISTRY DEFINITION ───────────────────────────

def _W(name: str, fn: GenFn, weight: float):
    return name, fn, weight


GENS_SIMPLE: List[Tuple[str, GenFn, float]] = [
    _W("addition",        arithmetic.gen_addition,        1.0),
    _W("subtraction",     arithmetic.gen_subtraction,     1.0),
    _W("multiplication",  arithmetic.gen_multiplication,  1.0),
    _W("division_exact",  arithmetic.gen_division_exact,  1.0),
    _W("addition_chain",      arithmetic.gen_addition_chain,        1.2),
    _W("multiplication_long", arithmetic.gen_multiplication_long,   1.2),
    _W("long_arith_chain",    arithmetic.gen_long_arith_chain,      1.2),
    _W("mixed_integer",   arithmetic.gen_mixed_expression, 0.7),
    _W("fractions",       arithmetic.gen_fraction_arith,  0.8),
    _W("linear_equation", arithmetic.gen_linear_equation, 1.0),
    _W("gcd_lcm",         arithmetic.gen_gcd_lcm,         0.8),
    _W("sequence",        arithmetic.gen_sequence_next,   0.8),
    _W("compare",         arithmetic.gen_compare,         0.8),
    _W("rounding",        arithmetic.gen_rounding,        0.6),
    _W("mod",             arithmetic.gen_modular,         0.6),
    _W("prime_test",      arithmetic.gen_prime_test,      0.5),
    _W("geometric_seq",   arithmetic.gen_geometric_sequence, 0.6),
    _W("pythagoras",      geometry.gen_geometry_pythagoras,    0.7),
    _W("triangle_area",   geometry.gen_geometry_triangle_area, 0.6),
    _W("circle",          geometry.gen_geometry_circle,        0.6),
    _W("polygon_angles",  geometry.gen_geometry_polygon_angles, 0.5),
    _W("distance_2d",     geometry.gen_geometry_distance_2d,   0.5),
    _W("vol_3d",          geometry.gen_geometry_3d_volume,     0.6),
    _W("surface_3d",      geometry.gen_geometry_3d_surface,    0.5),
    _W("stats_descriptive", stats.gen_stats_descriptive,       0.6),
    _W("arith_series",    analysis.gen_arith_series_sum,       0.5),
    _W("base_conv",       number_theory.gen_base_conversion,   0.6),
    _W("sci_notation",    number_theory.gen_scientific_notation, 0.5),
    _W("simple_interest", number_theory.gen_simple_interest,   0.5),
    _W("graph_complete",  stats.gen_graph_complete_edges,      0.4),
    _W("pigeonhole",      stats.gen_pigeonhole,                0.4),
    _W("word_distance",   word.gen_word_distance_rate,         0.6),
    _W("word_age",        word.gen_word_age,                   0.5),
    _W("word_percent",    word.gen_word_percent,               0.6),
]

GENS_MEDIUM: List[Tuple[str, GenFn, float]] = [
    _W("addition",         arithmetic.gen_addition,         0.5),
    _W("subtraction",      arithmetic.gen_subtraction,      0.5),
    _W("multiplication",   arithmetic.gen_multiplication,   0.5),
    _W("division_exact",   arithmetic.gen_division_exact,   0.5),
    _W("addition_chain",       arithmetic.gen_addition_chain,        1.0),
    _W("multiplication_long",  arithmetic.gen_multiplication_long,   1.0),
    _W("long_arith_chain",     arithmetic.gen_long_arith_chain,      1.0),
    _W("mixed_integer",    arithmetic.gen_mixed_expression, 0.8),
    _W("fractions",        arithmetic.gen_fraction_arith,   0.8),
    _W("linear_equation",  arithmetic.gen_linear_equation,  1.0),
    _W("system_2x2",       algebra.gen_system_2x2,          1.0),
    _W("quadratic",        algebra.gen_quadratic_complete,  1.0),
    _W("polynomial_eval",  algebra.gen_polynomial_eval,     0.8),
    _W("inequality",       algebra.gen_inequality_linear,   0.8),
    _W("absolute_value",   algebra.gen_absolute_value,      0.7),
    _W("trig_exact",       analysis.gen_trig_exact,         1.0),
    _W("trig_identity",    analysis.gen_trig_identity_sympy, 0.8),
    _W("trig_equation",    analysis.gen_trig_equation_sympy, 0.8),
    _W("trig_inverse",     analysis.gen_trig_inverse_sympy,  0.7),
    _W("logarithm",        analysis.gen_logarithm,           0.8),
    _W("combinatorics",    stats.gen_combinatorics,          0.8),
    _W("probability",      stats.gen_probability_basic,      0.5),
    _W("prime_factorization", number_theory.gen_prime_factorization, 0.7),
    _W("euler_totient",    number_theory.gen_euler_totient,  0.5),
    _W("complex_arith",    algebra.gen_complex_arithmetic,   0.8),
    _W("vector_ops",       algebra.gen_vector_ops,           0.8),
    _W("det_2x2",          linear_algebra.gen_determinant_2x2, 0.8),
    _W("matrix_add",       linear_algebra.gen_matrix_add,    0.7),
    _W("geometric_seq",    arithmetic.gen_geometric_sequence, 0.6),
    _W("set_theory",       algebra.gen_set_theory,           0.7),
    _W("logic_prop",       logic.gen_logic_propositional,    0.8),
    _W("logic_quant",      logic.gen_logic_quantifier_sympy, 0.7),
    _W("deduction_chain",  logic.gen_deduction_chain,        0.7),
    _W("deduction_validity", logic.gen_deduction_validity,   0.6),
    _W("knights_knaves_z3", puzzles_z3.gen_knights_knaves_z3, 0.7),
    _W("zebra_z3",         puzzles_z3.gen_zebra_z3,          0.6),
    _W("sat_witness_z3",   puzzles_z3.gen_sat_witness_z3,    0.6),
    _W("latex_simplify",   logic.gen_latex_simplify_sympy,   0.7),
    _W("bool_truth_table", boolean.gen_boolean_truth_table,  0.7),
    _W("bool_simplify",    boolean.gen_boolean_simplify_sympy, 0.6),
    _W("bool_sat",         boolean.gen_boolean_satisfiability, 0.6),
    _W("bool_tautology",   boolean.gen_boolean_tautology,    0.6),
    _W("pythagoras",       geometry.gen_geometry_pythagoras, 0.7),
    _W("triangle_area",    geometry.gen_geometry_triangle_area, 0.7),
    _W("circle",           geometry.gen_geometry_circle,     0.6),
    _W("polygon_angles",   geometry.gen_geometry_polygon_angles, 0.5),
    _W("distance_2d",      geometry.gen_geometry_distance_2d, 0.6),
    _W("line_equation",    geometry.gen_geometry_line_equation, 0.6),
    _W("vol_3d",           geometry.gen_geometry_3d_volume,  0.7),
    _W("surface_3d",       geometry.gen_geometry_3d_surface, 0.6),
    _W("stats_descriptive", stats.gen_stats_descriptive,     0.7),
    _W("stats_variance",   stats.gen_stats_variance,         0.6),
    _W("stats_range_iqr",  stats.gen_stats_range_iqr,        0.5),
    _W("prob_conditional", stats.gen_prob_conditional,       0.6),
    _W("arith_series",     analysis.gen_arith_series_sum,    0.6),
    _W("geom_series",      analysis.gen_geom_series_sum,     0.6),
    _W("function_domain",  analysis.gen_function_domain,     0.6),
    _W("function_composition", analysis.gen_function_composition, 0.6),
    _W("function_inverse", analysis.gen_function_inverse_linear, 0.6),
    _W("conic_circle_eq",  geometry.gen_conic_circle_eq,     0.5),
    _W("conic_parabola",   geometry.gen_conic_parabola,      0.5),
    # NEW geometry
    _W("geom_heron",         geometry.gen_geometry_heron,                0.6),
    _W("geom_triangle_ineq", geometry.gen_geometry_triangle_inequality,  0.5),
    _W("geom_law_of_cos",    geometry.gen_geometry_law_of_cosines,       0.6),
    _W("geom_law_of_sines",  geometry.gen_geometry_law_of_sines,         0.5),
    _W("geom_special_triangle", geometry.gen_geometry_special_triangle,  0.5),
    _W("geom_sector",        geometry.gen_geometry_sector,               0.5),
    _W("geom_quadrilateral", geometry.gen_geometry_quadrilateral,        0.6),
    _W("conic_ellipse",      geometry.gen_conic_ellipse,                 0.5),
    _W("conic_hyperbola",    geometry.gen_conic_hyperbola,               0.5),
    _W("geom_distance_3d",   geometry.gen_geometry_distance_3d,          0.5),
    _W("geom_centroid",      geometry.gen_geometry_centroid,             0.5),
    _W("word_distance",    word.gen_word_distance_rate,      0.6),
    _W("word_work",        word.gen_word_work_rate,          0.6),
    _W("word_mixture",     word.gen_word_mixture,            0.6),
    _W("word_age",         word.gen_word_age,                0.5),
    _W("word_percent",     word.gen_word_percent,            0.6),
    _W("word_multi_step",  word.gen_word_multi_step,         0.7),
    _W("base_conv",        number_theory.gen_base_conversion, 0.6),
    _W("sci_notation",     number_theory.gen_scientific_notation, 0.5),
    _W("simple_interest",  number_theory.gen_simple_interest, 0.6),
    _W("compound_interest", number_theory.gen_compound_interest, 0.6),
    _W("graph_degree_sum", stats.gen_graph_degree_sum,       0.5),
    _W("graph_complete",   stats.gen_graph_complete_edges,   0.5),
    _W("graph_handshake",  stats.gen_graph_handshake,        0.5),
    _W("incl_excl",        stats.gen_inclusion_exclusion,    0.6),
    _W("pigeonhole",       stats.gen_pigeonhole,             0.5),
    _W("derangement",      stats.gen_derangement,            0.4),
    _W("lin_indep_2d",     linear_algebra.gen_linear_independence_2d, 0.5),
    _W("adversarial_arith", adversarial.gen_adversarial_arithmetic, 0.5),
    _W("trick_undefined",  trick.gen_trick_undefined,        0.4),
    _W("trick_no_solution", trick.gen_trick_no_solution,     0.4),
    _W("trick_ambiguous",  trick.gen_trick_ambiguous,        0.2),
]

GENS_HARD: List[Tuple[str, GenFn, float]] = [
    _W("system_2x2",       algebra.gen_system_2x2,          0.8),
    _W("system_3x3",       algebra.gen_system_3x3,          0.8),
    _W("quadratic",        algebra.gen_quadratic_complete,  0.8),
    _W("cubic_roots",      algebra.gen_polynomial_cubic,    0.8),
    _W("polynomial_eval",  algebra.gen_polynomial_eval,     0.6),
    _W("inequality",       algebra.gen_inequality_linear,   0.6),
    _W("trig_identity",    analysis.gen_trig_identity_sympy, 0.8),
    _W("trig_equation",    analysis.gen_trig_equation_sympy, 0.8),
    _W("trig_inverse",     analysis.gen_trig_inverse_sympy,  0.7),
    _W("matrix_mul",       linear_algebra.gen_matrix_multiply, 1.0),
    _W("det_3x3",          linear_algebra.gen_determinant_3x3, 1.0),
    _W("matrix_inverse",   linear_algebra.gen_matrix_inverse_2x2, 0.8),
    _W("matrix_transpose", linear_algebra.gen_matrix_transpose, 0.6),
    _W("eigenvalue",       linear_algebra.gen_eigenvalue_2x2, 1.0),
    _W("eigenvector",      linear_algebra.gen_eigenvector_2x2, 0.8),
    _W("derivative_basic", analysis.gen_derivative_basic_sympy, 1.0),
    _W("derivative_poly",  analysis.gen_derivative_poly,    1.0),
    _W("integral_indef",   analysis.gen_integral_indefinite_sympy, 1.0),
    _W("integral_def",     analysis.gen_integral_definite,  1.0),
    _W("limit",            analysis.gen_limit_sympy,        0.8),
    _W("logarithm",        analysis.gen_logarithm,          0.7),
    _W("exponential",      analysis.gen_exponential,        0.7),
    _W("combinatorics",    stats.gen_combinatorics,         0.7),
    _W("modular_arith",    number_theory.gen_modular_arithmetic, 0.7),
    _W("complex_arith",    algebra.gen_complex_arithmetic,  0.8),
    _W("vector_ops",       algebra.gen_vector_ops,          0.8),
    _W("set_theory",       algebra.gen_set_theory,          0.6),
    _W("logic_prop",       logic.gen_logic_propositional,   0.7),
    _W("logic_quant",      logic.gen_logic_quantifier_sympy, 0.7),
    _W("deduction_chain",  logic.gen_deduction_chain,       0.7),
    _W("deduction_validity", logic.gen_deduction_validity,  0.5),
    _W("knights_knaves_z3", puzzles_z3.gen_knights_knaves_z3, 0.8),
    _W("zebra_z3",         puzzles_z3.gen_zebra_z3,         0.7),
    _W("sat_witness_z3",   puzzles_z3.gen_sat_witness_z3,   0.7),
    _W("proof_concept",    logic.gen_proof_concept,         0.4),
    _W("latex_simplify",   logic.gen_latex_simplify_sympy,  0.8),
    _W("latex_computation", logic.gen_latex_computation,    0.7),
    _W("bool_dnf",         boolean.gen_boolean_dnf,         0.6),
    _W("bool_sat",         boolean.gen_boolean_satisfiability, 0.6),
    _W("bool_tautology",   boolean.gen_boolean_tautology,   0.6),
    _W("bool_simplify",    boolean.gen_boolean_simplify_sympy, 0.5),
    _W("triangle_area",    geometry.gen_geometry_triangle_area, 0.6),
    _W("vol_3d",           geometry.gen_geometry_3d_volume, 0.5),
    _W("surface_3d",       geometry.gen_geometry_3d_surface, 0.5),
    _W("stats_descriptive", stats.gen_stats_descriptive,    0.5),
    _W("stats_variance",   stats.gen_stats_variance,        0.7),
    _W("prob_conditional", stats.gen_prob_conditional,      0.6),
    _W("prob_bayes",       stats.gen_prob_bayes,            0.7),
    _W("prob_expected_value", stats.gen_prob_expected_value, 0.7),
    _W("arith_series",     analysis.gen_arith_series_sum,   0.5),
    _W("geom_series",      analysis.gen_geom_series_sum,    0.7),
    _W("taylor_series",    analysis.gen_taylor_series_sympy, 0.6),
    _W("function_composition", analysis.gen_function_composition, 0.5),
    _W("function_inverse", analysis.gen_function_inverse_linear, 0.5),
    _W("function_domain",  analysis.gen_function_domain,    0.5),
    _W("ode_separable",    analysis.gen_ode_separable,      0.7),
    _W("ode_linear_first", analysis.gen_ode_linear_first,   0.7),
    _W("conic_circle_eq",  geometry.gen_conic_circle_eq,    0.5),
    _W("conic_parabola",   geometry.gen_conic_parabola,     0.5),
    # NEW geometry (hard tier)
    _W("geom_heron",         geometry.gen_geometry_heron,                0.6),
    _W("geom_law_of_cos",    geometry.gen_geometry_law_of_cosines,       0.7),
    _W("geom_law_of_sines",  geometry.gen_geometry_law_of_sines,         0.6),
    _W("geom_sector",        geometry.gen_geometry_sector,               0.5),
    _W("geom_quadrilateral", geometry.gen_geometry_quadrilateral,        0.5),
    _W("conic_ellipse",      geometry.gen_conic_ellipse,                 0.6),
    _W("conic_hyperbola",    geometry.gen_conic_hyperbola,               0.6),
    _W("geom_distance_3d",   geometry.gen_geometry_distance_3d,          0.5),
    _W("geom_centroid",      geometry.gen_geometry_centroid,             0.4),
    _W("word_work",        word.gen_word_work_rate,         0.5),
    _W("word_mixture",     word.gen_word_mixture,           0.5),
    _W("word_multi_step",  word.gen_word_multi_step,        0.8),
    _W("compound_interest", number_theory.gen_compound_interest, 0.6),
    _W("graph_handshake",  stats.gen_graph_handshake,       0.5),
    _W("graph_degree_sum", stats.gen_graph_degree_sum,      0.4),
    _W("extended_euclidean", number_theory.gen_extended_euclidean, 0.7),
    _W("chinese_remainder", number_theory.gen_chinese_remainder, 0.7),
    _W("fermat_little",    number_theory.gen_fermat_little, 0.6),
    _W("incl_excl",        stats.gen_inclusion_exclusion,   0.6),
    _W("derangement",      stats.gen_derangement,           0.5),
    _W("pigeonhole",       stats.gen_pigeonhole,            0.4),
    _W("lin_indep_2d",     linear_algebra.gen_linear_independence_2d, 0.5),
    _W("euler_totient",    number_theory.gen_euler_totient, 0.6),
    _W("prime_factorization", number_theory.gen_prime_factorization, 0.5),
    _W("geometric_seq",    arithmetic.gen_geometric_sequence, 0.6),
    _W("adversarial_arith", adversarial.gen_adversarial_arithmetic, 0.7),
    _W("trick_undefined",  trick.gen_trick_undefined,       0.5),
    _W("trick_no_solution", trick.gen_trick_no_solution,    0.5),
    _W("trick_ambiguous",  trick.gen_trick_ambiguous,       0.3),
]

REGISTRY: Dict[str, List[Tuple[str, GenFn, float]]] = {
    "simple": GENS_SIMPLE,
    "medium": GENS_MEDIUM,
    "hard":   GENS_HARD,
}


# ─────────────────────────── CAPACITY-AWARE PICKER ───────────────────────────

class CapacityAwareSampler:
    """Tracks per-generator distinct-output counts and throttles the weight of
    saturating generators. Use one sampler instance per run."""

    def __init__(self, capacity: Dict[str, int] = None):
        self.capacity = dict(GEN_CAPACITY)
        if capacity:
            self.capacity.update(capacity)
        self.distinct_count: Dict[str, int] = {}      # gen_name -> distinct outputs seen
        self.seen_problems: Dict[str, set] = {}       # gen_name -> set(problems)

    def record(self, name: str, problem: str) -> bool:
        """Returns True if problem is *new* for this generator."""
        bucket = self.seen_problems.setdefault(name, set())
        if problem in bucket:
            return False
        bucket.add(problem)
        self.distinct_count[name] = len(bucket)
        return True

    def saturation(self, name: str) -> float:
        """Fraction of the generator's capacity already covered (0 → 1.0)."""
        cap = self.capacity.get(name)
        if cap is None or cap <= 0:
            return 0.0
        return min(1.0, self.distinct_count.get(name, 0) / cap)

    def effective_weight(self, name: str, base_weight: float) -> float:
        sat = self.saturation(name)
        if sat >= 0.95:
            # >95% of capacity exhausted: cut to 5% of weight.
            return base_weight * 0.05
        if sat >= 0.7:
            # graceful taper: between 70% and 95% saturation, decay to 0.05x.
            return base_weight * (1 - sat) * 0.5
        return base_weight

    def pick(self, difficulty: str) -> Tuple[str, GenFn]:
        pool = REGISTRY[difficulty]
        weighted = [(name, fn, self.effective_weight(name, w)) for name, fn, w in pool]
        if all(w == 0 for _, _, w in weighted):
            # Everyone saturated: fall back to uniform among non-zero-base entries.
            weighted = [(n, fn, w) for n, fn, w in pool]
        name = choice_weighted([(n, w) for n, _, w in weighted])
        fn = next(f for n, f, _ in weighted if n == name)
        return name, fn


def saturation_report(sampler: CapacityAwareSampler) -> List[Tuple[str, int, int, float]]:
    """For each generator that produced anything, return (name, distinct, capacity, saturation)."""
    out = []
    for name, count in sorted(sampler.distinct_count.items()):
        cap = sampler.capacity.get(name)
        sat = sampler.saturation(name)
        out.append((name, count, cap if cap is not None else -1, sat))
    return out
