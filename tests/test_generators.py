"""Every registered math generator returns a well-formed (problem, answer, meta) triple."""
import pytest

from math_generator import _tests


def test_all_generators_well_formed():
    try:
        _tests.main()
    except SystemExit as e:  # _tests.main() exits 1 on any failure
        pytest.fail(f"generator smoke test failed (exit {e.code})")
