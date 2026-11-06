"""Math problem generator for LLM training.

Run as `python -m math_generator --n 10000 --out math_train.jsonl`.
See `registry.py` for the full list of generators and capacity-aware sampling."""

from .registry import (
    REGISTRY, HARDCODED_LEVELS, GEN_CAPACITY,
    CapacityAwareSampler, saturation_report,
)

__all__ = [
    "REGISTRY", "HARDCODED_LEVELS", "GEN_CAPACITY",
    "CapacityAwareSampler", "saturation_report",
]
