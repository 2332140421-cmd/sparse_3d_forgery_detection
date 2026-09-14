"""Five-time raw-state and multi-order sequence pilot.

This package is deliberately independent of the formal detection chain.  It
reads the already materialized 289-query ParticleSequence artifacts and writes
research-only results under the V7 data directory.
"""

from .representation import (
    CONDITIONS,
    TARGET_OFFSETS_S,
    build_five_time_unit,
    compute_derivatives,
    deterministic_permutation,
)

__all__ = [
    "CONDITIONS",
    "TARGET_OFFSETS_S",
    "build_five_time_unit",
    "compute_derivatives",
    "deterministic_permutation",
]
