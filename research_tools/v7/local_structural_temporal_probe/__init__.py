"""Fixed local-relation temporal supervised pilot tooling for V7."""

from .representation import (
    ARM_NAMES,
    COMPONENT_CONFIG,
    TARGET_OFFSETS_S,
    build_window_support,
    arm_inputs_for_triplet,
    compute_local_derivatives,
)

__all__ = [
    "ARM_NAMES",
    "COMPONENT_CONFIG",
    "TARGET_OFFSETS_S",
    "arm_inputs_for_triplet",
    "build_window_support",
    "compute_local_derivatives",
]
