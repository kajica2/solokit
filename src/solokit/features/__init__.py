"""YAML-driven feature extraction.

The `FeatureMachine` reads a YAML config describing which features to
compute over a Solo, and how to combine them. Feature functions live in
`solokit.features.pitch`, `solokit.features.rhythm`, etc. — they're
plain Python functions you can also call directly.

The pattern is borrowed from the original MeloSpyLib's Feature Machine
but cleaner: Pydantic for the config schema, registry-based dispatch.
"""

from solokit.features.machine import FeatureConfig, FeatureMachine, FeatureResult, MachineConfig

__all__ = [
    "FeatureConfig",
    "FeatureMachine",
    "FeatureResult",
    "MachineConfig",
]
