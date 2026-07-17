"""The Feature Machine — YAML-driven feature extraction.

You define features in a YAML file like:

    name: jazz_basic
    features:
      - name: pitch_class_histogram
        type: histogram
        function: solokit.features.pitch:pitch_class_histogram
      - name: ioi_histogram
        type: histogram
        function: solokit.features.rhythm:ioi_histogram
        bins: 16
        log_scale: true

The machine loads this config, looks up each function by its importable
path, runs it over a Solo, and returns a dict of {feature_name: value}.

Why YAML?
    - Researchers can define new features without writing Python.
    - Features become shareable artifacts (commit a YAML, run a study).
    - Same pattern as the original MeloSpyLib Feature Machine.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import yaml

FeatureFn: TypeAlias = Callable[..., Any]
FeatureResult: TypeAlias = dict[str, Any]


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """A single feature definition loaded from YAML.

    Attributes:
        name: Human-readable name used as the result key.
        function: Importable path like "solokit.features.pitch:pitch_class_histogram".
        type: One of "scalar", "histogram", "array".
        args: Keyword arguments passed to the function.
    """

    name: str
    function: str
    type: str = "scalar"
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MachineConfig:
    """Top-level YAML config for the FeatureMachine."""

    name: str
    features: tuple[FeatureConfig, ...]

    @classmethod
    def from_yaml(cls, path: str | Path) -> MachineConfig:
        """Load a MachineConfig from a YAML file."""
        data = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MachineConfig:
        """Build a MachineConfig from a parsed YAML dict."""
        if "features" not in data:
            msg = f"YAML config missing 'features' key; got keys: {list(data)}"
            raise ValueError(msg)
        features = tuple(FeatureConfig(**feat) for feat in data["features"])
        return cls(name=data.get("name", "unnamed"), features=features)


class FeatureMachine:
    """The Feature Machine — applies a YAML config to a Solo.

    Lookup is by importable path: "package.module:function_name". We
    cache resolved functions so re-runs are fast.
    """

    __slots__ = ("config", "_registry")

    def __init__(self, config: MachineConfig) -> None:
        self.config = config
        self._registry: dict[str, FeatureFn] = {}

    @classmethod
    def from_yaml(cls, path: str | Path) -> FeatureMachine:
        """Build a FeatureMachine from a YAML config file."""
        return cls(MachineConfig.from_yaml(path))

    def _resolve(self, path: str) -> FeatureFn:
        if path in self._registry:
            return self._registry[path]
        try:
            mod_name, func_name = path.split(":")
        except ValueError as exc:
            msg = (
                f"Function path {path!r} must be 'package.module:function_name'"
            )
            raise ValueError(msg) from exc
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, func_name)
        if not callable(fn):
            msg = f"{path!r} resolved to {fn!r}, not a callable"
            raise TypeError(msg)
        self._registry[path] = fn
        return fn

    def extract(self, solo) -> FeatureResult:  # noqa: ANN001 — avoid circular import
        """Run every configured feature over the Solo.

        Returns a dict {feature_name: result}. Each feature function
        is called as `fn(solo, **args)`. The first arg is the Solo.
        """
        results: FeatureResult = {}
        for feat in self.config.features:
            fn = self._resolve(feat.function)
            try:
                results[feat.name] = fn(solo, **feat.args)
            except Exception as exc:  # noqa: BLE001 — surface as result, don't crash the run
                results[feat.name] = {"error": str(exc), "type": type(exc).__name__}
        return results

    def __call__(self, solo) -> FeatureResult:  # noqa: ANN001
        return self.extract(solo)
