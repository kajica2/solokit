"""Tests for the Feature Machine and individual feature functions."""

from __future__ import annotations

from pathlib import Path

import pytest

from solokit.features import FeatureConfig, FeatureMachine, MachineConfig
from solokit.features.pitch import (
    chromaticism_ratio,
    pitch_class_histogram,
    pitch_class_distribution,
    pitch_range,
)
from solokit.features.rhythm import (
    duration_stats,
    note_density,
    rhythmic_variability,
)


class TestPitchFeatures:
    def test_pitch_class_histogram_basic(self, sample_solo) -> None:
        hist = pitch_class_histogram(sample_solo)
        assert hist.shape == (12,)
        assert hist.sum() == 9  # 9 notes in sample_solo

    def test_pitch_class_histogram_all_in_scale(self, sample_solo) -> None:
        # All notes are in C major → 0 chromaticism
        hist = pitch_class_histogram(sample_solo)
        # Sample solo: C, D, E, F, G, F, E, D, C → PC: 0, 2, 4, 5, 7, 5, 4, 2, 0
        assert hist[0] == 2
        assert hist[2] == 2
        assert hist[4] == 2
        assert hist[5] == 2
        assert hist[7] == 1

    def test_pitch_range(self, sample_solo) -> None:
        # C4 (60) to G4 (67) → range = 7
        assert pitch_range(sample_solo) == 7

    def test_chromaticism_ratio(self, sample_solo) -> None:
        # All C major → ratio = 0
        assert chromaticism_ratio(sample_solo) == 0.0

    def test_pitch_class_distribution_sums_to_one(self, sample_solo) -> None:
        dist = pitch_class_distribution(sample_solo)
        assert sum(dist.values()) == pytest.approx(1.0)


class TestRhythmFeatures:
    def test_note_density(self, sample_solo) -> None:
        # 9 notes over 10 beats → 0.9
        assert note_density(sample_solo) == pytest.approx(0.9)

    def test_duration_stats(self, sample_solo) -> None:
        stats = duration_stats(sample_solo)
        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats

    def test_rhythmic_variability_uniform(self, sample_solo) -> None:
        # All notes 1 beat apart → variability = 0
        assert rhythmic_variability(sample_solo) == 0.0


class TestFeatureMachine:
    def test_load_from_yaml(self) -> None:
        path = Path(__file__).parent.parent / "features" / "basic.yaml"
        if not path.exists():
            pytest.skip("features/basic.yaml not found")
        machine = FeatureMachine.from_yaml(path)
        assert machine.config.name == "basic_jazz_features"
        assert len(machine.config.features) > 0

    def test_extract_runs_all_features(self, sample_solo) -> None:
        # Inline config for a quick test
        config = MachineConfig(
            name="test",
            features=(
                FeatureConfig(
                    name="range",
                    function="solokit.features.pitch:pitch_range",
                ),
            ),
        )
        machine = FeatureMachine(config)
        result = machine.extract(sample_solo)
        assert "range" in result
        assert result["range"] == 7

    def test_invalid_function_path(self, sample_solo) -> None:
        config = MachineConfig(
            name="bad",
            features=(FeatureConfig(name="x", function="not:a:real:path"),),
        )
        machine = FeatureMachine(config)
        with pytest.raises(ValueError):
            machine.extract(sample_solo)
