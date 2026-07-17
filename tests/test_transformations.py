"""Tests for the four pattern transformations."""

from __future__ import annotations

import pytest

from solokit.patterns.transformations import (
    cdpcx,
    fuzzy_interval,
    interval,
    pitch,
    transform,
)


class TestPitch:
    def test_identity(self) -> None:
        assert pitch([60, 62, 64]) == [60, 62, 64]

    def test_empty(self) -> None:
        assert pitch([]) == []


class TestInterval:
    def test_major_scale_up(self) -> None:
        # C major scale
        assert interval([60, 62, 64, 65, 67, 69, 72]) == [2, 2, 1, 2, 2, 3]

    def test_desending(self) -> None:
        assert interval([72, 71, 70]) == [-1, -1]

    def test_chromatic_unchanged(self) -> None:
        assert interval([60, 61, 62, 63]) == [1, 1, 1]

    def test_too_short(self) -> None:
        assert interval([60]) == []

    def test_transposition_invariant(self) -> None:
        # Same pattern in Bb should produce same intervals as in C
        assert interval([60, 62, 64]) == interval([58, 60, 62])


class TestFuzzyInterval:
    def test_merges_semitone_and_tone(self) -> None:
        # 1 semitone and 2 semitones should be different magnitudes
        result = fuzzy_interval([60, 61, 63])
        # [60, 61] = 1 → +1; [61, 63] = 2 → +2
        assert result == [1, 2]

    def test_unison(self) -> None:
        assert fuzzy_interval([60, 60, 60]) == [0, 0]


class TestCDPCX:
    def test_c_major_scale_degrees(self) -> None:
        # C major scale over C chord → degrees 0,1,2,3,4,5,6
        result = cdpcx([60, 62, 64, 65, 67, 69, 71], chord_roots=[0] * 7)
        assert result == [0, 1, 2, 3, 4, 5, 6]

    def test_non_chord_tone_marked(self) -> None:
        # F# over C chord → not in scale → -1
        result = cdpcx([66], chord_roots=[0])
        assert result == [-1]

    def test_minor_chord(self) -> None:
        # A minor scale (A, B, C, D, E, F, G) over A chord → degrees 0..6
        # A=57, B=59, C=60, D=62, E=64, F=65, G=67
        result = cdpcx(
            [57, 59, 60, 62, 64, 65, 67],
            chord_roots=[9] * 7,  # A = MIDI 9 (root class)
            chord_qualities=["m7"] * 7,
        )
        assert result == [0, 1, 2, 3, 4, 5, 6]


class TestTransform:
    def test_dispatch_pitch(self) -> None:
        assert transform([60, 62, 64], "pitch") == [60, 62, 64]

    def test_dispatch_interval(self) -> None:
        assert transform([60, 62, 64], "interval") == [2, 2]

    def test_dispatch_fuzzy(self) -> None:
        assert transform([60, 61, 63], "fuzzyinterval") == [1, 2]

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown transformation"):
            transform([60, 62], "nonsense")  # type: ignore[arg-type]
