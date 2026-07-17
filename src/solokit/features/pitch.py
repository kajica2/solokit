"""Pitch-related features.

These are the leaf functions the Feature Machine dispatches to. Each
takes a Solo as its first argument and returns a scalar / histogram /
array depending on the feature.

TODO: flesh these out with the full jazzomat feature set:
- pitch_class_histogram (12-bin chromatic)
- pitch_class_distribution (normalized)
- interval_histogram (semitone differences, signed, -12..+12)
- tonalness (Krumhansl-Schmuckler key profile correlation)
- contour (Huron-style: horizontal / ascending / descending / concave / convex)
- pitch_range (max - min)
- chromaticism ratio (non-scale-tone fraction)
- blue-note frequency (b3, b5, b7 emphasis)
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from solokit.core.solo import Solo


def pitch_class_histogram(solo: Solo, bins: int = 12) -> np.ndarray:
    """12-bin chromatic pitch-class distribution.

    Returns a length-12 array of occurrence counts (not normalized).
    C=0, C#=1, ..., B=11.
    """
    pitches = solo.transcription.pitches
    if not pitches:
        return np.zeros(bins, dtype=np.int64)
    pc = np.array([p % bins for p in pitches], dtype=np.int64)
    return np.bincount(pc, minlength=bins)


def pitch_range(solo: Solo) -> int:
    """Range in semitones (max - min) of the solo's pitched notes.

    Returns 0 for an all-rest solo.
    """
    pitches = solo.transcription.pitches
    if not pitches:
        return 0
    return int(max(pitches)) - int(min(pitches))


def chromaticism_ratio(solo: Solo) -> float:
    """Fraction of notes that lie outside the diatonic scale.

    Very rough: assumes C major / A minor (no key info used yet).
    For better results, pass the key in and use proper scale templates.
    """
    pitches = solo.transcription.pitches
    if not pitches:
        return 0.0
    diatonic = {0, 2, 4, 5, 7, 9, 11}  # C major
    out_of_scale = sum(1 for p in pitches if (p % 12) not in diatonic)
    return out_of_scale / len(pitches)


def pitch_class_distribution(solo: Solo) -> dict[int, float]:
    """Normalized pitch-class distribution as a dict {0..11: proportion}.

    Useful when you want named keys (C, C#, ...) rather than array indices.
    """
    hist = pitch_class_histogram(solo)
    total = hist.sum()
    if total == 0:
        return {i: 0.0 for i in range(12)}
    return {i: float(hist[i]) / float(total) for i in range(12)}


# Aliases used by the original MeloSpyLib feature names
def pc_histogram(solo: Solo) -> np.ndarray:
    """Alias for pitch_class_histogram (matches MeloSpyLib naming)."""
    return pitch_class_histogram(solo)
