"""Rhythm-related features.

TODO: implement the full set:
- ioi_histogram (inter-onset intervals, optionally log-scaled)
- duration_histogram (note durations)
- density (notes per beat)
- sync_ratio (fraction of notes on strong beats)
- rhythmic_variability (CV of IOIs)
- phrase_length_stats
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from solokit.core.solo import Solo


def note_density(solo: Solo) -> float:
    """Notes per beat (averaged over the solo)."""
    total = solo.total_beats
    if total <= 0:
        return 0.0
    return len(solo.transcription.pitches) / total


def ioi_histogram(solo: Solo, bins: int = 16, log_scale: bool = True) -> np.ndarray:
    """Histogram of inter-onset intervals.

    With log_scale=True, the bin edges are log-spaced (better for the
    long tail of rests). Without it, linear from 0 to max(IOI).
    """
    notes = solo.transcription.notes
    if len(notes) < 2:
        return np.zeros(bins, dtype=np.int64)
    onsets = np.array([n.onset_beat for n in notes])
    iois = np.diff(onsets)
    iois = iois[iois > 0]
    if iois.size == 0:
        return np.zeros(bins, dtype=np.int64)
    if log_scale:
        edges = np.logspace(np.log10(max(iois.min(), 1e-3)), np.log10(iois.max()), bins + 1)
    else:
        edges = np.linspace(0, iois.max(), bins + 1)
    return np.histogram(iois, bins=edges)[0].astype(np.int64)


def duration_stats(solo: Solo) -> dict[str, float]:
    """Mean / median / std of note durations (in beats)."""
    durations = np.array(
        [n.duration_beats for n in solo.transcription.notes if not n.is_rest]
    )
    if durations.size == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0}
    return {
        "mean": float(durations.mean()),
        "median": float(np.median(durations)),
        "std": float(durations.std()),
    }


def rhythmic_variability(solo: Solo) -> float:
    """Coefficient of variation of inter-onset intervals.

    Higher = more rhythmically varied. 0 = all notes equally spaced.
    """
    notes = solo.transcription.notes
    if len(notes) < 3:
        return 0.0
    onsets = np.array([n.onset_beat for n in notes])
    iois = np.diff(onsets)
    iois = iois[iois > 0]
    if iois.size == 0 or iois.mean() == 0:
        return 0.0
    return float(iois.std() / iois.mean())
