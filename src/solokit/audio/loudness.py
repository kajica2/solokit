"""Loudness estimation per note.

Given audio + a transcription, estimate how loud each note was played.
Useful for performance analysis (dynamics, accents) and for expressive
pattern search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from solokit.core.transcription import Transcription


@dataclass(frozen=True, slots=True)
class LoudnessTrack:
    """Per-note loudness estimate.

    Attributes:
        notes: The input notes (one row per note).
        loudness_db: Loudness in dB FS for each note's duration.
        peak_db: Peak dB FS within each note.
        rms_db: RMS dB FS over each note's duration.
    """

    notes: tuple  # tuple[NoteEvent, ...]
    loudness_db: np.ndarray
    peak_db: np.ndarray
    rms_db: np.ndarray


def estimate_loudness_critical_band(
    audio: np.ndarray,
    sample_rate: int,
    transcription: Transcription,
) -> LoudnessTrack:
    """Critical-band loudness estimator (port from pymus).

    TODO: implement. The original uses an approximation of Moore &
    Glasberg's loudness model suitable for monophonic signals.
    """
    raise NotImplementedError("estimate_loudness_critical_band is a stub")
