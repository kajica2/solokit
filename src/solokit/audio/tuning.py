"""Tuning frequency estimation.

Given a recording, estimate how far the instrument is tuned from
A=440Hz. Useful for matching old vinyl rips or non-standard
instrument tunings (baritone sax, etc.).
"""

from __future__ import annotations

import numpy as np


def estimate_tuning_mauch(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_length: int = 8192,
) -> float:
    """Estimate tuning offset from A=440 in cents.

    Uses the NNLS chroma method (Mauch 2010). Returns a value in
    [-50, +50] cents (most instruments tune within ±50 cents of
    standard).

    TODO: implement via librosa or the original pymus NNLS code.
    """
    raise NotImplementedError("estimate_tuning_mauch is a stub")


def retune_transcription_to_a440(
    transcription,  # noqa: ANN001 — accept any transcription-like
    cents_offset: float,
) -> None:
    """Apply a tuning offset (in cents) to a Transcription in place.

    Rounds each MIDI pitch to the nearest integer after the offset.
    """
    raise NotImplementedError("retune_transcription_to_a440 is a stub")
