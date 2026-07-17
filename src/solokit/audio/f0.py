"""F0 (fundamental frequency) tracking over an audio signal.

Ported from pymus/sisa/f0_tracking. Two methods:
- Peak tracking (Abeßer DAFX 2014): fast, monophonic, classic DSP
- pYIN: probabilistic, more accurate on noisy signals

Returns F0 in Hz at each frame, plus a confidence score.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class F0Track:
    """F0 estimate at each frame.

    Attributes:
        times: Frame times in seconds.
        f0_hz: F0 in Hz (NaN where no pitch detected).
        confidence: Per-frame confidence in [0, 1].
        voiced: Boolean mask of frames that contain pitch.
    """

    times: np.ndarray
    f0_hz: np.ndarray
    confidence: np.ndarray
    voiced: np.ndarray


def track_f0_peak(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_length: int = 2048,
    hop_length: int = 256,
    f0_min: float = 50.0,
    f0_max: float = 2000.0,
) -> F0Track:
    """Peak-tracking F0 estimator (Abeßer DAFX 2014).

    TODO: implement. The original pymus implementation uses STFT +
    harmonic product spectrum + peak picking. ~100 lines of DSP.
    """
    raise NotImplementedError("track_f0_peak is a stub — see docstring")


def track_f0_pyin(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_length: int = 2048,
    hop_length: int = 256,
    f0_min: float = 50.0,
    f0_max: float = 2000.0,
) -> F0Track:
    """pYIN F0 estimator.

    TODO: implement via librosa.pyin (which wraps a reference impl).
    """
    raise NotImplementedError("track_f0_pyin is a stub — see docstring")
