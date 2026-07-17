"""Phrase segmentation — split a Solo's transcription into phrases.

The original pymus/melospy stack offered several segmentation strategies.
This module is a stub for now; implementations live in `_strategies.py`
once you fill them in. Common approaches:

- Rest-based: split at long rests (> threshold beats)
- Length-based: cap phrases at a max beat count + max note count
- Hybrid: rest + length fallback
- ML-based: use a trained model to find phrase boundaries

References:
- Pfleiderer, Frieler (2010) — "The Jazzomat project: Issues and methods"
- Cambouropoulos (2006) — "Towards a General Computational Theory of
  Musical Structure"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from solokit.core.phrase import Phrase
    from solokit.core.transcription import Transcription


@dataclass(frozen=True, slots=True)
class SegmentationConfig:
    """Knobs for phrase segmentation.

    Defaults chosen for bebop / post-bop solo transcriptions (the bulk of
    WJAZD). Tune per-corpus.
    """

    rest_threshold_beats: float = 0.5
    max_phrase_beats: float = 16.0
    max_phrase_notes: int = 24
    min_phrase_notes: int = 3
    skip_first_beats: float = 0.0  # for Omnibook-style transcriptions with head intro


def segment_by_rest(
    transcription: Transcription,
    config: SegmentationConfig | None = None,
) -> tuple[Phrase, ...]:
    """Split a Transcription into phrases at rests longer than the threshold.

    Returns phrases with onset/end beat set. Falls back to a length-based
    split if a phrase exceeds `max_phrase_beats` or `max_phrase_notes`.

    TODO: implement. This is a stub. Reference implementation in
    `solokit/patterns/ngram.py` shows the beat-arithmetic style.
    """
    raise NotImplementedError("segment_by_rest is a stub — see docstring for the spec")


def segment_hybrid(
    transcription: Transcription,
    config: SegmentationConfig | None = None,
) -> tuple[Phrase, ...]:
    """Rest + length hybrid segmentation.

    Use rest detection first; if no rest appears within
    `max_phrase_beats`, force a split there. Drops phrases shorter than
    `min_phrase_notes`.

    TODO: implement.
    """
    raise NotImplementedError("segment_hybrid is a stub — see docstring for the spec")
