"""N-gram extraction over transcribed melodies.

Given a sequence of pitches and a transformation, produce a stream of
N-grams. This is the same algorithm DTL uses, decoupled from the
server.

The output length is `n - 1` for `interval` / `fuzzy_interval` (since
the transformation shrinks by 1) and `n` for `pitch` (identity). For
`cdpcx` it's `n` if you pass pitches; intermediate handling is up to
the caller.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from solokit.patterns.transformations import (
    Transformation,
    cdpcx,
    fuzzy_interval,
    interval,
    pitch,
    transform,
)

if TYPE_CHECKING:
    from solokit.core.phrase import Phrase
    from solokit.core.solo import Solo


@dataclass(frozen=True, slots=True)
class NGram:
    """A single N-gram pattern instance.

    Attributes:
        values: The transformed values (tuple of ints).
        source_id: Optional identifier of the source solo/phrase.
        onset_beat: Onset in beats within the source (None if unknown).
    """

    values: tuple[int, ...]
    source_id: str | None = None
    onset_beat: float | None = None

    def __len__(self) -> int:
        return len(self.values)

    def __hash__(self) -> int:
        return hash((self.values, self.source_id))

    @classmethod
    def from_pitches(
        cls,
        pitches: Sequence[int],
        n: int,
        transformation: Transformation = "interval",
    ) -> tuple[NGram, ...]:
        """Extract N-grams from a raw pitch sequence.

        For "interval"/"fuzzyinterval", the result has length len(pitches) - n
        and each N-gram is of length n-1 (because the transform shrinks by 1).
        For "pitch"/"cdpcx", the result has length len(pitches) - n + 1.
        """
        if n < 2:
            msg = f"n must be >= 2, got {n}"
            raise ValueError(msg)
        transformed = transform(pitches, transformation)
        grams: list[NGram] = []
        for i in range(len(transformed) - n + 1):
            grams.append(cls(values=tuple(transformed[i : i + n])))
        return tuple(grams)

    @classmethod
    def from_phrase(
        cls,
        phrase: Phrase,
        n: int,
        transformation: Transformation = "interval",
    ) -> tuple[NGram, ...]:
        """Extract N-grams from a Phrase, carrying onset_beat from the source."""
        pitches = phrase.pitches
        grams = cls.from_pitches(pitches, n, transformation)
        if not grams or phrase.notes[0].onset_beat is None:
            return grams
        # Approximate onset_beat for each gram as the onset of its first note
        n_notes_per_gram = n if transformation == "pitch" else n + 1
        result = []
        for idx, gram in enumerate(grams):
            note_idx = min(idx, len(phrase.notes) - 1)
            onset = phrase.notes[note_idx].onset_beat
            result.append(NGram(values=gram.values, source_id=phrase.id, onset_beat=onset))
        return tuple(result)


class NGramExtractor:
    """Configurable N-gram extractor with caching.

    Use directly::

        extractor = NGramExtractor(n=5, transformation="interval")
        grams = extractor.extract_from_pitches([60, 59, 58, 62, 57, 60])

    Or in batch over a Solo / list of phrases.
    """

    __slots__ = ("n", "transformation", "_cache")

    def __init__(self, n: int, transformation: Transformation = "interval") -> None:
        if n < 2:
            msg = f"n must be >= 2, got {n}"
            raise ValueError(msg)
        if transformation not in ("pitch", "interval", "fuzzyinterval", "cdpcx"):
            msg = f"Unknown transformation {transformation!r}"
            raise ValueError(msg)
        self.n = n
        self.transformation = transformation
        self._cache: dict[tuple[int, ...], tuple[NGram, ...]] = {}

    def extract_from_pitches(self, pitches: Sequence[int]) -> tuple[NGram, ...]:
        """Extract N-grams from a raw pitch sequence.

        Results are cached on the input pitch tuple, so re-extracting
        the same sequence (common when comparing many patterns) is free.
        """
        key = tuple(pitches)
        if key in self._cache:
            return self._cache[key]
        result = NGram.from_pitches(key, self.n, self.transformation)
        self._cache[key] = result
        return result

    def extract_from_phrase(self, phrase: Phrase) -> tuple[NGram, ...]:
        """Extract N-grams from a Phrase (carries onset_beat)."""
        return NGram.from_phrase(phrase, self.n, self.transformation)

    def extract_from_solo(self, solo: Solo) -> Iterable[NGram]:
        """Yield N-grams from a Solo's transcription, one per phrase if available.

        If the Solo has phrases, extract from each. Otherwise segment-less
        extraction over the full transcription.
        """
        if solo.phrases:
            for phrase in solo.phrases:
                yield from self.extract_from_phrase(phrase)
        else:
            pitches = solo.transcription.pitches
            yield from self.extract_from_pitches(pitches)

    # Convenience constructors ------------------------------------------------

    @classmethod
    def interval(cls, n: int) -> NGramExtractor:
        """Shorthand for NGramExtractor(n, "interval")."""
        return cls(n=n, transformation="interval")

    @classmethod
    def pitch_based(cls, n: int) -> NGramExtractor:
        """Shorthand for NGramExtractor(n, "pitch")."""
        return cls(n=n, transformation="pitch")

    @classmethod
    def fuzzy(cls, n: int) -> NGramExtractor:
        """Shorthand for NGramExtractor(n, "fuzzyinterval")."""
        return cls(n=n, transformation="fuzzyinterval")


__all__ = ["NGram", "NGramExtractor"]
