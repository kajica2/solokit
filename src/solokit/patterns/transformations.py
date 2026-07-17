"""Pattern transformations.

The four transformations used in DTL / jazzomat pattern search, defined
as pure functions on sequences of integers (MIDI pitches).

Why four transformations?
    - pitch: literal MIDI. Only matches transcriptions in the same key.
    - interval: relative semitone differences. Matches across keys.
    - fuzzy_interval: buckets small intervals. Matches sloppy/human transcriptions.
    - cdpcx: chord-diatonic pitch-class. Matches across the underlying harmony.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, TypeAlias

# ----------------------------------------------------------------------------
# Type aliases
# ----------------------------------------------------------------------------

Transformation: TypeAlias = Literal["pitch", "interval", "fuzzyinterval", "cdpcx"]
TransformFn: TypeAlias = Callable[[Sequence[int]], list[int]]


# ----------------------------------------------------------------------------
# Individual transformations
# ----------------------------------------------------------------------------


def pitch(pitches: Sequence[int]) -> list[int]:
    """Identity — return the pitches as-is.

    Use this when you want exact MIDI match (e.g. "did anyone play this
    specific phrase note-for-note?"). Length-preserving.
    """
    return list(pitches)


def interval(pitches: Sequence[int]) -> list[int]:
    """Semitone-difference between consecutive notes.

    Length: n-1. A scalar (no reference pitch) so transpositions match.
    This is the most useful transformation for jazz: a phrase played in
    Bb and the same phrase in Eb will both match.
    """
    if len(pitches) < 2:
        return []
    return [b - a for a, b in zip(pitches, pitches[1:], strict=False)]


def fuzzy_interval(pitches: Sequence[int]) -> list[int]:
    """Interval with small differences merged.

    Reduces 1-semitone and 2-semitone differences to a single bucket
    (or, equivalently, treats semitone-vs-tone ambiguity). Helps when
    transcriptions are imprecise (human ear vs. MIDI extraction).

    Length: n-1. Uses the same contour but loses the semitone/tone
    distinction.
    """
    if len(pitches) < 2:
        return []
    out: list[int] = []
    for a, b in zip(pitches, pitches[1:], strict=False):
        diff = b - a
        sign = 0 if diff == 0 else (1 if diff > 0 else -1)
        magnitude = abs(diff)
        if magnitude <= 1:
            out.append(sign * 1)  # 1-semitone or unison → ±1 or 0
        elif magnitude == 2:
            out.append(sign * 2)  # whole tone → ±2
        elif magnitude <= 4:
            out.append(sign * 3)  # minor 3rd / major 3rd → ±3 (both are "a 3rd")
        else:
            out.append(sign * magnitude)
    return out


def cdpcx(
    pitches: Sequence[int],
    chord_roots: Sequence[int] | None = None,
    chord_qualities: Sequence[str] | None = None,
) -> list[int]:
    """Chord-diatonic pitch class.

    For each note, return its position within the underlying chord's
    scale (0-6 for 7th chords, 0-3 for triads). The most abstract
    transformation: two phrases with the same harmonic function will
    match even if they use different scale notes.

    TODO: support the no-chord case (assume C major by default), and the
    full quality vocabulary (maj7, m7, 7, m7b5, dim7, alt, etc.).
    """
    if len(pitches) < 1:
        return []
    if chord_roots is None:
        # Without chord info, fall back to major scale degrees from C
        chord_roots = [0] * len(pitches)
        chord_qualities = ["maj7"] * len(pitches)
    if chord_qualities is None:
        chord_qualities = ["maj7"] * len(pitches)

    # Major scale intervals: 0, 2, 4, 5, 7, 9, 11 → degrees 0..6
    major_scale = (0, 2, 4, 5, 7, 9, 11)
    out: list[int] = []
    for pitch_val, root, quality in zip(pitches, chord_roots, chord_qualities, strict=False):
        # distance from root, wrapped into [0, 12)
        chroma = (pitch_val - root) % 12
        # Note: "maj7" starts with "m" but is NOT minor — check explicit forms first.
        is_minor = (
            quality in ("m", "m7", "min", "min7", "minor")
            or (quality.startswith("m") and not quality.startswith("maj"))
        )
        if is_minor:
            scale = (0, 2, 3, 5, 7, 8, 10)
        elif "dim" in quality:
            scale = (0, 2, 3, 5, 6, 8, 9, 11)
        elif "alt" in quality:
            scale = (0, 3, 4, 7, 9, 10)
        else:  # maj7 / 7 / dom
            scale = major_scale
        if chroma in scale:
            out.append(scale.index(chroma))
        else:
            out.append(-1)  # chromatic non-scale tone
    return out


# ----------------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------------

_REGISTRY: dict[Transformation, TransformFn] = {
    "pitch": pitch,
    "interval": interval,
    "fuzzyinterval": fuzzy_interval,
    "cdpcx": cdpcx,
}


def transform(pitches: Sequence[int], kind: Transformation) -> list[int]:
    """Apply the named transformation.

    >>> transform([60, 62, 64], "interval")
    [2, 2]
    >>> transform([60, 62, 64], "pitch")
    [60, 62, 64]
    """
    try:
        fn = _REGISTRY[kind]
    except KeyError as exc:
        msg = f"Unknown transformation {kind!r}; choose from {list(_REGISTRY)}"
        raise ValueError(msg) from exc
    return fn(pitches)
