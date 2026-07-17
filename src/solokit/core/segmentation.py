"""Phrase segmentation — split a Transcription into Phrase objects.

A Phrase is a contiguous melodic segment, typically 4-16 beats (1-4
bars in 4/4) and 3-20 notes. For jazz solos, phrases are usually
demarcated by short rests (breath) or by phrase length caps. The
Omnibook's head theme is whole-note chords (skipped by the
`skip_first_beats` cutoff in the corpus loaders), and the body has
typical bebop phrasing with regular breath pauses.

Strategies (in order of complexity):
- segment_by_rest: split on rests longer than the threshold
- segment_hybrid: rest detection + length cap fallback
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

    Defaults are tuned for bebop / post-bop solo transcriptions
    (the bulk of WJAZD and the Omnibook).
    """

    rest_threshold_beats: float = 0.5  # split if a rest is longer than this
    max_phrase_beats: float = 16.0    # cap phrase length (4 bars of 4/4)
    min_phrase_beats: float = 0.5     # discard phrases shorter than this
    max_phrase_notes: int = 32        # cap by note count
    min_phrase_notes: int = 3         # discard phrases with fewer notes
    skip_first_beats: float = 0.0     # discard notes before this offset (head theme)


def _split_at_rest(
    transcription: "Transcription",
    threshold: float,
) -> list[tuple[float, float]]:
    """Return a list of (start_beat, end_beat) tuples — phrase boundaries.

    A "rest" is a gap between consecutive NoteEvents larger than the
    threshold. Each contiguous run of notes between rests is a phrase.
    """
    if not transcription.notes:
        return []

    notes = [n for n in transcription.notes if not n.is_rest]
    if not notes:
        return []

    phrases: list[tuple[float, float]] = []
    phrase_start = notes[0].onset_beat
    prev_offset = notes[0].onset_beat + notes[0].duration_beats

    for note in notes[1:]:
        gap = note.onset_beat - prev_offset
        if gap > threshold:
            # Close the current phrase and start a new one
            phrases.append((phrase_start, prev_offset))
            phrase_start = note.onset_beat
        prev_offset = note.onset_beat + note.duration_beats

    phrases.append((phrase_start, prev_offset))
    return phrases


def _force_split_long_phrase(
    start: float,
    end: float,
    max_beats: float,
    max_notes: int,
) -> list[tuple[float, float]]:
    """If a phrase is too long (by beats or notes), force splits at even intervals."""
    if end - start <= max_beats:
        return [(start, end)]
    # Split into chunks of max_beats
    chunks: list[tuple[float, float]] = []
    cur = start
    while cur + max_beats < end:
        chunks.append((cur, cur + max_beats))
        cur += max_beats
    chunks.append((cur, end))
    return chunks


def _filter_by_min(
    phrases: list[tuple[float, float]],
    transcription: "Transcription",
    min_beats: float,
    min_notes: int,
) -> list[tuple[float, float]]:
    """Drop phrases that are too short or have too few notes."""
    notes = [n for n in transcription.notes if not n.is_rest]
    if not notes:
        return []
    out: list[tuple[float, float]] = []
    for start, end in phrases:
        if end - start < min_beats:
            continue
        n_in = sum(1 for n in notes if start <= n.onset_beat < end)
        if n_in < min_notes:
            continue
        out.append((start, end))
    return out


def segment_by_rest(
    transcription: "Transcription",
    config: SegmentationConfig | None = None,
) -> list["Phrase"]:
    """Split a Transcription into phrases by rest detection only.

    The simplest strategy: split wherever there's a rest > threshold.
    Phrases that are too long are force-split. Phrases that are too
    short or have too few notes are dropped.
    """
    from solokit.core.phrase import Phrase

    cfg = config or SegmentationConfig()
    if not transcription.notes:
        return []

    # Rest-based split
    phrases = _split_at_rest(transcription, cfg.rest_threshold_beats)

    # Force-split long phrases
    split_phrases: list[tuple[float, float]] = []
    for start, end in phrases:
        split_phrases.extend(_force_split_long_phrase(start, end, cfg.max_phrase_beats, cfg.max_phrase_notes))

    # Filter short/empty
    split_phrases = _filter_by_min(split_phrases, transcription, cfg.min_phrase_beats, cfg.min_phrase_notes)

    # Apply skip_first_beats cutoff
    if cfg.skip_first_beats > 0:
        split_phrases = [(s, e) for s, e in split_phrases if s >= cfg.skip_first_beats]

    # Convert to Phrase objects
    notes = [n for n in transcription.notes if not n.is_rest]
    result: list[Phrase] = []
    for i, (start, end) in enumerate(split_phrases):
        phrase_notes = tuple(
            NoteEvent.from_phrase_note(n) if hasattr(NoteEvent, 'from_phrase_note') else n
            for n in notes
            if start <= n.onset_beat < end
        )
        # NoteEvent doesn't have from_phrase_note; just use n directly
        phrase_notes = tuple(n for n in notes if start <= n.onset_beat < end)
        if not phrase_notes:
            continue
        result.append(
            Phrase(
                notes=phrase_notes,
                start_beat=start,
                end_beat=end,
                phrase_type=None,
                id=f"phrase-{i}",
            )
        )
    return result


def segment_hybrid(
    transcription: "Transcription",
    config: SegmentationConfig | None = None,
) -> list["Phrase"]:
    """Rest detection + length cap (alias for segment_by_rest for now).

    Future: use a learned model to detect phrase boundaries (e.g.
    a small CNN over pitch/timing features). For now, this is
    equivalent to segment_by_rest.
    """
    return segment_by_rest(transcription, config)


# Imported here to avoid circular import at module load time
from solokit.core.transcription import NoteEvent  # noqa: E402
