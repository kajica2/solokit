"""Phrase — a contiguous melodic segment within a Solo.

A Phrase is a list of NoteEvents with optional phrase-level metadata
(chord, bar range, type classification like "head" / "solo" / "lick").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from solokit.core.transcription import NoteEvent


@dataclass(frozen=True, slots=True)
class Phrase:
    """A contiguous melodic segment within a solo.

    Attributes:
        notes: Ordered list of NoteEvents that make up the phrase.
        start_beat: Bar-relative beat where the phrase begins (0-indexed).
        end_beat: Bar-relative beat where the phrase ends.
        chord_progression: Chord labels (Roman numerals or absolute) over the phrase.
        phrase_type: Optional classification (e.g. "head", "solo", "lick", "turnaround").
        id: Optional stable identifier (e.g. within-corpus phrase ID).
    """

    notes: tuple[NoteEvent, ...]
    start_beat: float
    end_beat: float
    chord_progression: tuple[str, ...] = field(default_factory=tuple)
    phrase_type: str | None = None
    id: str | None = None

    def __post_init__(self) -> None:
        if not self.notes:
            msg = "Phrase must contain at least one note"
            raise ValueError(msg)
        if self.end_beat < self.start_beat:
            msg = f"end_beat ({self.end_beat}) < start_beat ({self.start_beat})"
            raise ValueError(msg)

    @property
    def duration_beats(self) -> float:
        """Length of the phrase in beats."""
        return self.end_beat - self.start_beat

    @property
    def pitches(self) -> tuple[int, ...]:
        """MIDI pitches of the phrase's notes (rests skipped)."""
        return tuple(n.pitch for n in self.notes if n.pitch is not None)

    @property
    def iois(self) -> tuple[float, ...]:
        """Inter-onset intervals in beats (length = n_notes - 1)."""
        result: list[float] = []
        for prev, curr in zip(self.notes, self.notes[1:], strict=False):
            result.append(curr.onset_beat - prev.onset_beat)
        return tuple(result)

    def with_notes(self, notes: tuple[NoteEvent, ...]) -> Phrase:
        """Return a new Phrase with replaced notes (preserves metadata)."""
        return Phrase(
            notes=notes,
            start_beat=self.start_beat,
            end_beat=self.end_beat,
            chord_progression=self.chord_progression,
            phrase_type=self.phrase_type,
            id=self.id,
        )
