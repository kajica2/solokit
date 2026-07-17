"""Solo — the top-level object: a transcribed performance with metadata.

A Solo bundles a Transcription (or its underlying NoteEvents) with
phrase-level structure, chord/section annotations, and the metadata
needed to look it up in a corpus (performer, year, key, etc.).

The shape mirrors the jazzomat `Solo` class but uses Python dataclasses
and is decoupled from music21.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from solokit.core.phrase import Phrase
    from solokit.core.transcription import Transcription


@dataclass(frozen=True, slots=True)
class SoloMetadata:
    """Identifying and contextual information about a solo.

    Compatible with both the DTL API's CSV columns and the WJAZD
    `reference_record` schema. The fields you don't have, leave None.
    """

    melid: str
    title: str
    performer: str
    recording_year: int | None = None
    track_year: int | None = None
    instrument: str | None = None
    key: str | None = None
    tempo_bpm: int | None = None
    style: str | None = None
    source_corpus: str | None = None
    audio_url: str | None = None
    extra: dict[str, str | int | float | None] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Solo:
    """A transcribed jazz solo with structure and metadata.

    Use `Solo.from_transcription(...)` to derive an un-phrased Solo from
    a raw Transcription. Use `Solo.with_phrases(...)` to attach phrase
    boundaries (typically from `solokit.core.segmentation`).
    """

    metadata: SoloMetadata
    transcription: Transcription
    phrases: tuple[Phrase, ...] = field(default_factory=tuple)
    sections: tuple[str, ...] = field(default_factory=tuple)  # e.g. ("A", "A", "B", "A")

    @classmethod
    def from_transcription(
        cls,
        metadata: SoloMetadata,
        transcription: Transcription,
    ) -> Solo:
        """Build a Solo with no phrase structure from a raw transcription."""
        return cls(metadata=metadata, transcription=transcription)

    @property
    def melid(self) -> str:
        return self.metadata.melid

    @property
    def total_beats(self) -> float:
        return self.transcription.total_beats

    def with_phrases(self, phrases: tuple[Phrase, ...]) -> Solo:
        """Return a new Solo with attached phrase structure."""
        return Solo(
            metadata=self.metadata,
            transcription=self.transcription,
            phrases=phrases,
            sections=self.sections,
        )

    def pitched_notes(self):
        """Iterate over non-rest NoteEvents in onset order."""
        return (n for n in self.transcription.notes if not n.is_rest)
