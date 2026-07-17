"""NoteEvent and Transcription — the atomic musical objects.

A Transcription is the result of converting a score (MIDI, MusicXML,
audio) into a flat list of timed NoteEvents. It carries no phrase
structure — that lives in Solo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class NoteEvent:
    """A single note (or rest) in a transcription.

    Times are in beats (not seconds) to make pattern comparisons
    tempo-independent. Conversions to/from seconds happen at I/O
    boundaries (see `solokit.audio`).

    Attributes:
        pitch: MIDI pitch (0-127), or None for a rest.
        onset_beat: Time of attack in beats from the start of the solo.
        duration_beats: Length in beats.
        velocity: Optional MIDI velocity 1-127 (None = unknown / unnoted).
    """

    pitch: int | None
    onset_beat: float
    duration_beats: float
    velocity: int | None = None

    def __post_init__(self) -> None:
        if self.pitch is not None and not 0 <= self.pitch <= 127:
            msg = f"pitch {self.pitch} out of MIDI range 0-127"
            raise ValueError(msg)
        if self.duration_beats <= 0:
            msg = f"duration_beats must be > 0, got {self.duration_beats}"
            raise ValueError(msg)

    @property
    def is_rest(self) -> bool:
        return self.pitch is None

    @property
    def offset_beat(self) -> float:
        return self.onset_beat + self.duration_beats


@dataclass(frozen=True, slots=True)
class Transcription:
    """A flat ordered sequence of NoteEvents.

    Acts as the lowest-level representation everything else reads. Most
    of the time you don't construct a Transcription by hand — it comes
    from a loader (`Transcription.from_midi(...)`) or an audio
    transcription step (`audio.transcribe.transcribe_wav`).
    """

    notes: tuple[NoteEvent, ...]
    tempo_bpm: float = 120.0
    time_signature: tuple[int, int] = (4, 4)
    key_signature: str | None = None

    def __post_init__(self) -> None:
        # Ensure time-ordering (defensive — loaders should already guarantee this)
        for prev, curr in zip(self.notes, self.notes[1:], strict=False):
            if curr.onset_beat < prev.onset_beat:
                msg = "Transcription notes must be ordered by onset_beat"
                raise ValueError(msg)

    @property
    def total_beats(self) -> float:
        if not self.notes:
            return 0.0
        return max(n.offset_beat for n in self.notes)

    @property
    def pitches(self) -> tuple[int, ...]:
        return tuple(n.pitch for n in self.notes if n.pitch is not None)

    def slice(self, start_beat: float, end_beat: float) -> Transcription:
        """Return a new Transcription containing only notes within [start, end)."""
        sliced = tuple(
            n for n in self.notes if start_beat <= n.onset_beat < end_beat
        )
        return Transcription(
            notes=sliced,
            tempo_bpm=self.tempo_bpm,
            time_signature=self.time_signature,
            key_signature=self.key_signature,
        )

    @classmethod
    def from_note_sequence(
        cls,
        notes: Sequence[NoteEvent],
        tempo_bpm: float = 120.0,
        time_signature: tuple[int, int] = (4, 4),
        key_signature: str | None = None,
    ) -> Transcription:
        """Construct from any sequence, sorting by onset_beat and stripping rests."""
        sorted_notes = tuple(sorted(notes, key=lambda n: n.onset_beat))
        return cls(
            notes=sorted_notes,
            tempo_bpm=tempo_bpm,
            time_signature=time_signature,
            key_signature=key_signature,
        )
