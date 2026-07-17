"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from solokit.core.phrase import Phrase
from solokit.core.solo import Solo, SoloMetadata
from solokit.core.transcription import NoteEvent, Transcription


@pytest.fixture
def sample_solo() -> Solo:
    """A short C-major-ish solo phrase for testing."""
    notes = (
        NoteEvent(pitch=60, onset_beat=0.0, duration_beats=1.0),  # C
        NoteEvent(pitch=62, onset_beat=1.0, duration_beats=1.0),  # D
        NoteEvent(pitch=64, onset_beat=2.0, duration_beats=1.0),  # E
        NoteEvent(pitch=65, onset_beat=3.0, duration_beats=1.0),  # F
        NoteEvent(pitch=67, onset_beat=4.0, duration_beats=1.0),  # G
        NoteEvent(pitch=65, onset_beat=5.0, duration_beats=1.0),  # F
        NoteEvent(pitch=64, onset_beat=6.0, duration_beats=1.0),  # E
        NoteEvent(pitch=62, onset_beat=7.0, duration_beats=1.0),  # D
        NoteEvent(pitch=60, onset_beat=8.0, duration_beats=2.0),  # C
    )
    t = Transcription.from_note_sequence(notes, tempo_bpm=120.0)
    return Solo.from_transcription(
        SoloMetadata(melid="test", title="Test Solo", performer="Tester", recording_year=2024),
        t,
    )


@pytest.fixture
def sample_phrase(sample_solo: Solo) -> Phrase:
    """The first 5 notes of sample_solo as a Phrase."""
    note_subset = sample_solo.transcription.notes[:5]
    return Phrase(notes=note_subset, start_beat=0.0, end_beat=5.0, phrase_type="lick")
