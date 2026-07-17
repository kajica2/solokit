"""Tests for the pYIN held-note merging logic."""

from __future__ import annotations

import pytest

from solokit.audio.transcribe import _merge_close_notes
from solokit.core.transcription import NoteEvent


def make_note(pitch: int, onset: float, duration: float) -> NoteEvent:
    return NoteEvent(pitch=pitch, onset_beat=onset, duration_beats=duration, velocity=None)


class TestMergeCloseNotes:
    def test_empty(self) -> None:
        assert _merge_close_notes([]) == []

    def test_single_note(self) -> None:
        notes = [make_note(60, 0.0, 1.0)]
        assert _merge_close_notes(notes) == notes

    def test_merges_exact_same_pitch_close_together(self) -> None:
        notes = [
            make_note(60, 0.0, 0.5),
            make_note(60, 0.6, 0.5),  # 0.1s gap → should merge
        ]
        merged = _merge_close_notes(notes, max_gap_s=0.1, max_pitch_diff=0)
        assert len(merged) == 1
        assert merged[0].pitch == 60
        assert merged[0].onset_beat == 0.0
        assert merged[0].duration_beats == pytest.approx(1.1)

    def test_does_not_merge_far_apart(self) -> None:
        notes = [
            make_note(60, 0.0, 0.5),
            make_note(60, 1.0, 0.5),  # 0.5s gap → don't merge
        ]
        merged = _merge_close_notes(notes, max_gap_s=0.1, max_pitch_diff=0)
        assert len(merged) == 2

    def test_does_not_merge_different_pitch(self) -> None:
        notes = [
            make_note(60, 0.0, 0.5),
            make_note(62, 0.6, 0.5),  # different pitch, close gap → don't merge
        ]
        merged = _merge_close_notes(notes, max_gap_s=0.1, max_pitch_diff=0)
        assert len(merged) == 2

    def test_merges_with_pitch_tolerance(self) -> None:
        # Vibrato: pitch oscillates ±1 semitone, all within 200ms gap
        notes = [
            make_note(60, 0.0, 0.5),
            make_note(61, 0.6, 0.5),  # +1 semitone, close gap
            make_note(60, 1.2, 0.5),  # back to 60
        ]
        merged = _merge_close_notes(notes, max_gap_s=0.2, max_pitch_diff=1)
        assert len(merged) == 1
        assert merged[0].pitch == 60  # first pitch wins
        assert merged[0].duration_beats == pytest.approx(1.7)

    def test_stops_at_pitch_jump(self) -> None:
        # Sequence: 60, 60, 60, 62, 62, 62 — should split at the 60→62 jump
        notes = [
            make_note(60, 0.0, 0.5),
            make_note(60, 0.6, 0.5),
            make_note(60, 1.2, 0.5),
            make_note(62, 1.8, 0.5),
            make_note(62, 2.4, 0.5),
            make_note(62, 3.0, 0.5),
        ]
        merged = _merge_close_notes(notes, max_gap_s=0.1, max_pitch_diff=0)
        assert len(merged) == 2
        assert merged[0].pitch == 60
        assert merged[1].pitch == 62

    def test_three_way_merge(self) -> None:
        # Three held notes that all merge into one
        notes = [
            make_note(60, 0.0, 0.3),
            make_note(60, 0.4, 0.3),
            make_note(60, 0.8, 0.3),
        ]
        merged = _merge_close_notes(notes, max_gap_s=0.1, max_pitch_diff=0)
        assert len(merged) == 1
        assert merged[0].onset_beat == 0.0
        assert merged[0].duration_beats == pytest.approx(1.1)

    def test_gap_exactly_at_threshold(self) -> None:
        # gap = max_gap_s should still merge (<=)
        notes = [
            make_note(60, 0.0, 0.5),
            make_note(60, 0.7, 0.5),  # gap = 0.2
        ]
        merged = _merge_close_notes(notes, max_gap_s=0.2, max_pitch_diff=0)
        assert len(merged) == 1
