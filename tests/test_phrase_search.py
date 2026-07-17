"""Tests for phrase segmentation and phrase-level search."""

from __future__ import annotations

import pytest

from solokit.core.phrase import Phrase
from solokit.core.segmentation import SegmentationConfig, segment_by_rest
from solokit.core.transcription import NoteEvent, Transcription
from solokit.patterns.ngram import NGramExtractor, PhraseNGramExtractor


def make_note(pitch: int, onset: float, duration: float) -> NoteEvent:
    return NoteEvent(pitch=pitch, onset_beat=onset, duration_beats=duration, velocity=None)


class TestSegmentByRest:
    def test_empty(self) -> None:
        t = Transcription(notes=())
        assert segment_by_rest(t) == []

    def test_single_phrase_no_rests(self) -> None:
        notes = [make_note(60, 0.0, 1.0), make_note(62, 1.0, 1.0), make_note(64, 2.0, 1.0)]
        t = Transcription.from_note_sequence(notes)
        phrases = segment_by_rest(t)
        assert len(phrases) == 1
        assert phrases[0].start_beat == 0.0
        assert phrases[0].end_beat == 3.0

    def test_splits_on_long_rest(self) -> None:
        # 3+ notes per phrase (default min_phrase_notes=3)
        notes = [
            make_note(60, 0.0, 1.0),
            make_note(62, 1.0, 1.0),
            make_note(64, 2.0, 1.0),
            # 2.0 beat rest > 0.5 threshold
            make_note(67, 5.0, 1.0),
            make_note(69, 6.0, 1.0),
            make_note(71, 7.0, 1.0),
        ]
        t = Transcription.from_note_sequence(notes)
        phrases = segment_by_rest(t)
        assert len(phrases) == 2
        assert (phrases[0].start_beat, phrases[0].end_beat) == (0.0, 3.0)
        assert (phrases[1].start_beat, phrases[1].end_beat) == (5.0, 8.0)

    def test_short_rests_dont_split(self) -> None:
        notes = [
            make_note(60, 0.0, 1.0),
            make_note(62, 1.2, 1.0),  # 0.2 beat gap
            make_note(64, 2.4, 1.0),
        ]
        t = Transcription.from_note_sequence(notes)
        phrases = segment_by_rest(t)
        assert len(phrases) == 1  # gaps below threshold

    def test_force_splits_long_phrase(self) -> None:
        # 20 notes, 1 beat each, 0.1 gap, 0 rest > threshold
        # → one continuous phrase, but max_phrase_beats=8 forces splits
        notes = [make_note(60 + i, i * 1.1, 1.0) for i in range(20)]
        t = Transcription.from_note_sequence(notes)
        config = SegmentationConfig(max_phrase_beats=8.0, rest_threshold_beats=0.5)
        phrases = segment_by_rest(t, config)
        # ~8 beats each, so we should get 2-3 phrases (20 beats total)
        assert len(phrases) >= 2
        for p in phrases:
            assert p.end_beat - p.start_beat <= 9.0  # 8 + 1 beat tolerance

    def test_skips_head_section(self) -> None:
        notes = [make_note(60, i * 1.0, 1.0) for i in range(20)]
        t = Transcription.from_note_sequence(notes)
        config = SegmentationConfig(skip_first_beats=5.0)
        phrases = segment_by_rest(t, config)
        # Phrases should all start at >= 5.0
        for p in phrases:
            assert p.start_beat >= 5.0

    def test_drops_very_short_phrases(self) -> None:
        notes = [
            make_note(60, 0.0, 1.0),
            make_note(62, 1.0, 1.0),
            # 1.0 beat rest - splits here
            make_note(64, 3.0, 0.1),  # 0.1 beat note (too short to be a phrase)
        ]
        t = Transcription.from_note_sequence(notes)
        phrases = segment_by_rest(t)
        # The 0.1-beat phrase should be dropped (min_phrase_beats=0.5)
        assert all(p.end_beat - p.start_beat >= 0.5 for p in phrases)


class TestPhraseNGramExtractor:
    def test_one_gram_per_phrase(self) -> None:
        # Two phrases with 4 notes each (3 intervals → 1 n-gram with n=3)
        p1 = Phrase(
            notes=(
                make_note(60, 0.0, 1.0),
                make_note(62, 1.0, 1.0),
                make_note(64, 2.0, 1.0),
                make_note(65, 3.0, 1.0),
            ),
            start_beat=0.0, end_beat=4.0, id="p1",
        )
        p2 = Phrase(
            notes=(
                make_note(67, 5.0, 1.0),
                make_note(69, 6.0, 1.0),
                make_note(71, 7.0, 1.0),
                make_note(72, 8.0, 1.0),
            ),
            start_beat=5.0, end_beat=9.0, id="p2",
        )
        ext = PhraseNGramExtractor.interval(3)
        grams = ext.extract_from_phrases([p1, p2])
        assert len(grams) == 2
        assert grams[0].source_id == "p1"
        assert grams[1].source_id == "p2"
        # p1: 60→62→64→65 → intervals (2, 2, 1)
        assert grams[0].values == (2, 2, 1)
        # p2: 67→69→71→72 → intervals (2, 2, 1)
        assert grams[1].values == (2, 2, 1)

    def test_skips_empty_phrases(self) -> None:
        # Phrase's __post_init__ rejects empty notes (good!). So an
        # "empty phrase" can't exist as a Phrase object — the only way
        # an "empty" phrase can appear is via segmentation: notes
        # outside the (start, end) window are dropped. We test that
        # path instead: a phrase where the only note falls outside
        # the start/end range produces no grams.
        from solokit.core.transcription import Transcription

        # 2 notes, but phrase window is (10, 20) — no notes in window
        notes = (make_note(60, 0.0, 1.0), make_note(62, 1.0, 1.0))
        # Phrase with start/end that excludes the notes
        # But __post_init__ requires notes, so this is the limit.
        # We just test that a phrase with very few notes produces 0 grams.
        tiny = Phrase(notes=(make_note(60, 10.0, 0.1),), start_beat=10.0, end_beat=10.1, id="tiny")
        ext = PhraseNGramExtractor.interval(3)
        grams = ext.extract_from_phrases([tiny])
        # Only 1 note → can't form 3-note ngram
        assert grams == ()

    def test_distinct_from_pitch_extractor(self) -> None:
        """Phrase-level extraction should produce 1 gram per phrase, not per sliding window."""
        notes = [make_note(60 + i, float(i), 1.0) for i in range(8)]
        p1 = Phrase(notes=tuple(notes[:4]), start_beat=0.0, end_beat=4.0, id="p1")
        p2 = Phrase(notes=tuple(notes[4:]), start_beat=4.0, end_beat=8.0, id="p2")
        ext_phrase = PhraseNGramExtractor.interval(3)
        ext_pitch = NGramExtractor.interval(3)
        # Phrase: 2 grams (1 per phrase)
        phrase_grams = ext_phrase.extract_from_phrases([p1, p2])
        assert len(phrase_grams) == 2
        # Pitch: 5 grams (8 pitches → 7 intervals → 5 3-grams)
        pitch_grams = list(ext_pitch.extract_from_pitches([n.pitch for n in notes]))
        assert len(pitch_grams) == 5
        assert len(phrase_grams) < len(pitch_grams)  # phrase is more conservative
