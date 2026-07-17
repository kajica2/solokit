"""Tests for N-gram extraction."""

from __future__ import annotations

import pytest

from solokit.patterns.ngram import NGram, NGramExtractor


class TestNGramFromPitches:
    def test_interval_extraction(self) -> None:
        # C major scale: C D E F G
        # interval: [2, 2, 1, 2]
        # n=3 grams over interval: [[2,2,1], [2,1,2]] — length 2
        grams = NGram.from_pitches([60, 62, 64, 65, 67], n=3, transformation="interval")
        assert len(grams) == 2
        assert grams[0].values == (2, 2, 1)
        assert grams[1].values == (2, 1, 2)

    def test_pitch_extraction(self) -> None:
        grams = NGram.from_pitches([60, 62, 64, 65, 67], n=3, transformation="pitch")
        assert len(grams) == 3
        assert grams[0].values == (60, 62, 64)
        assert grams[1].values == (62, 64, 65)
        assert grams[2].values == (64, 65, 67)  # sliding window, all 3-grams full-length

    def test_too_small_n_raises(self) -> None:
        with pytest.raises(ValueError, match="n must be >= 2"):
            NGram.from_pitches([60, 62, 64], n=1, transformation="interval")

    def test_too_short_input(self) -> None:
        assert NGram.from_pitches([60], n=3, transformation="interval") == ()

    def test_immutable(self) -> None:
        g = NGram(values=(1, 2, 3), source_id="x")
        with pytest.raises(Exception):  # frozen dataclass
            g.values = (4, 5, 6)  # type: ignore[misc]


class TestNGramExtractor:
    def test_basic_usage(self) -> None:
        ext = NGramExtractor.interval(3)
        grams = ext.extract_from_pitches([60, 62, 64, 65, 67])
        assert len(grams) == 2

    def test_cache_works(self) -> None:
        ext = NGramExtractor.interval(3)
        g1 = ext.extract_from_pitches([60, 62, 64, 65, 67])
        g2 = ext.extract_from_pitches([60, 62, 64, 65, 67])  # cached
        assert g1 is g2

    def test_invalid_transformation(self) -> None:
        with pytest.raises(ValueError, match="Unknown transformation"):
            NGramExtractor(n=3, transformation="nonsense")  # type: ignore[arg-type]

    def test_extract_from_solo(self, sample_solo) -> None:
        ext = NGramExtractor.interval(4)
        grams = list(ext.extract_from_solo(sample_solo))
        # sample_solo has 9 notes → 8 intervals → 5 4-interval grams
        assert len(grams) == 5
        # All should be NGram instances
        assert all(isinstance(g, NGram) for g in grams)

    def test_extract_from_phrase(self, sample_phrase) -> None:
        ext = NGramExtractor.interval(4)
        grams = ext.extract_from_phrase(sample_phrase)
        assert len(grams) > 0
        # Onset beats should be carried over
        assert all(g.onset_beat is not None for g in grams)
