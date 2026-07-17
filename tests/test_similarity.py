"""Tests for similarity computation and search."""

from __future__ import annotations

import pytest

from solokit.patterns.ngram import NGram, NGramExtractor
from solokit.patterns.similarity import Match, pattern_similarity, search_patterns


class TestPatternSimilarity:
    def test_identical(self) -> None:
        dist, sim = pattern_similarity([1, 2, 3], [1, 2, 3])
        assert dist == 0
        assert sim == 1.0

    def test_completely_different(self) -> None:
        dist, sim = pattern_similarity([1, 1, 1], [9, 9, 9])
        assert dist == 3
        assert sim == 0.0

    def test_one_substitution(self) -> None:
        dist, sim = pattern_similarity([1, 2, 3], [1, 2, 9])
        assert dist == 1
        assert sim == pytest.approx(2 / 3)

    def test_different_lengths(self) -> None:
        dist, sim = pattern_similarity([1, 2], [1, 2, 3])
        # 1 insertion = 1 edit
        assert dist == 1
        # max_len = 3, so sim = 1 - 1/3 = 0.667
        assert sim == pytest.approx(2 / 3)

    def test_both_empty(self) -> None:
        dist, sim = pattern_similarity([], [])
        assert dist == 0
        assert sim == 1.0


class TestSearchPatterns:
    def _corpus(self) -> list[NGram]:
        ext = NGramExtractor.interval(3)
        phrases = [
            [60, 62, 64, 65, 67],   # [2,2,1,2] → 2 grams
            [62, 64, 65, 67, 69],   # [2,1,2,2] → 2 grams (B major scale fragment)
            [60, 62, 64, 65, 64],   # [2,2,1,-1] → 2 grams
        ]
        out = []
        for p in phrases:
            out.extend(ext.extract_from_pitches(p))
        return out

    def test_finds_exact_match(self) -> None:
        corpus = self._corpus()
        # First corpus entry's grams include (2, 2, 1)
        matches = search_patterns([2, 2, 1], corpus, min_similarity=0.8)
        assert len(matches) >= 1
        assert matches[0].similarity == 1.0
        assert matches[0].edit_distance == 0

    def test_finds_close_match(self) -> None:
        corpus = self._corpus()
        # (2, 2, 2) is close to (2, 2, 1) — 1 edit
        matches = search_patterns([2, 2, 2], corpus, min_similarity=0.6)
        # Should find the (2, 2, 1) gram from the first entry
        sims = [m.similarity for m in matches]
        assert any(s > 0.6 for s in sims)

    def test_threshold_filters(self) -> None:
        corpus = self._corpus()
        # At very high threshold, only exact matches pass
        matches = search_patterns([2, 2, 1], corpus, min_similarity=0.99)
        assert all(m.similarity >= 0.99 for m in matches)

    def test_length_filter(self) -> None:
        corpus = self._corpus()
        # All corpus grams are length 3; query length 5 with max_length_difference=0
        # → no matches
        matches = search_patterns([1, 2, 3, 4, 5], corpus, max_length_difference=0)
        assert matches == []

    def test_length_filter_with_tolerance(self) -> None:
        corpus = self._corpus()
        # Same query, but allow length difference
        matches = search_patterns([1, 2, 3, 4, 5], corpus, max_length_difference=2)
        # Won't be high similarity, but should at least compute
        assert isinstance(matches, list)

    def test_min_frequency(self) -> None:
        # Build a corpus with one pattern repeated and a different one
        gram = NGram(values=(1, 2, 3), source_id="x")
        corpus = [gram, gram, NGram(values=(9, 9, 9), source_id="y")]
        # min_frequency=2 should drop (9,9,9) entirely (only appears once)
        # but keep both instances of (1,2,3) — each is a separate match
        matches = search_patterns([1, 2, 3], corpus, min_similarity=0.8, min_frequency=2)
        assert len(matches) == 2
        assert all(m.source.values == (1, 2, 3) for m in matches)

    def test_min_frequency_filters_unique_grams(self) -> None:
        # (1, 2, 3) appears once, (5, 5, 5) appears 3 times — min_frequency=2
        # should only allow matches against (5, 5, 5), not (1, 2, 3)
        corpus = [
            NGram(values=(1, 2, 3), source_id="x"),
            NGram(values=(5, 5, 5), source_id="y"),
            NGram(values=(5, 5, 5), source_id="y"),
            NGram(values=(5, 5, 5), source_id="y"),
        ]
        matches = search_patterns([5, 5, 5], corpus, min_similarity=0.8, min_frequency=2)
        assert len(matches) == 3
        assert all(m.source.values == (5, 5, 5) for m in matches)

    def test_invalid_min_similarity(self) -> None:
        with pytest.raises(ValueError, match="min_similarity"):
            search_patterns([1, 2, 3], [], min_similarity=1.5)

    def test_results_sorted(self) -> None:
        corpus = self._corpus()
        matches = search_patterns([2, 2, 1], corpus, min_similarity=0.5)
        sims = [m.similarity for m in matches]
        assert sims == sorted(sims, reverse=True)
