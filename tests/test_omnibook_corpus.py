"""Tests for the local Omnibook corpus."""

from __future__ import annotations

from collections import Counter

import pytest

from solokit.corpora import OmnibookCorpus


@pytest.fixture(scope="module")
def corpus() -> OmnibookCorpus:
    """Load the Omnibook once for the whole test module."""
    return OmnibookCorpus()


@pytest.mark.network  # requires local MusicXML data
class TestOmnibookCorpus:
    def test_loads_50_solos(self, corpus: OmnibookCorpus) -> None:
        assert len(corpus) == 50
        assert corpus.total_notes > 10_000

    def test_solo_metadata(self, corpus: OmnibookCorpus) -> None:
        s = list(corpus)[0]
        assert s.metadata.performer == "Charlie Parker"
        assert s.metadata.instrument == "alto saxophone"
        assert s.metadata.source_corpus == "omnibook"
        assert s.metadata.title  # non-empty

    def test_transcription_has_pitches(self, corpus: OmnibookCorpus) -> None:
        s = list(corpus)[0]
        assert len(s.transcription.pitches) > 50
        # Range should be in a reasonable saxophone range
        assert 40 <= min(s.transcription.pitches) <= 80
        assert 60 <= max(s.transcription.pitches) <= 100

    def test_search_finds_known_pattern(self, corpus: OmnibookCorpus) -> None:
        # The classic descending-then-leap bebop pattern, known to be
        # in many Parker solos (validated against DTL 2026-07-16).
        results = corpus.search(
            [-1, -1, 4, -5, -2],
            transformation="interval",
            min_similarity=0.7,
            max_length_difference=1,
        )
        assert len(results) > 0, "Expected at least one match for classic bebop pattern"
        # All matches should be in Charlie Parker solos
        assert all("Parker" in r.performer for r in results)
        # Matches should be across multiple solos
        unique_titles = {r.title for r in results}
        assert len(unique_titles) >= 3, (
            f"Expected matches in 3+ different solos, got {len(unique_titles)}: {unique_titles}"
        )

    def test_search_returns_onset_position(self, corpus: OmnibookCorpus) -> None:
        results = corpus.search(
            [-1, -1, 4, -5, -2],
            transformation="interval",
            min_similarity=0.7,
            max_length_difference=1,
            limit=5,
        )
        # At least some results should have a numeric onset position
        with_position = [r for r in results if r.start_position is not None]
        assert len(with_position) > 0, "Expected at least one result with start_position"

    def test_search_top_songs_by_pattern_count(self, corpus: OmnibookCorpus) -> None:
        results = corpus.search(
            [-1, -1, 4, -5, -2],
            transformation="interval",
            min_similarity=0.7,
            max_length_difference=1,
            limit=200,
        )
        counts = Counter(r.title for r in results)
        # Some song should have 2+ instances of this pattern
        assert max(counts.values()) >= 2

    def test_strict_search_no_false_positives(self, corpus: OmnibookCorpus) -> None:
        # With exact length and no edits, only true exact matches should appear
        results = corpus.search(
            [1, 2, 3],  # simple 3-interval pattern
            transformation="interval",
            min_similarity=1.0,  # exact match only
            max_length_difference=0,
        )
        for r in results:
            assert r.match.similarity == 1.0
            assert r.match.edit_distance == 0

    def test_per_solo_breakdown(self, corpus: OmnibookCorpus) -> None:
        """Search results can be aggregated by song."""
        results = corpus.search(
            [-1, -1, 4, -5, -2],
            transformation="interval",
            min_similarity=0.7,
            max_length_difference=1,
            limit=500,
        )
        per_solo = Counter(r.title for r in results)
        # We know from earlier validation that Au Private 2 has multiple instances
        # (it was the most common in the top results)
        assert per_solo.get("Au Private 2", 0) >= 2
