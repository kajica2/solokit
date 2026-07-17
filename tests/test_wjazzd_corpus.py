"""Tests for the local WJAZD corpus."""

from __future__ import annotations

import time

import pytest

from solokit.corpora import WJAZDCorpus


@pytest.fixture(scope="module")
def corpus() -> WJAZDCorpus:
    """Load the WJAZD corpus once for the whole test module."""
    return WJAZDCorpus()


class TestWJAZDCorpus:
    def test_loads_456_solos(self, corpus: WJAZDCorpus) -> None:
        assert len(corpus) == 456

    def test_solo_metadata_is_rich(self, corpus: WJAZDCorpus) -> None:
        """WJAZD has more metadata than Omnibook (year, instrument, style)."""
        coltrane_blue_train = corpus.get_solo_by_melid(218)  # from earlier query
        assert coltrane_blue_train is not None
        m = coltrane_blue_train.metadata
        assert "Coltrane" in m.performer
        assert "Blue Train" in m.title
        assert m.recording_year is not None
        assert m.instrument
        assert m.style
        assert m.key  # e.g. "Eb-maj"
        assert m.tempo_bpm is not None
        # Extra metadata preserved
        assert m.extra.get("chorus_count") is not None

    def test_search_finds_known_pattern(self, corpus: WJAZDCorpus) -> None:
        # The classic descending-then-leap bebop pattern
        results = corpus.search(
            [-1, -1, 4, -5, -2],
            transformation="interval",
            min_similarity=0.7,
            max_length_difference=1,
            limit=20,
        )
        assert len(results) > 0
        # All results should be real jazz performers
        for r in results:
            assert r.performer
            assert r.title

    def test_search_returns_year_and_instrument(self, corpus: WJAZDCorpus) -> None:
        results = corpus.search(
            [-1, -1, 4, -5, -2],
            transformation="interval",
            min_similarity=0.7,
            max_length_difference=1,
            limit=20,
        )
        # Most results have year + instrument; some WJAZD solos lack year
        # (no record_info link). Require that AT LEAST 80% have year.
        with_year = [r for r in results if r.year is not None]
        assert len(with_year) >= 0.8 * len(results), (
            f"Only {len(with_year)}/{len(results)} have year"
        )
        # All results should have instrument
        for r in results:
            assert r.instrument is not None, f"No instrument for {r.title}"

    def test_search_is_fast(self, corpus: WJAZDCorpus) -> None:
        """The full 200k+ event search should complete in under 5s."""
        t0 = time.time()
        results = corpus.search(
            [-1, -1, 4, -5, -2],
            transformation="interval",
            min_similarity=0.7,
            max_length_difference=1,
            limit=100,
        )
        elapsed = time.time() - t0
        assert elapsed < 5.0, f"Search took {elapsed:.1f}s (should be < 5s)"
        assert len(results) > 0

    def test_search_cmajor_scale_fragment(self, corpus: WJAZDCorpus) -> None:
        # C major scale fragment (interval 2,2,1,2) — very common pattern
        results = corpus.search(
            [2, 2, 1, 2],
            transformation="interval",
            min_similarity=0.95,  # near-exact match
            limit=50,
        )
        # WJAZD is much larger than Omnibook, this should yield many matches
        assert len(results) >= 3

    def test_search_per_solo_breakdown(self, corpus: WJAZDCorpus) -> None:
        from collections import Counter

        results = corpus.search(
            [-1, -1, 4, -5, -2],
            transformation="interval",
            min_similarity=0.7,
            max_length_difference=1,
            limit=500,
        )
        per_solo = Counter(r.title for r in results)
        # At least some song should have multiple instances
        assert max(per_solo.values(), default=0) >= 2

    def test_get_solo_by_melid_returns_none_for_invalid(self, corpus: WJAZDCorpus) -> None:
        assert corpus.get_solo_by_melid(99999) is None

    def test_includes_jazz_canon(self, corpus: WJAZDCorpus) -> None:
        """Sanity check: the corpus has the jazz canon we expect."""
        performers = {s.metadata.performer for s in corpus}
        for expected in ("John Coltrane", "Miles Davis", "Charlie Parker"):
            assert any(expected in p for p in performers), (
                f"{expected} not found in {len(performers)} performers"
            )
