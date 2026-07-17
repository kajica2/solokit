"""Similarity computation and corpus search.

Uses normalized Levenshtein distance (1 - edit_distance / max_length)
which is the same metric DTL uses. The edit distance is the count of
insertions, deletions, and substitutions needed to turn one pattern
into another.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rapidfuzz.distance import Levenshtein

if TYPE_CHECKING:
    from solokit.core.solo import Solo
    from solokit.patterns.ngram import NGram


@dataclass(frozen=True, slots=True)
class Match:
    """A single search match between a query pattern and a corpus pattern.

    Attributes:
        source: The corpus NGram that matched.
        edit_distance: Number of edits (insert/delete/substitute) between query and source.
        similarity: Normalized similarity in [0.0, 1.0]. 1.0 = identical.
    """

    source: NGram
    edit_distance: int
    similarity: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.similarity <= 1.0:
            msg = f"similarity {self.similarity} not in [0, 1]"
            raise ValueError(msg)


def pattern_similarity(query: Sequence[int], target: Sequence[int]) -> tuple[int, float]:
    """Compute (edit_distance, similarity) for two patterns.

    Similarity is normalized: 1 - (edit_distance / max_len(query, target)).
    Identical patterns → (0, 1.0). Completely different patterns of the
    same length → (n, 0.0).
    """
    if not query and not target:
        return 0, 1.0
    q = list(query)
    t = list(target)
    dist = Levenshtein.distance(q, t)
    max_len = max(len(q), len(t))
    sim = 1.0 - (dist / max_len) if max_len > 0 else 1.0
    return dist, sim


def search_patterns(
    query: Sequence[int],
    corpus: Iterable[NGram],
    *,
    min_similarity: float = 0.8,
    max_length_difference: int = 0,
    max_edit_distance: int | None = None,
    min_frequency: int = 1,
    limit: int | None = None,
) -> list[Match]:
    """Search a corpus of NGrams for matches to a query pattern.

    Args:
        query: The pattern to search for (typically already transformed).
        corpus: Iterable of NGrams to search over.
        min_similarity: Minimum similarity threshold (0.5-1.0).
        max_length_difference: Allow patterns up to this many values longer/shorter.
        max_edit_distance: Optional hard cap on edit distance (overrides similarity for cutoff).
        min_frequency: Minimum number of identical instances (for grouped corpora).
        limit: Optional cap on the number of returned matches.

    Returns:
        Matches sorted by similarity descending. Each Match references
        the source NGram so you can look up the source solo/phrase.

    Notes:
        - The corpus may yield duplicate NGram tuples (e.g. one per
          instance). Set min_frequency > 1 to filter rare patterns.
        - For DTL-compatible search over the DTL1000 corpus, use
          `solokit.corpora.DTLCorpus.search(...)` which calls this
          under the hood.
    """
    if not 0.5 <= min_similarity <= 1.0:
        msg = f"min_similarity {min_similarity} not in [0.5, 1.0]"
        raise ValueError(msg)
    if max_length_difference < 0:
        msg = "max_length_difference must be >= 0"
        raise ValueError(msg)

    # First pass: frequency count (if min_frequency > 1)
    if min_frequency > 1:
        freq: dict[tuple[int, ...], int] = {}
        for gram in corpus:
            freq[gram.values] = freq.get(gram.values, 0) + 1
        corpus = (g for g in corpus if freq[g.values] >= min_frequency)

    # Length filter — fast reject
    qlen = len(query)
    min_len = qlen - max_length_difference
    max_len = qlen + max_length_difference

    # Compute matches
    matches: list[Match] = []
    query_list = list(query)
    for gram in corpus:
        glen = len(gram.values)
        if glen < min_len or glen > max_len:
            continue
        dist, sim = pattern_similarity(query_list, gram.values)
        if max_edit_distance is not None and dist > max_edit_distance:
            continue
        if sim < min_similarity:
            continue
        matches.append(Match(source=gram, edit_distance=dist, similarity=sim))

    # Sort by similarity desc, then by source melid for determinism
    matches.sort(key=lambda m: (-m.similarity, m.source.source_id or ""))

    if limit is not None:
        matches = matches[:limit]
    return matches


def search_solo(
    query: Sequence[int],
    solo: Solo,
    n: int = 5,
    transformation: str = "interval",
    **kwargs,
) -> list[Match]:
    """Search a single Solo for matches to a query pattern.

    Convenience wrapper that builds an NGramExtractor, extracts grams
    from the solo, and searches.
    """
    from solokit.patterns.ngram import NGramExtractor  # avoid circular import

    extractor = NGramExtractor(n=n, transformation=transformation)  # type: ignore[arg-type]
    grams = list(extractor.extract_from_solo(solo))
    return search_patterns(query, grams, **kwargs)


__all__ = ["Match", "pattern_similarity", "search_patterns", "search_solo"]
