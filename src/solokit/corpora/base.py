"""Abstract Corpus interface.

Every concrete corpus (DTL, WJAZD, Omnibook, EsAC) implements this.
Local corpora (WJAZD, Omnibook) load from disk / SQLite. Remote
corpora (DTL) hit an HTTP API. The interface hides the difference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from solokit.core.solo import Solo
    from solokit.patterns.ngram import NGram
    from solokit.patterns.similarity import Match


class CorpusError(RuntimeError):
    """Raised when a corpus operation fails (network, parse, etc.)."""


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single match from a corpus search, with corpus-specific metadata.

    Attributes:
        match: The underlying Match (edit distance, similarity, source NGram).
        performer: Performer name.
        title: Solo title.
        year: Recording year.
        melid: Corpus-specific identifier (used to construct audio URLs).
        audio_url: Direct link to the audio snippet, if available.
        instrument: Instrument name (sax, trumpet, etc.).
        database: Which corpus the match came from.
        start_position: Time in the recording where the pattern begins.
        duration: Length of the match in seconds.
    """

    match: Match
    performer: str
    title: str
    year: int | None
    melid: str
    audio_url: str | None
    database: str
    instrument: str | None = None
    start_position: float | None = None
    duration: float | None = None


class Corpus(ABC):
    """Abstract base class for a transcription corpus."""

    name: str = "unnamed"

    @abstractmethod
    def __iter__(self) -> Iterable[Solo]:
        """Iterate over the corpus's solos (may be slow for remote corpora)."""

    @abstractmethod
    def search(
        self,
        pattern: Sequence[int],
        *,
        transformation: str = "interval",
        min_similarity: float = 0.8,
        max_length_difference: int = 0,
        max_edit_distance: int | None = None,
        min_frequency: int = 1,
        limit: int = 50,
    ) -> list[SearchResult]:
        """Search the corpus for matches to a pattern.

        Args:
            pattern: The pattern values, already transformed
                (i.e. for "interval" search, the differences, not the pitches).
            transformation: Echo of which transformation was used (for logging).
            min_similarity: Minimum similarity threshold.
            max_length_difference: Length tolerance.
            max_edit_distance: Optional hard cap on edit distance.
            min_frequency: Minimum number of identical instances.
            limit: Maximum results to return.

        Returns:
            SearchResults sorted by similarity descending.
        """

    def all_ngrams(
        self,
        n: int = 5,
        transformation: str = "interval",
    ) -> Iterable[NGram]:
        """Yield all NGrams in the corpus (for offline indexing / similarity)."""
        from solokit.patterns.ngram import NGramExtractor  # avoid circular import

        extractor = NGramExtractor(n=n, transformation=transformation)  # type: ignore[arg-type]
        for solo in self:
            yield from extractor.extract_from_solo(solo)
