"""Pattern extraction, transformation, and similarity search.

The algorithms here are the same family as Dig That Lick (DTL): N-gram
extraction, four transformations (pitch / interval / fuzzy / CDPCX), and
normalized Levenshtein distance for similarity.

Reference:
- Pfleiderer, Frieler, Abeßer (2017) — "Inside the Jazzomat"
"""

from solokit.patterns.ngram import (
    NGram,
    NGramExtractor,
    PhraseNGramExtractor,
)
from solokit.patterns.similarity import Match, search_patterns
from solokit.patterns.transformations import (
    Transformation,
    cdpcx,
    fuzzy_interval,
    interval,
    pitch,
    transform,
)

__all__ = [
    "Match",
    "NGram",
    "NGramExtractor",
    "PhraseNGramExtractor",
    "Transformation",
    "cdpcx",
    "fuzzy_interval",
    "interval",
    "pitch",
    "search_patterns",
    "transform",
]
