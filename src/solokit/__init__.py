"""solokit — pattern search and analysis for jazz solo transcriptions.

A modernized reimagining of the Jazzomat research stack, built on music21
and compatible with the Dig That Lick pattern-search API.

Quick start::

    from solokit.patterns import NGramExtractor, search_patterns
    from solokit.corpora import DTLCorpus

    extractor = NGramExtractor(n=5, transformation="interval")
    pattern = extractor.from_pitches([60, 59, 58, 62, 57])

    corpus = DTLCorpus()
    matches = corpus.search(pattern, min_similarity=0.8)
"""

from __future__ import annotations

__version__ = "0.2.0"
__all__ = ["__version__"]
