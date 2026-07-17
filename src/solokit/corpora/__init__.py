"""Corpus access — load and search collections of transcribed solos.

The interface is uniform across corpora (DTL, WJAZD, Omnibook, EsAC):
you instantiate the corpus, optionally configure it, and call `.search(...)`.
"""

from solokit.corpora.base import Corpus, CorpusError, SearchResult
from solokit.corpora.dtl1000 import DTLCorpus
from solokit.corpora.omnibook import OmnibookCorpus

__all__ = [
    "Corpus",
    "CorpusError",
    "DTLCorpus",
    "OmnibookCorpus",
    "SearchResult",
]
