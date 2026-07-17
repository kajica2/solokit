"""Core data model: Solo, Phrase, NoteEvent, Transcription.

These are the atomic objects that everything else operates on. They are
intentionally simple dataclasses (no music21 inheritance) so the rest of
the library stays decoupled from any one symbolic-music backend.
"""

from solokit.core.phrase import Phrase
from solokit.core.solo import Solo, SoloMetadata
from solokit.core.transcription import NoteEvent, Transcription

__all__ = [
    "NoteEvent",
    "Phrase",
    "Solo",
    "SoloMetadata",
    "Transcription",
]
