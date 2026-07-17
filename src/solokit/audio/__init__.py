"""Audio I/O and analysis.

The killer feature in this module is `transcribe.transcribe_wav` — given
a .wav of a jazz solo, return a `Transcription` you can drop into the
pattern search pipeline. This is the piece the original pymus stack
didn't have (it was score-informed, not auto-transcribing).

The other modules (F0, tuning, loudness) port over the pymus/sisa
functionality for score-informed performance analysis.
"""

from solokit.audio.transcribe import transcribe_wav

__all__ = ["transcribe_wav"]
