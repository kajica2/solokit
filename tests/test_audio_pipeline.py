"""End-to-end audio → transcription → pattern search smoke test.

The killer feature: synthesize an audio file, transcribe it, search for
a known pattern in the result. If this works, the product thesis is real.

Findings from the smoke test:
- basic-pitch (Spotify's polyphonic CNN) is the wrong tool for monophonic
  jazz solos. It detects harmonics as separate notes, and even with
  monophonic post-processing, output is noisy.
- librosa.pYIN is the right tool. It nails all 8 synth notes in a C major
  scale, and the resulting n-grams match the known pattern.

This test uses pYIN as the default monophonic backend. basic-pitch is
also tested but with a relaxed assertion (any notes returned).

Run with:    pytest -m audio tests/test_audio_pipeline.py
Skip with:   pytest tests/        (default)
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

# Try the heavy import up-front; skip the module if audio deps are missing.
try:
    from solokit.audio import transcribe_wav

    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

# All tests in this module require audio deps + are slow.
pytestmark = [pytest.mark.audio, pytest.mark.slow]

# Skip the whole module cleanly if imports failed
if not HAS_AUDIO:
    pytest.skip("solokit[audio] not installed", allow_module_level=True)


# ----------------------------------------------------------------------------
# Synthesis
# ----------------------------------------------------------------------------

SAMPLE_RATE = 22050
NOTE_DURATION_S = 1.0
GAP_S = 0.3  # 300ms gap so the held-note merge (200ms tolerance) doesn't collapse distinct notes

# C major scale ascending, 8 notes (one octave)
TEST_PATTERN = [60, 62, 64, 65, 67, 69, 71, 72]


def synthesize_melody(
    pitches: list[int],
    sample_rate: int = SAMPLE_RATE,
    note_duration: float = NOTE_DURATION_S,
    gap: float = GAP_S,
) -> np.ndarray:
    """Synthesize a monophonic melody with piano-like timbre.

    Uses additive synthesis with 6 harmonics (1/n^2 amplitude decay)
    and a piano-style envelope (fast attack, exponential decay) so
    the signal is realistic enough for pYIN to extract clean pitch
    contours.

    The original 0.5s/0.05s-gap synth was too short — pYIN could only
    detect 1 of 6 notes. 1.0s notes with 0.2s gaps work reliably.
    """
    samples_per_note = int(sample_rate * note_duration)
    samples_per_gap = int(sample_rate * gap)
    step = samples_per_note + samples_per_gap
    out = np.zeros(len(pitches) * step, dtype=np.float32)

    t = np.arange(samples_per_note, dtype=np.float32) / sample_rate
    attack = int(0.045 * sample_rate)  # 45ms attack

    for i, midi in enumerate(pitches):
        freq = 440.0 * 2.0 ** ((midi - 69) / 12.0)
        # Harmonics 1-6 with 1/n^2 amplitude (piano-like spectrum)
        harmonics = sum(
            amp * np.sin(2.0 * np.pi * n * freq * t)
            for n, amp in enumerate([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125], start=1)
        ).astype(np.float32)
        # Piano envelope: quick attack, exponential decay
        env = np.exp(-2.0 * t / note_duration, dtype=np.float32)
        env[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
        # Normalize and apply envelope
        note_wave = harmonics / np.max(np.abs(harmonics)) * 0.5 * env
        start = i * step
        out[start : start + samples_per_note] = note_wave

    return out


def save_wav(samples: np.ndarray, path: Path, sample_rate: int = SAMPLE_RATE) -> None:
    """Save float32 mono audio to a 16-bit PCM WAV file."""
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


def test_synthesis_produces_correct_length() -> None:
    """Sanity check: synth output has expected duration."""
    audio = synthesize_melody(TEST_PATTERN)
    expected_samples = len(TEST_PATTERN) * int(SAMPLE_RATE * (NOTE_DURATION_S + GAP_S))
    assert len(audio) == expected_samples
    assert audio.max() <= 1.0
    assert audio.min() >= -1.0


def test_pyin_transcription_is_accurate(tmp_path: Path) -> None:
    """The killer test: synth → pYIN → expect the exact 8 notes back."""
    audio = synthesize_melody(TEST_PATTERN)
    wav_path = tmp_path / "synth_solo.wav"
    save_wav(audio, wav_path)

    t = transcribe_wav(wav_path, model="pyin")
    detected = [n.pitch for n in t.notes if n.pitch is not None]

    print(f"\nSynthesized: {TEST_PATTERN}")
    print(f"pYIN got:    {detected}")

    # pYIN should get all 8 notes exactly right on this clean synth
    assert detected == TEST_PATTERN, (
        f"pYIN transcription mismatch. Expected {TEST_PATTERN}, got {detected}"
    )


def test_basic_pitch_returns_something(tmp_path: Path) -> None:
    """basic-pitch (polyphonic) returns notes but is noisy for monophonic.

    This test documents the limitation rather than asserting correctness.
    basic-pitch is designed for polyphonic music and detects harmonics
    as separate notes on monophonic input. For jazz solos, use pYIN.
    """
    audio = synthesize_melody(TEST_PATTERN)
    wav_path = tmp_path / "synth_solo.wav"
    save_wav(audio, wav_path)

    t = transcribe_wav(wav_path, model="basic-pitch", monophonic=True)
    detected = [n.pitch for n in t.notes if n.pitch is not None]

    # We just verify the pipeline runs end-to-end, not that basic-pitch
    # gives good output. The fundamental vs harmonics issue is documented
    # in the test_pyin test.
    assert len(detected) > 0, "basic-pitch returned no notes"


def test_synth_to_transcribe_to_search_finds_pattern(tmp_path: Path) -> None:
    """End-to-end: synth → pYIN → search → expect exact match on the known pattern."""
    from solokit.patterns import NGramExtractor, search_patterns

    audio = synthesize_melody(TEST_PATTERN)
    wav_path = tmp_path / "synth_solo.wav"
    save_wav(audio, wav_path)

    t = transcribe_wav(wav_path, model="pyin")
    detected = [n.pitch for n in t.notes if n.pitch is not None]
    assert detected == TEST_PATTERN, f"pYIN failed: {detected}"

    # The synthesized C major scale gives intervals [2,2,1,2,2,1,1] (7 intervals).
    # The first 3-interval gram is (2, 2, 1).
    extractor = NGramExtractor.interval(3)
    grams = extractor.extract_from_pitches(detected)

    expected_first_gram = (2, 2, 1)
    matches = search_patterns(
        list(expected_first_gram),
        grams,
        min_similarity=0.99,  # should be exact
    )

    print(f"\nDetected:     {detected}")
    print(f"Interval grams: {[list(g.values) for g in grams]}")
    print(f"Searching for {expected_first_gram} → {len(matches)} matches")

    assert len(matches) >= 1, (
        f"Expected to find {expected_first_gram} in transcription; "
        f"got 0. Grams: {[g.values for g in grams]}"
    )
    assert matches[0].similarity == 1.0, (
        f"Expected sim=1.0, got sim={matches[0].similarity}"
    )
    assert matches[0].source.values == expected_first_gram


def test_search_finds_transposed_pattern(tmp_path: Path) -> None:
    """Interval search is key-transposition-invariant.

    Synthesize in D, search for the C-major interval pattern (2, 2, 1)
    — should still match because intervals don't depend on key.
    """
    from solokit.patterns import NGramExtractor, search_patterns

    d_pattern = [p + 2 for p in TEST_PATTERN]
    audio = synthesize_melody(d_pattern)
    wav_path = tmp_path / "synth_d_solo.wav"
    save_wav(audio, wav_path)

    t = transcribe_wav(wav_path, model="pyin")
    detected = [n.pitch for n in t.notes if n.pitch is not None]
    assert detected == d_pattern, f"pYIN failed on D: {detected}"

    extractor = NGramExtractor.interval(3)
    grams = extractor.extract_from_pitches(detected)

    matches = search_patterns([2, 2, 1], grams, min_similarity=0.99)
    assert len(matches) >= 1, (
        f"Interval search failed across transposition. "
        f"D-detected: {detected}, grams: {[g.values for g in grams]}"
    )
