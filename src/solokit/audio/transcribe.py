"""Audio → MIDI transcription.

The "missing piece" in the original jazzomat stack. The jazzomat pymus
library was score-informed: you gave it audio + a transcription and it
estimated tuning, F0, loudness. It did NOT take a raw .wav and produce
a transcription.

We use Spotify's `basic-pitch` (https://github.com/spotify/basic-pitch)
as the default transcription engine. It's a CNN trained on multi-
instrument polyphonic music — for monophonic jazz solos (the typical
use case) we apply a greedy monophonic post-processing filter that
collapses overlapping notes to a single dominant note.

Other engines to consider:
- `crema` (multi-model, slow but accurate)
- `mt3` (Google's transcription transformer, requires TF)
- `omnizart` (Spotify, multi-instrument)
- `librosa.pyin` for F0-only extraction (very fast, no onset detection)

Install with:  pip install solokit[audio]
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from solokit.core.transcription import Transcription


# ----------------------------------------------------------------------------
# Monophonic post-processing
# ----------------------------------------------------------------------------


def _to_monophonic_greedy(
    note_events: Sequence[tuple[float, float, int, float, ...]],
) -> list[tuple[float, float, int, float, ...]]:
    """Collapse polyphonic note_events to monophonic via greedy selection.

    basic-pitch outputs every note it detects, including overlapping ones
    from overtones and noise. For monophonic content (a single instrument
    playing one note at a time) we want at most one note at any moment.

    Algorithm: sort by start time; for each event, if it doesn't overlap
    the previous accepted event, accept it. If it does overlap, keep the
    one with the higher velocity (more confident).

    Args:
        note_events: Sequence of (start_s, end_s, pitch_midi, velocity, ...)
            tuples as returned by basic-pitch.

    Returns:
        A non-overlapping subset of the input events, sorted by start time.
    """
    if not note_events:
        return []

    # Sort by start time, then by velocity desc (so higher-confidence notes
    # come first when there's a tie)
    sorted_events = sorted(
        note_events,
        key=lambda e: (float(e[0]), -float(e[3]) if len(e) > 3 else 0.0),
    )

    accepted: list[tuple[float, float, int, float, ...]] = []
    for event in sorted_events:
        start = float(event[0])
        end = float(event[1])
        # Find any accepted event that overlaps
        replaced = False
        for i, existing in enumerate(accepted):
            ex_start = float(existing[0])
            ex_end = float(existing[1])
            if start < ex_end and end > ex_start:
                # Overlap. Keep the higher-velocity one.
                ev_vel = float(event[3]) if len(event) > 3 else 0.0
                ex_vel = float(existing[3]) if len(existing) > 3 else 0.0
                if ev_vel > ex_vel:
                    accepted[i] = event
                replaced = True
                break
        if not replaced:
            accepted.append(event)

    # Re-sort by start time for a clean output
    accepted.sort(key=lambda e: float(e[0]))
    return accepted


def _merge_close_notes(
    notes: list,  # noqa: ANN001 — list[NoteEvent], kept loose to avoid circular import
    *,
    max_gap_s: float = 0.1,
    max_pitch_diff: int = 0,
) -> list:
    """Merge adjacent notes that are close in time and have the same pitch.

    pYIN's F0 contour has small wobbles (vibrato, breath) that cause it to
    briefly go unvoiced inside a sustained note. The default segmentation
    emits a new "note" at each unvoiced gap, so a single held vowel turns
    into 2-5 same-pitch notes with short durations and small gaps.

    This pass glues them back together: if note N+1 starts within
    `max_gap_s` of where note N ended, and the pitches are within
    `max_pitch_diff` semitones, extend note N to cover note N+1.

    Args:
        notes: Sorted list of NoteEvent (by onset_beat, ascending).
        max_gap_s: Maximum gap (in seconds) between notes to consider for merge.
        max_pitch_diff: Maximum pitch difference in semitones (0 = exact match,
            1 = allow ±1 semitone for vibrato).

    Returns:
        New list with adjacent close notes merged. Order is preserved.
    """
    if not notes:
        return notes
    merged: list = [notes[0]]
    # Tiny epsilon to handle floating-point comparison (0.4 - 0.3 = 0.10000...3)
    eps = 1e-9
    for note in notes[1:]:
        prev = merged[-1]
        gap = note.onset_beat - (prev.onset_beat + prev.duration_beats)
        if gap <= max_gap_s + eps and abs(note.pitch - prev.pitch) <= max_pitch_diff:
            # Replace last with a note that extends over both
            from solokit.core.transcription import NoteEvent  # local to avoid cycles

            merged[-1] = NoteEvent(
                pitch=prev.pitch,
                onset_beat=prev.onset_beat,
                duration_beats=(note.onset_beat + note.duration_beats) - prev.onset_beat,
                velocity=prev.velocity,
            )
        else:
            merged.append(note)
    return merged


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------


def transcribe_wav(
    path: str | Path,
    *,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    min_note_length_ms: float = 50.0,
    min_frequency: float | None = None,
    max_frequency: float | None = None,
    min_velocity: float = 0.3,
    monophonic: bool = True,
    model: str = "basic-pitch",
) -> Transcription:
    """Transcribe a .wav file to a Transcription.

    Args:
        path: Path to the audio file (any format librosa/soundfile can read).
        onset_threshold: Confidence threshold for note onset (0-1).
            Only used by model="basic-pitch".
        frame_threshold: Confidence threshold for note continuation (0-1).
            Only used by model="basic-pitch".
        min_note_length_ms: Drop notes shorter than this.
        min_frequency: Drop notes below this frequency in Hz (e.g. 60 for low E).
        max_frequency: Drop notes above this frequency in Hz.
        min_velocity: Drop notes with confidence below this (0-1).
            Only used by model="basic-pitch".
        monophonic: If True, apply a monophonic post-processing filter.
            Required for monophonic content (jazz solos) when using
            model="basic-pitch" because the polyphonic model outputs
            spurious overlapping notes from overtones.
        model: Which transcription model. Options:
            - "basic-pitch" (default): Spotify's polyphonic CNN, with
              monophonic post-processing. Works but is the wrong tool
              for clean monophonic input — it detects harmonics as
              separate notes. Use this for multi-instrument or unclear
              audio.
            - "pyin": librosa's pYIN. Designed for monophonic F0
              extraction. Better choice for clean jazz solos. Returns
              note events by segmenting the F0 contour.

    Returns:
        A Transcription with the detected note events.

    Raises:
        ImportError: If the audio extra is not installed.
        FileNotFoundError: If the audio file doesn't exist.
    """
    if model == "pyin":
        return _transcribe_pyin(
            path,
            min_note_length_ms=min_note_length_ms,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
        )

    if model != "basic-pitch":
        msg = f"Unknown model {model!r}; choose from 'basic-pitch', 'pyin'"
        raise NotImplementedError(msg)

    try:
        # scipy >= 1.14 moved `gaussian` from scipy.signal to scipy.signal.windows.
        # basic-pitch 0.3.0 still calls the old name. Patch it in before
        # importing basic_pitch so internal calls resolve.
        import scipy.signal
        import scipy.signal.windows

        if not hasattr(scipy.signal, "gaussian"):
            scipy.signal.gaussian = scipy.signal.windows.gaussian  # type: ignore[attr-defined]

        from basic_pitch import ICASSP_2022_MODEL_PATH  # type: ignore[import-untyped]  # noqa: F401
        from basic_pitch.inference import predict as bp_predict  # type: ignore[import-untyped]
    except ImportError as exc:
        msg = (
            "Failed to import basic-pitch. "
            "Install with: pip install 'solokit[audio]'. "
            f"Underlying error: {type(exc).__name__}: {exc}"
        )
        raise ImportError(msg) from exc

    p = Path(path)
    if not p.exists():
        msg = f"Audio file not found: {p}"
        raise FileNotFoundError(msg)

    # Use the ONNX model if available — the default TF SavedModel is broken
    # with TF >= 2.16 (basic-pitch uses an old `_UserObject.add_slot` API).
    # ONNX model ships with basic-pitch and works with onnxruntime.
    import os
    onnx_path = os.environ.get("SOLOKIT_BASIC_PITCH_MODEL")
    if onnx_path is None:
        # basic_pitch stores its model in a directory; sibling .onnx is the ONNX variant
        from pathlib import Path as _P

        model_dir = _P(ICASSP_2022_MODEL_PATH)
        candidate = model_dir.parent / f"{model_dir.name}.onnx"
        if candidate.is_file():
            onnx_path = str(candidate)
    if onnx_path is None:
        # Fall back to the default (will fail on TF >= 2.16 with a clearer message)
        model_arg: str | os.PathLike = ICASSP_2022_MODEL_PATH
    else:
        model_arg = onnx_path

    # basic_pitch.predict returns (model_output, midi_data, note_events)
    # - model_output: Dict[str, ndarray] with keys like 'onset', 'frame', 'contour'
    # - midi_data: pretty_midi.PrettyMIDI
    # - note_events: List[Tuple[start_s, end_s, pitch_midi, velocity, [pitch_bend_values]]]
    result = bp_predict(str(p), model_arg, onset_threshold=onset_threshold, frame_threshold=frame_threshold)
    note_events = result[2]

    # Monophonic post-processing: collapse overlapping notes.
    # Without this, basic-pitch outputs every note it detects (including
    # overtones and noise) and a 6-note input becomes 60-130 events.
    if monophonic:
        note_events = _to_monophonic_greedy(note_events)

    # Apply filters
    from solokit.core.transcription import NoteEvent, Transcription

    min_len_s = min_note_length_ms / 1000.0
    filtered: list[NoteEvent] = []
    for event in note_events:
        start_s = float(event[0])
        end_s = float(event[1])
        pitch = int(round(float(event[2])))
        velocity_raw = float(event[3]) if len(event) > 3 else 1.0
        if end_s - start_s < min_len_s:
            continue
        if min_frequency is not None:
            f_hz = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
            if f_hz < min_frequency:
                continue
        if max_frequency is not None:
            f_hz = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
            if f_hz > max_frequency:
                continue
        if velocity_raw < min_velocity:
            continue
        filtered.append(
            NoteEvent(
                pitch=pitch,
                onset_beat=start_s,  # in seconds for now; converted to beats below
                duration_beats=end_s - start_s,
                velocity=int(round(velocity_raw * 127)) if velocity_raw <= 1.0 else int(round(velocity_raw)),
            )
        )

    # Detect tempo via librosa and convert seconds → beats.
    # This is a hack — ideally the user provides a known tempo or the
    # transcription engine returns beat-aligned events. For jazz solos
    # the tempo is often unknown and we default to 120 BPM.
    try:
        import librosa  # type: ignore[import-untyped]
        import numpy as np  # type: ignore[import-untyped]

        y, sr = librosa.load(str(p), sr=None, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    except ImportError:
        bpm = 120.0

    # Convert to beats
    seconds_per_beat = 60.0 / bpm
    beat_filtered = [
        NoteEvent(
            pitch=n.pitch,
            onset_beat=n.onset_beat / seconds_per_beat,
            duration_beats=n.duration_beats / seconds_per_beat,
            velocity=n.velocity,
        )
        for n in filtered
    ]

    return Transcription.from_note_sequence(beat_filtered, tempo_bpm=bpm)


# ----------------------------------------------------------------------------
# pYIN backend (monophonic F0 detection)
# ----------------------------------------------------------------------------


def _transcribe_pyin(
    path: str | Path,
    *,
    min_note_length_ms: float = 50.0,
    min_frequency: float | None = None,
    max_frequency: float | None = None,
) -> Transcription:
    """Transcribe using librosa's pYIN (monophonic F0 detection).

    pYIN is a classical signal-processing approach to monophonic pitch
    detection, well-suited for clean jazz solos (single instrument, no
    harmony). It returns an F0 contour; we segment contiguous voiced
    regions into note events.

    Much better than basic-pitch for clean monophonic input — basic-pitch
    detects harmonics as separate notes because it was trained on
    polyphonic data.
    """
    try:
        import librosa  # type: ignore[import-untyped]
        import numpy as np  # type: ignore[import-untyped]
    except ImportError as exc:
        msg = "pYIN requires librosa: pip install 'solokit[audio]'"
        raise ImportError(msg) from exc

    from solokit.core.transcription import NoteEvent, Transcription

    p = Path(path)
    if not p.exists():
        msg = f"Audio file not found: {p}"
        raise FileNotFoundError(msg)

    y, sr = librosa.load(str(p), sr=None, mono=True)

    f0_min = min_frequency if min_frequency is not None else librosa.note_to_hz("C2")
    f0_max = max_frequency if max_frequency is not None else librosa.note_to_hz("C7")

    # librosa.pyin returns (f0_hz, voiced_flag, voiced_prob)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=f0_min,
        fmax=f0_max,
        sr=sr,
    )

    # Frame times
    frame_times = librosa.frames_to_time(
        np.arange(len(f0)),
        sr=sr,
        hop_length=512,  # librosa.pyin default
    )

    # Segment the F0 contour into notes: contiguous voiced regions = notes
    min_len_frames = max(1, int(min_note_length_ms / 1000.0 * sr / 512))
    notes: list[NoteEvent] = []

    in_note = False
    note_start_frame = 0
    note_f0s: list[float] = []

    for i, is_voiced in enumerate(voiced_flag):
        if is_voiced and f0[i] > 0:
            if not in_note:
                in_note = True
                note_start_frame = i
                note_f0s = []
            note_f0s.append(f0[i])
        else:
            if in_note:
                note_len = i - note_start_frame
                if note_len >= min_len_frames and note_f0s:
                    median_f0 = float(np.median(note_f0s))
                    if median_f0 > 0:
                        # Convert frequency → MIDI pitch
                        midi_pitch = int(round(69 + 12 * np.log2(median_f0 / 440.0)))
                        onset_s = float(frame_times[note_start_frame])
                        offset_s = float(frame_times[min(i, len(frame_times) - 1)])
                        notes.append(
                            NoteEvent(
                                pitch=midi_pitch,
                                onset_beat=onset_s,
                                duration_beats=offset_s - onset_s,
                                velocity=None,
                            )
                        )
                in_note = False

    # Handle a note that extends to the end
    if in_note and note_f0s:
        median_f0 = float(np.median(note_f0s))
        if median_f0 > 0:
            midi_pitch = int(round(69 + 12 * np.log2(median_f0 / 440.0)))
            onset_s = float(frame_times[note_start_frame])
            offset_s = float(frame_times[-1])
            notes.append(
                NoteEvent(
                    pitch=midi_pitch,
                    onset_beat=onset_s,
                    duration_beats=offset_s - onset_s,
                    velocity=None,
                )
            )

    # Detect tempo for beat conversion
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    except Exception:  # noqa: BLE001
        bpm = 120.0

    seconds_per_beat = 60.0 / bpm
    beat_notes = [
        NoteEvent(
            pitch=n.pitch,
            onset_beat=n.onset_beat / seconds_per_beat,
            duration_beats=n.duration_beats / seconds_per_beat,
            velocity=n.velocity,
        )
        for n in notes
    ]

    # Merge adjacent close notes (pYIN's vibrato artifact).
    # Default: 200ms gap tolerance, ±1 semitone pitch tolerance — handles
    # both pitch drift and the brief unvoiced gaps from vibrato.
    beat_notes = _merge_close_notes(beat_notes, max_gap_s=0.2, max_pitch_diff=1)

    return Transcription.from_note_sequence(beat_notes, tempo_bpm=bpm)
