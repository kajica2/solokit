"""Local Omnibook corpus — 50 Charlie Parker solo transcriptions in MusicXML.

Data source: https://homepages.loria.fr/evincent/omnibook/
License: CC BY-NC-SA 2.0 (research use, attribution required)

Citation:
    Ken Déguernel, Emmanuel Vincent, and Gérard Assayag.
    Using Multidimensional Sequences for Improvisation in the OMax Paradigm,
    in Proceedings of the 13th Sound and Music Computing Conference, 2016.

Each MusicXML file contains both the head (theme) and the solo (Parker
improvisation). We segment the file to extract just the solo section —
the first 30-40 beats are typically the head (whole notes), and the
solo starts when note density increases.

This corpus is a fully local fallback for DTL. It's small (~10MB, 50
files), loads in <1 second, and lets you search even when DTL is down.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING

from solokit.corpora.base import Corpus, CorpusError, SearchResult
from solokit.patterns.similarity import search_patterns
from solokit.patterns.transformations import transform as _transform

# Local alias to keep the search code short
def transform_for_search(pitches, kind):
    return _transform(pitches, kind)

if TYPE_CHECKING:
    from solokit.core.solo import Solo
    from solokit.patterns.ngram import NGram


def _build_solo_from_musicxml(path: Path) -> "Solo | None":
    """Load a MusicXML file, extract the solo, return as a Solo.

    Returns None if the file can't be parsed or contains no usable notes.
    """
    try:
        import music21 as m21
    except ImportError as exc:
        msg = "OmnibookCorpus requires music21: pip install music21"
        raise CorpusError(msg) from exc

    from solokit.core.solo import Solo, SoloMetadata
    from solokit.core.transcription import NoteEvent, Transcription

    try:
        score = m21.converter.parse(str(path))
    except Exception as exc:  # noqa: BLE001
        # Skip files we can't parse
        return None

    # The solo part is typically the second part (index 1) — first part is piano.
    # Fall back to the first part if only one exists.
    if not score.parts:
        return None
    solo_part = score.parts[1] if len(score.parts) > 1 else score.parts[0]

    # Extract notes with timing. music21's `n.offset` is measure-relative,
    # so we compute absolute offset from the measure number. All Omnibook
    # files are in 4/4, so each measure = 4 beats.
    notes_raw: list[tuple[float, float, int]] = []  # (onset_beats, dur_beats, midi)
    for n in solo_part.recurse().notes:
        # Skip harmony annotations (ChordSymbol is a Chord subclass but
        # represents written chord symbols like "Cmaj7", not sound events)
        if isinstance(n, m21.harmony.ChordSymbol):
            continue
        if not n.measureNumber:
            # Skip elements outside any measure (rare)
            continue
        abs_offset = (n.measureNumber - 1) * 4.0 + float(n.offset)
        if isinstance(n, m21.note.Rest):
            notes_raw.append((abs_offset, float(n.duration.quarterLength), -1))
            continue
        if isinstance(n, m21.note.Note):
            notes_raw.append((abs_offset, float(n.duration.quarterLength), int(n.pitch.midi)))
        elif isinstance(n, m21.chord.Chord):
            # Real chord (multiple simultaneous pitched notes). Take the
            # top note as the melody.
            top_pitch = max(int(p.midi) for p in n.pitches)
            notes_raw.append((abs_offset, float(n.duration.quarterLength), top_pitch))

    if not notes_raw:
        return None

    # Find where the solo starts. The head is whole-note chords in the first
    # 30-40 beats. The solo begins when the median note duration over a 4-beat
    # window drops below 1 beat.
    onset_threshold = _find_solo_start(notes_raw, window=4.0, dur_threshold=1.0, min_offset=8.0)

    # Filter to solo region, drop rests, and re-base onset to 0
    solo_notes = [
        (onset - onset_threshold, dur, pitch)
        for (onset, dur, pitch) in notes_raw
        if onset >= onset_threshold and pitch >= 0
    ]

    if len(solo_notes) < 4:
        return None

    note_events = tuple(
        NoteEvent(
            pitch=pitch,
            onset_beat=onset,
            duration_beats=dur,
            velocity=None,
        )
        for (onset, dur, pitch) in solo_notes
    )

    # Title from filename (e.g., "Anthropology.xml" → "Anthropology")
    title = path.stem.replace("_", " ")

    transcription = Transcription.from_note_sequence(note_events, tempo_bpm=120.0)
    return Solo.from_transcription(
        SoloMetadata(
            melid=path.stem,
            title=title,
            performer="Charlie Parker",
            recording_year=None,  # Omnibook doesn't include year metadata
            instrument="alto saxophone",
            source_corpus="omnibook",
        ),
        transcription,
    )


def _find_solo_start(
    notes: list[tuple[float, float, int]],
    *,
    window: float = 4.0,
    dur_threshold: float = 1.0,
    min_offset: float = 8.0,
) -> float:
    """Find the onset offset where the solo begins.

    The Omnibook head consists of whole-note chords (durations of 4+ beats).
    The solo starts when note durations drop below `dur_threshold` beats.

    Slides a window of `window` beats across the piece; finds the first
    window past `min_offset` where the median note duration is < threshold.
    """
    # Collect all (onset, duration) pairs, sorted
    onsets_durs = sorted((n[0], n[1]) for n in notes)
    if not onsets_durs:
        return 0.0

    # Default: return min_offset if we can't find a clear break
    default = min_offset

    # Slide through the piece in 1-beat increments
    for start in [float(i) for i in range(int(min_offset), int(onsets_durs[-1][0]))]:
        end = start + window
        durs_in_window = [d for (o, d) in onsets_durs if start <= o < end]
        if len(durs_in_window) < 3:
            continue
        if median(durs_in_window) < dur_threshold:
            return start

    return default


class OmnibookCorpus(Corpus):
    """Local 50-solo Charlie Parker Omnibook corpus (MusicXML).

    All data is loaded into memory at construction. Searches are local
    and instant — no network dependency.

    Default data location: `<solokit>/data/omnibook/`
    Override with `data_dir=...`.

    Usage::

        corpus = OmnibookCorpus()  # uses bundled data
        results = corpus.search(
            [-1, -1, 4, -5, -2],   # a 5-interval pattern
            transformation="interval",
            min_similarity=0.7,
        )
        for r in results:
            print(f"{r.title} (from {r.solo.metadata.title})")
    """

    name = "omnibook"

    def __init__(self, data_dir: str | Path | None = None) -> None:
        if data_dir is None:
            # Default to the bundled data.
            # Layout: src/solokit/corpora/omnibook.py → ../../.. → project root
            data_dir = Path(__file__).parent.parent.parent.parent / "data" / "omnibook"
        self._data_dir = Path(data_dir)
        if not self._data_dir.is_dir():
            msg = f"Omnibook data not found at {self._data_dir}"
            raise CorpusError(msg)

        # Eagerly load all solos
        self._solos: list[Solo] = []
        for xml_path in sorted(self._data_dir.glob("*.xml")):
            solo = _build_solo_from_musicxml(xml_path)
            if solo is not None:
                self._solos.append(solo)

    def __iter__(self) -> Iterable[Solo]:
        return iter(self._solos)

    def __len__(self) -> int:
        return len(self._solos)

    @property
    def total_notes(self) -> int:
        return sum(len(s.transcription.notes) for s in self._solos)

    def search(
        self,
        pattern,
        *,
        transformation: str = "interval",
        min_similarity: float = 0.8,
        max_length_difference: int = 0,
        max_edit_distance: int | None = None,
        min_frequency: int = 1,
        limit: int = 50,
    ) -> list[SearchResult]:
        """Search the Omnibook for pattern matches.

        Builds an NGram corpus from all 50 solos, searches, then groups
        results by solo so you can see "which songs contain this pattern".
        """
        from solokit.patterns.ngram import NGram  # avoid circular import

        # Gram length is the length of the pattern (in values).
        n = len(pattern)

        # Build grams with onset_beat carried over from the source notes.
        gram_list: list[NGram] = []
        gram_to_solo: dict[int, "Solo"] = {}

        for solo in self._solos:
            pitches = solo.transcription.pitches
            onsets = [
                n.onset_beat for n in solo.transcription.notes if n.pitch is not None
            ]
            if len(pitches) < n:
                continue
            transformed = transform_for_search(pitches, transformation)
            for i in range(len(transformed) - n + 1):
                gram = NGram(
                    values=tuple(transformed[i : i + n]),
                    source_id=solo.metadata.melid,
                    onset_beat=onsets[i] if i < len(onsets) else None,
                )
                gram_list.append(gram)
                gram_to_solo[id(gram)] = solo

        raw_matches = search_patterns(
            list(pattern),
            gram_list,
            min_similarity=min_similarity,
            max_length_difference=max_length_difference,
            max_edit_distance=max_edit_distance,
            min_frequency=min_frequency,
        )

        results: list[SearchResult] = []
        for match in raw_matches[:limit]:
            solo = gram_to_solo.get(id(match.source))
            if solo is None:
                continue
            results.append(
                SearchResult(
                    match=match,
                    performer=solo.metadata.performer,
                    title=solo.metadata.title,
                    year=solo.metadata.recording_year,
                    melid=solo.metadata.melid,
                    audio_url=None,  # Omnibook has no audio
                    database="omnibook",
                    instrument=solo.metadata.instrument,
                    start_position=match.source.onset_beat,
                    duration=None,
                )
            )
        return results

    def all_ngrams(
        self,
        n: int = 5,
        transformation: str = "interval",
    ) -> Iterable[NGram]:
        """Yield all NGrams across the corpus (offline indexing)."""
        from solokit.patterns.ngram import NGramExtractor

        extractor = NGramExtractor(n=n, transformation=transformation)  # type: ignore[arg-type]
        for solo in self:
            yield from extractor.extract_from_solo(solo)
