"""Local Weimar Jazz Database (WJAZD) corpus — 456 hand-transcribed jazz solos.

Data source: https://jazzomat.hfm-weimar.de/download/download.html
License: ODbL 1.0 (Open Data Commons), free for research
File: wjazzd.db (SQLite, 41MB)

Schema (key tables):
- solo_info: 456 solos with melid, performer, title, instrument, key,
  style, avgtempo, chorus_count
- melody: 200,809 events with onset (seconds), pitch (MIDI),
  duration (seconds), bar, beat, plus F0/loudness analysis
- composition_info: title, composer, form, tonality
- record_info: album, label, release date

Compared to the Omnibook corpus, WJAZD has 9x more solos (456 vs 50),
covers 100+ performers, all major styles (Bebop through Free Jazz),
and rich per-note metadata (F0 deviation, loudness, beat positions)
that the Omnibook MusicXML files lack.

Like OmnibookCorpus, this is fully local — no network, instant searches,
and works when DTL is down.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from solokit.corpora.base import Corpus, CorpusError, SearchResult
from solokit.patterns.similarity import search_patterns
from solokit.patterns.transformations import transform as _transform

if TYPE_CHECKING:
    from solokit.core.solo import Solo
    from solokit.patterns.ngram import NGram

# Instrument code → human-readable name (subset; full map at jazzomat docs)
INSTRUMENT_NAMES = {
    "as": "Alto saxophone",
    "ts": "Tenor saxophone",
    "ss": "Soprano saxophone",
    "bs": "Baritone saxophone",
    "ts-c": "C melody saxophone",
    "tp": "Trumpet",
    "cor": "Cornet",
    "tb": "Trombone",
    "cl": "Clarinet",
    "bcl": "Bass clarinet",
    "p": "Piano",
    "g": "Guitar",
    "vib": "Vibraphone",
}


def _load_solo(cur: sqlite3.Cursor, melid: int) -> "Solo | None":
    """Load a single solo (metadata + notes) from the WJAZD DB."""
    from solokit.core.solo import Solo, SoloMetadata
    from solokit.core.transcription import NoteEvent, Transcription

    # Metadata
    row = cur.execute(
        """
        SELECT s.melid, s.performer, s.title, s.titleaddon, s.instrument,
               s.style, s.avgtempo, s.tempoclass, s.key, s.signature,
               s.chorus_count, c.composer, c.form,
               r.artist, r.recordtitle, r.releasedate
        FROM solo_info s
        LEFT JOIN composition_info c ON s.compid = c.compid
        LEFT JOIN record_info r ON s.recordid = r.recordid
        WHERE s.melid = ?
        """,
        (melid,),
    ).fetchone()
    if not row:
        return None

    (
        mid, performer, title, titleaddon, instrument, style, avgtempo,
        tempoclass, key, signature, chorus_count, composer, form,
        rec_artist, rec_title, release_date,
    ) = row
    full_title = f"{title} {titleaddon}".strip() if title else "Unknown"

    # Notes — onset in seconds, pitch in MIDI
    notes_raw = cur.execute(
        """
        SELECT onset, pitch, duration
        FROM melody
        WHERE melid = ? AND pitch > 0
        ORDER BY onset
        """,
        (melid,),
    ).fetchall()

    if len(notes_raw) < 4:
        return None

    # Convert seconds → beats using the solo's avg tempo
    bpm = avgtempo if avgtempo and avgtempo > 0 else 120.0
    seconds_per_beat = 60.0 / bpm
    note_events = tuple(
        NoteEvent(
            pitch=int(round(pitch)),
            onset_beat=onset / seconds_per_beat,
            duration_beats=duration / seconds_per_beat,
            velocity=None,
        )
        for (onset, pitch, duration) in notes_raw
    )

    transcription = Transcription.from_note_sequence(
        note_events, tempo_bpm=bpm, time_signature=_parse_time_sig(signature), key_signature=key
    )

    # Parse year from release_date (e.g. "1989" or "1989-03-15")
    year = None
    if release_date:
        try:
            year = int(release_date[:4])
        except (ValueError, TypeError):
            year = None

    return Solo.from_transcription(
        SoloMetadata(
            melid=f"wjazzd-{mid}",
            title=full_title,
            performer=performer or "Unknown",
            recording_year=year,
            track_year=year,
            instrument=INSTRUMENT_NAMES.get(instrument or "", instrument or ""),
            key=key,
            tempo_bpm=int(bpm) if avgtempo else None,
            style=style,
            source_corpus="wjazzd",
            extra={
                "wjazzd_melid": mid,
                "composer": composer,
                "form": form,
                "chorus_count": chorus_count,
                "tempoclass": tempoclass,
            },
        ),
        transcription,
    )


def _parse_time_sig(sig: str | None) -> tuple[int, int]:
    """Parse '4/4' into (4, 4). Defaults to (4, 4) on failure."""
    if not sig:
        return (4, 4)
    try:
        num, denom = sig.split("/")
        return (int(num), int(denom))
    except (ValueError, AttributeError):
        return (4, 4)


def _transform_for_search(pitches, kind):
    return _transform(pitches, kind)


class WJAZDCorpus(Corpus):
    """Local 456-solo Weimar Jazz Database (WJAZD) corpus.

    Backed by a SQLite database. Loaded lazily: we read solo_info at
    construction, then open a new connection for each `search()` to
    stream the melody events. For 200k+ events, this keeps memory
    usage low (~10MB) while supporting instant pattern search.

    Default data location: `<solokit>/data/wjazzd.db`
    Override with `db_path=...`.

    Usage::

        corpus = WJAZDCorpus()
        results = corpus.search(
            [-1, -1, 4, -5, -2],   # a 5-interval pattern
            transformation="interval",
            min_similarity=0.7,
        )
        for r in results:
            print(f"{r.title} - {r.performer} ({r.year})")
    """

    name = "wjazzd"

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent.parent / "data" / "wjazzd.db"
        self._db_path = Path(db_path)
        if not self._db_path.is_file():
            msg = (
                f"WJAZD database not found at {self._db_path}. "
                "Download from https://jazzomat.hfm-weimar.de/download/downloads/wjazzd.db"
            )
            raise CorpusError(msg)

        # Eagerly load the solo_info index
        con = sqlite3.connect(str(self._db_path))
        try:
            self._melids: list[int] = [
                r[0]
                for r in con.execute("SELECT melid FROM solo_info ORDER BY melid").fetchall()
            ]
        finally:
            con.close()

    def __iter__(self) -> Iterator[Solo]:
        con = sqlite3.connect(str(self._db_path))
        try:
            cur = con.cursor()
            for melid in self._melids:
                solo = _load_solo(cur, melid)
                if solo is not None:
                    yield solo
        finally:
            con.close()

    def __len__(self) -> int:
        return len(self._melids)

    def get_solo_by_melid(self, melid: int) -> Solo | None:
        """Load a specific solo by its WJAZD melid."""
        con = sqlite3.connect(str(self._db_path))
        try:
            return _load_solo(con.cursor(), melid)
        finally:
            con.close()

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
        """Search WJAZD for pattern matches.

        Streams all 200k+ melody events from the SQLite DB, extracts
        n-grams on the fly, and searches. Returns SearchResults sorted
        by similarity. Use the `performer` and `style` fields to filter
        results programmatically.
        """
        from solokit.patterns.ngram import NGram  # avoid circular import

        # Gram length is the length of the pattern (in values).
        # The pattern is already in the target representation
        # (e.g. [-1, -1, 4, -5, -2] is a 5-interval gram).
        n = len(pattern)

        con = sqlite3.connect(str(self._db_path))
        try:
            cur = con.cursor()

            # Build grams with source_id pointing to the solo's melid
            # We need onset_beat too, so we read (onset, pitch) pairs
            # per solo and extract grams locally.
            gram_list: list[NGram] = []
            gram_to_solo: dict[int, Solo] = {}

            for melid in self._melids:
                solo = _load_solo(cur, melid)
                if solo is None or len(solo.transcription.pitches) < n:
                    continue
                pitches = solo.transcription.pitches
                onsets = [n.onset_beat for n in solo.transcription.notes if n.pitch is not None]
                transformed = _transform_for_search(pitches, transformation)
                for i in range(len(transformed) - n + 1):
                    gram = NGram(
                        values=tuple(transformed[i : i + n]),
                        source_id=solo.metadata.melid,
                        onset_beat=onsets[i] if i < len(onsets) else None,
                    )
                    gram_list.append(gram)
                    gram_to_solo[id(gram)] = solo

        finally:
            con.close()

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
                    audio_url=None,  # WJAZD has no audio URLs in the DB
                    database="wjazzd",
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
