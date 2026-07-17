# solokit

Pattern search and analysis toolkit for jazz solo transcriptions.

A modernized reimagining of the [Jazzomat](https://jazzomat.hfm-weimar.de/) research
stack (MeloSpyLib / pymus), built on top of [music21](https://web.mit.edu/music21/)
and compatible with the [Dig That Lick](https://dig-that-lick.hfm-weimar.de/)
pattern-search API.

## What it does

- **Pattern search** — type or extract a melodic phrase, find every historical
  jazz recording that contains a similar shape (interval, pitch, fuzzy, or
  chord-scale pitch class). Same algorithmic family as DTL.
- **Feature extraction** — YAML-driven feature machine. Define a feature
  (pitch histogram, IOI, contour) in YAML; run it over a solo.
- **Audio transcription** *(optional)* — feed in a `.wav` of a jazz solo,
  get back note events using Spotify's `basic-pitch` (polyphonic) or
  `librosa.pyin` (monophonic, recommended for jazz). Plug the result into
  the pattern search and you have a "drop-a-solo, see who else played it"
  loop.
- **Score-informed analysis** — given audio + a transcription, estimate
  tuning deviation, F0 contour, and loudness per note. Ported from
  `pymus/sisa`.
- **FastAPI server** + **Click CLI** + **HTML frontend**.

## Installation

```bash
# Core (pattern search, feature extraction, CLI)
pip install -e .

# With audio transcription support
pip install -e ".[audio]"

# Everything (server, viz, dev tools)
pip install -e ".[all]"
```

## Quick start

```bash
# Search the local WJAZD corpus (456 jazz solos, instant, offline)
solokit search --pattern '-1 -1 4 -5 -2' --corpus wjazzd

# Local Charlie Parker Omnibook (50 solos)
solokit search --pattern '-1 -1 4 -5 -2' --corpus omnibook

# Remote DTL corpus (larger, sometimes down)
solokit search --pattern '-1 -1 4 -5 -2' --corpus dtl

# Extract features from a local transcription
solokit features path/to/solo.mid --config features/basic.yaml

# Transcribe audio (requires pip install 'solokit[audio]')
solokit transcribe path/to/solo.wav

# Start the API server
solokit serve --port 8000
```

```python
from solokit.corpora import WJAZDCorpus  # local, instant

corpus = WJAZDCorpus()  # 456 solos, 200k+ events
matches = corpus.search(
    [-1, -1, 4, -5, -2],  # classic bebop pattern
    transformation="interval",
    min_similarity=0.7,
    max_length_difference=1,
)
for m in matches[:10]:
    print(f"{m.match.similarity:.2f}  {m.year}  {m.performer} - {m.title}")
```

## Architecture

```
solokit/
├── core/          # Solo, Phrase, Transcription (data model)
├── features/      # YAML-driven feature machine + individual features
├── patterns/      # NGram extraction, similarity search, transformations
├── corpora/       # Loaders for DTL (remote), Omnibook + WJAZD (local), EsAC
├── audio/         # Transcription (pYIN, basic-pitch), F0, tuning, loudness
├── api/           # FastAPI server + HTTP client
├── frontend/      # Single-page HTML UI (served at /)
└── cli.py         # Click command-line interface

data/
├── omnibook/      # 50 Charlie Parker solos in MusicXML (CC BY-NC-SA 2.0)
└── wjazzd.db      # 456 hand-transcribed jazz solos in SQLite (ODbL 1.0)
```

The pipeline:

```
audio file ─┐
            ├─→ [transcription] → NoteEvents ─┬─→ [pattern extraction] → search → matches
score MIDI ─┘                                  └─→ [feature machine]   → features
```

## Frontend

Run `solokit serve` and open <http://localhost:8000/>. You'll get a single-page
web UI with:

- A **pattern search** panel — type or paste a pattern, pick corpora, set
  similarity / length tolerance, see matches in a sortable table.
- A **drop a solo** panel — upload a `.wav` of a jazz solo, get the
  transcription as note chips, derive the interval pattern, and
  one-click "use in search" to chain into the corpus search.
- Live status indicator in the header showing server version + available
  corpora.
- Per-corpus error reporting — if DTL is down, the local corpora still
  return results and a toast tells you what failed.

CORS is permissive by default for local dev. Override with the
`SOLOKIT_CORS_ORIGINS` env var (comma-separated).

## Testing

```bash
# Unit + endpoint tests (pytest)
pytest

# End-to-end test (Puppeteer) — boots a uvicorn, runs the browser test, tears down
tests/e2e/run.sh
```

The e2e test takes screenshots into `tests/e2e/screenshots/` so you can
eyeball what the UI looks like in headless mode.

## Corpora

| Name | Source | Size | Offline? | License | Coverage |
|---|---|---|---|---|---|
| `wjazzd` | local SQLite | **456 solos** | **Yes** | ODbL 1.0 | Coltrane (20), Miles (19), Parker (17), Bebop→Free |
| `omnibook` | local MusicXML | 50 solos | **Yes** | CC BY-NC-SA 2.0 | Charlie Parker |
| `dtl` | DTL HTTP API | 1736+ solos | No (network) | (research use) | DTL1000 auto-transcribed |
| `esac` | (not yet implemented) | 6000+ folk songs | — | research | Essen Folk Song Collection |

Use `--corpus wjazzd` for instant offline searches across the largest local corpus. Use `--corpus dtl` for the full DTL1000 catalog when the network is up.

## License

MIT. The bundled Omnibook corpus is CC BY-NC-SA 2.0 (non-commercial). The bundled WJAZD database is ODbL 1.0.
