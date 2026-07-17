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
  get back note events using Spotify's `basic-pitch`. Plug the result into
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
# Search the LOCAL Omnibook corpus (50 Charlie Parker solos, instant, offline)
solokit search --pattern '-1 -1 4 -5 -2' --corpus omnibook

# Same pattern against the remote DTL corpus (larger, sometimes down)
solokit search --pattern '-1 -1 4 -5 -2' --corpus dtl

# Extract features from a local transcription
solokit features path/to/solo.mid --config features/basic.yaml

# Transcribe audio (requires pip install 'solokit[audio]')
solokit transcribe path/to/solo.wav

# Start the API server
solokit serve --port 8000
```

```python
from solokit.patterns import NGramExtractor, search_patterns
from solokit.corpora import OmnibookCorpus  # local, instant

# Extract a pattern from a phrase
extractor = NGramExtractor.interval(5)
pattern = extractor.extract_from_pitches([60, 59, 58, 62, 57])[0].values

# Search the local Omnibook corpus
corpus = OmnibookCorpus()
matches = corpus.search(pattern, min_similarity=0.7, max_length_difference=1)
for match in matches[:10]:
    print(f"{match.match.similarity:.2f}  {match.title}  (beat {match.start_position:.1f})")
```

## Architecture

```
solokit/
├── core/          # Solo, Phrase, Transcription (data model)
├── features/      # YAML-driven feature machine + individual features
├── patterns/      # NGram extraction, similarity search, transformations
├── corpora/       # Loaders for DTL (remote), Omnibook (local), WJAZD, EsAC
├── audio/         # Transcription (pYIN, basic-pitch), F0, tuning, loudness
├── api/           # FastAPI server + HTTP client
└── cli.py         # Click command-line interface

data/
└── omnibook/      # 50 Charlie Parker solos in MusicXML (CC BY-NC-SA 2.0)
```

The pipeline:

```
audio file ─┐
            ├─→ [transcription] → NoteEvents ─┬─→ [pattern extraction] → search → matches
score MIDI ─┘                                  └─→ [feature machine]   → features
```

## Corpora

| Name | Source | Size | Offline? | License |
|---|---|---|---|---|
| `dtl` | DTL HTTP API | 1736+ solos | No (network) | (research use) |
| `omnibook` | local MusicXML | 50 solos | **Yes** | CC BY-NC-SA 2.0 |
| `wjazzd` | (not yet implemented) | 456 solos | — | research |
| `esac` | (not yet implemented) | 6000+ folk songs | — | research |

Use `--corpus omnibook` for fast offline searches. The Omnibook is small but contains canonical Charlie Parker phrases (validated against DTL on 2026-07-16).

## License

MIT. The bundled Omnibook corpus is CC BY-NC-SA 2.0 (non-commercial).
