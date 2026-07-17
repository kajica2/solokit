# Changelog

All notable changes to solokit are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2026-07-17

### Added

- **`solokit download-corpora`** CLI command. New users can bootstrap
  the WJAZD SQLite (41MB) and Omnibook MusicXML (10MB) into `data/`
  with one command. The data is excluded from the PyPI sdist for
  license attribution reasons; this command is the canonical way to
  fetch it. Verifies file format (SQLite magic, ZIP magic) to catch
  404 HTML pages.
- `solokit.data` module with `CorpusSpec`, `download_wjazzd()`,
  `download_omnibook()`, `download_all()`, `data_status()`.
- `solokit.cli_download` subcommand with `--status`, `--force`,
  `--data-dir`, and `--corpus` filtering.
- 12 new tests in `tests/test_data.py` (total: 119 passing).

### Changed

- Bumped version 0.1.0 → 0.2.0.
- `data_dir` default in `download-corpora` now uses the current
  working directory (was previously the install location of the
  package, which broke for PyPI installs).
- `data_status()` checks file format via magic bytes, not file
  size — smaller test files no longer trigger false positives.

### For PyPI publishers

```bash
# 1. Test on TestPyPI first
.venv/bin/python -m build
.venv/bin/twine upload --repository testpypi dist/*

# 2. After verifying, publish to PyPI
.venv/bin/twine upload dist/*
```

[0.2.0]: https://github.com/kajica2/solokit/releases/tag/v0.2.0

## [0.1.0] - 2026-07-17

### Added

- **Core data model** (`solokit.core`): `Solo`, `Phrase`, `Transcription`,
  `NoteEvent` as frozen dataclasses. Decoupled from music21.
- **Pattern extraction & search** (`solokit.patterns`):
  - 4 transformations: `pitch`, `interval`, `fuzzyinterval`, `cdpcx`
  - `NGramExtractor` with caching
  - `search_patterns()` with normalized Levenshtein similarity,
    length tolerance, min-frequency
- **Feature machine** (`solokit.features`): YAML-driven feature
  extraction, importlib-based function dispatch. Initial pitch +
  rhythm features.
- **Three corpora**:
  - `OmnibookCorpus` — 50 Charlie Parker solos in MusicXML
  - `WJAZDCorpus` — 456 hand-transcribed jazz solos in SQLite
  - `DTLCorpus` — remote client for the Dig That Lick API
- **Audio transcription** (`solokit.audio`):
  - `pYIN` backend (librosa) — fast, monophonic-correct
  - `basic-pitch` backend (Spotify CNN) — polyphonic with monophonic
    post-processing
  - Held-note merge to fix pYIN's vibrato segmentation artifact
- **FastAPI server** with 10 routes (`/search`, `/transcribe`,
  `/features`, `/corpora`, `/healthz`, etc.)
- **Click CLI** with `search`, `features`, `transcribe`, `serve`,
  `health` subcommands
- **89 tests** covering transformations, n-grams, similarity,
  features, API schemas, audio pipeline (synth + transcribe +
  search), Omnibook corpus, WJAZD corpus

### Known issues

- DTL backend is currently broken (JSONDecodeError on every search
  request as of 2026-07-17). The local WJAZD and Omnibook corpora
  work independently. See `data/ATTRIBUTION.md` for licenses.
- Audio deps are heavy (~1GB for PyTorch + basic-pitch). Install
  with `pip install 'solokit[audio]'` only if you need transcription.
- The `cdpcx` transformation in `cdpcx()` does not yet support
  full quality vocabulary (maj7 / m7 / 7 / m7b5 / dim7 / alt).

[0.1.0]: https://github.com/kajica2/solokit/releases/tag/v0.1.0
