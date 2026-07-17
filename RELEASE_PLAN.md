# Release plan: solokit 0.1.0

## What solokit is

A pattern-search and analysis toolkit for jazz solo transcriptions, modernizing
the jazzomat research stack (MeloSpyLib / pymus) on top of music21 and
compatible with the Dig That Lick pattern-search API.

**Three local corpora shipped**: WJAZD (456 solos), Omnibook (50 Charlie
Parker solos), and a remote DTL client for fallback.

## Goals for 0.1.0

- [x] Public GitHub release under MIT
- [x] 89 tests passing locally
- [x] README with quickstart + corpus table
- [ ] GitHub Actions CI on push
- [ ] PyPI release (optional)
- [ ] Hugging Face Space or Fly.io demo deployment (optional)

## Why online

- **Discoverability** — researchers searching for "jazz pattern search" or
  "MeloSpyLib python" should find this.
- **Reproducibility** — the corpus loading code + bundled data means
  anyone can `pip install solokit` and search the same 506 solos locally.
- **Collaborators** — once public, others (e.g. the jazzomat team) can
  file issues or send PRs.
- **Citation** — having a citable artifact (DOI via Zenodo) is the
  currency of academic work.

## Phases

### Phase 1: Code polish (now)

Pre-release hygiene that the public repo will see.

1. **LICENSE** (MIT for code) — `LICENSE` file at project root.
2. **Data attribution** — `data/ATTRIBUTION.md` listing the Omnibook
   (CC BY-NC-SA 2.0, Déguernel et al. 2016) and WJAZD (ODbL 1.0, jazzomat
   project). Each has a citation requirement.
3. **CITATION.cff** — GitHub's citation format. Includes authors,
   version, DOI placeholder, license.
4. **CHANGELOG.md** — keep-a-Changelog format. 0.1.0 entry summarizes
   what shipped.
5. **GitHub Actions CI** — `.github/workflows/ci.yml` runs pytest on
   Python 3.11 and 3.12. Skips audio-marked tests (those need PyTorch
   and ~1GB install; the smoke test can run on a self-hosted runner
   if we care enough).
6. **Bump to 0.1.0** — already at 0.1.0 in pyproject.toml, but verify
   the README and CLI version match.

### Phase 2: GitHub push

```bash
cd /Users/kajicadjuric/Documents/research/solokit

# Create public repo on GitHub (kajica2 account already auth'd via gh CLI)
gh repo create solokit --public --source=. --description "Pattern search and analysis for jazz solo transcriptions" --push
```

The `gh repo create --push` flag creates the repo AND pushes the local
git history in one step. After that, the repo is at
`https://github.com/kajica2/solokit`.

Then:
- Add topics: `music`, `jazz`, `music-information-retrieval`,
  `pattern-search`, `transcription`, `python`, `pypi`
- Enable GitHub Pages for docs (optional)
- Add a `SECURITY.md` for vulnerability reporting

### Phase 3: PyPI (optional, do later)

Make `pip install solokit` work for end users.

```bash
# Build sdist + wheel
.venv/bin/python -m pip install build twine
.venv/bin/python -m build

# Test on TestPyPI first
.venv/bin/twine upload --repository testpypi dist/*

# Then to production PyPI
.venv/bin/twine upload dist/*
```

Caveats:
- PyPI sdist/wheel must NOT include the 41MB wjazzd.db. Add it to
  `MANIFEST.in` exclude and have a one-time download script
  (`solokit download-corpora`) that fetches it on first use.
- The bundled omnibook/ MusicXML (10MB) is also large. Same
  treatment.

### Phase 4: Live demo (optional, do much later)

A live API that someone can curl. Cheapest options:

| Host | Free tier | Setup time |
|---|---|---|
| Hugging Face Spaces | Yes, FastAPI supported | 15 min |
| Fly.io | Yes (with credit card) | 30 min |
| Render | Yes | 20 min |

Hugging Face Spaces is the easiest — drop a `Dockerfile` + `app.py`,
push to a Space, done.

For the demo, ship a 2-page HTML: search box + visualization.

## Out of scope for 0.1.0

- Audio transcription in production (basic-pitch integration works
  but pYIN is the recommended backend; need to make this clearer in
  the docs)
- EsAC corpus (6000 folk songs) — would make 0.2.0
- MIDI / MusicXML ingestion for local files
- HTML frontend
- Docker / docker-compose

## Risks

1. **DTL is currently broken** (JSONDecodeError as of 2026-07-17). The
   DTL corpus client is a useful feature but the corpus itself is
   unreliable. The local WJAZD + Omnibook corpora are the stable path
   for now. README should make this clear.

2. **Audio deps are heavy** (~1GB for PyTorch + basic-pitch). The
   `[audio]` extra is opt-in. Document the install size.

3. **No DOI yet.** If the user wants academic citability, mint a
   Zenodo DOI on first release. Free, takes 5 minutes.

4. **Repository name collision.** `solokit` might be taken on PyPI.
   Check before publishing. Fallback names: `jazz-pattern-search`,
   `melospy2`, `jazz-o-mat`.

## Next concrete step

Run: `gh repo create solokit --public --source=. --push`

That's the "commit to online" moment.
