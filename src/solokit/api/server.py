"""FastAPI server.

Endpoints:
- GET  /              — solokit frontend (single-page app)
- GET  /static/*      — frontend static assets (CSS, JS)
- GET  /healthz       — liveness check
- GET  /corpora       — list available corpora
- POST /search        — pattern search (DTL-compatible)
- POST /transcribe    — audio → transcription (requires [audio] extra)
- POST /features      — extract features from a transcription

Run with:  uvicorn solokit.api.server:app --reload
Or via CLI:  solokit serve
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from solokit import __version__
from solokit.api.schemas import (
    FeatureRequest,
    FeatureResponse,
    HealthResponse,
    MatchResult,
    SearchRequest,
    SearchResponse,
    TranscribeResponse,
)
from solokit.corpora import DTLCorpus, OmnibookCorpus, WJAZDCorpus
from solokit.corpora.base import SearchResult
from solokit.patterns.ngram import NGramExtractor
from solokit.patterns.similarity import search_patterns


# All corpora that ship in the solokit package and can be referenced by name.
# Keep this in sync with what's exported from solokit.corpora.
AVAILABLE_CORPORA: tuple[str, ...] = ("dtl", "omnibook", "wjazzd")

# Where the bundled frontend lives relative to this file.
# Layout: src/solokit/api/server.py  →  ../../../frontend/  (i.e. project_root/frontend/)
_FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize a DTL corpus client at startup, close on shutdown."""
    app.state.dtl = DTLCorpus()
    yield
    app.state.dtl.close()


def _cors_origins() -> list[str]:
    """Resolve the list of allowed CORS origins.

    Defaults to permissive (localhost-only) for local development.
    Override with the SOLOKIT_CORS_ORIGINS env var (comma-separated).
    """
    raw = os.environ.get("SOLOKIT_CORS_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return [
        "http://localhost",
        "http://localhost:8000",
        "http://127.0.0.1",
        "http://127.0.0.1:8000",
        "http://localhost:3000",  # common dev frontend port
        "http://localhost:5173",  # vite default
    ]


def create_app() -> FastAPI:
    """Build the FastAPI app. Allows custom configuration in tests."""
    app = FastAPI(
        title="solokit API",
        version=__version__,
        description="Pattern search and analysis for jazz solo transcriptions.",
        lifespan=lifespan,
    )

    # CORS — permissive for local dev, configurable via env for production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Frontend (served at /)
    # ------------------------------------------------------------------

    if _FRONTEND_DIR.exists():
        static_dir = _FRONTEND_DIR / "static"
        if static_dir.exists():
            app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        async def index() -> FileResponse:
            index_file = _FRONTEND_DIR / "index.html"
            if not index_file.exists():
                return FileResponse(  # type: ignore[return-value]
                    _fallback_index_path(),
                    media_type="text/html",
                )
            return FileResponse(str(index_file), media_type="text/html")  # type: ignore[return-value]
    else:
        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        async def index() -> str:
            return _FALLBACK_INDEX_HTML

    # ------------------------------------------------------------------
    # Health & metadata
    # ------------------------------------------------------------------

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            corpora=list(AVAILABLE_CORPORA),
        )

    @app.get("/corpora")
    async def list_corpora() -> dict[str, list[str] | str]:
        return {
            "corpora": list(AVAILABLE_CORPORA),
            "default": "wjazzd",
        }

    # ------------------------------------------------------------------
    # Pattern search
    # ------------------------------------------------------------------

    @app.post("/search", response_model=SearchResponse)
    async def search(req: SearchRequest) -> SearchResponse:
        """Dispatch a pattern search to each requested corpus and aggregate.

        Each corpus (dtl, omnibook, wjazzd) has its own concrete search backend.
        Local corpora (omnibook, wjazzd) are instant and offline; dtl is the
        remote DTL HTTP API and may fail. We run them in series and merge,
        filtering duplicates by (database, melid) and sorting by similarity
        descending.
        """
        t0 = time.perf_counter()

        aggregated: dict[tuple[str, str], SearchResult] = {}
        errors: dict[str, str] = {}

        for db in req.databases:
            try:
                if db == "dtl":
                    dtl: DTLCorpus = app.state.dtl
                    res = dtl.search(
                        req.pattern,
                        transformation=req.transformation,
                        databases=("dtl",),
                        min_similarity=req.min_similarity,
                        max_length_difference=req.max_length_difference,
                        max_edit_distance=req.max_edit_distance,
                        min_frequency=req.min_frequency,
                        limit=req.limit,
                    )
                elif db == "wjazzd":
                    corpus = WJAZDCorpus()
                    res = corpus.search(
                        req.pattern,
                        transformation=req.transformation,
                        min_similarity=req.min_similarity,
                        max_length_difference=req.max_length_difference,
                        max_edit_distance=req.max_edit_distance,
                        min_frequency=req.min_frequency,
                        limit=req.limit,
                    )
                elif db == "omnibook":
                    corpus = OmnibookCorpus()
                    res = corpus.search(
                        req.pattern,
                        transformation=req.transformation,
                        min_similarity=req.min_similarity,
                        max_length_difference=req.max_length_difference,
                        max_edit_distance=req.max_edit_distance,
                        min_frequency=req.min_frequency,
                        limit=req.limit,
                    )
                else:
                    errors[db] = f"Unknown corpus: {db}"
                    continue
                for r in res:
                    key = (r.database, r.melid)
                    # If duplicate, keep the one with higher similarity
                    if key not in aggregated or r.match.similarity > aggregated[key].match.similarity:
                        aggregated[key] = r
            except Exception as exc:
                errors[db] = f"{type(exc).__name__}: {exc}"
                # Continue with other corpora; partial results are still useful

        results = sorted(
            aggregated.values(),
            key=lambda r: (-r.match.similarity, r.match.edit_distance, r.year or 0),
        )[: req.limit]

        matches = [
            MatchResult(
                melid=r.melid,
                performer=r.performer,
                title=r.title,
                year=r.year,
                database=r.database,
                instrument=r.instrument,
                similarity=r.match.similarity,
                edit_distance=r.match.edit_distance,
                start_position=r.start_position,
                duration=r.duration,
                audio_url=r.audio_url,
            )
            for r in results
        ]
        took_ms = (time.perf_counter() - t0) * 1000
        return SearchResponse(
            query=req,
            matches=matches,
            total=len(matches),
            took_ms=took_ms,
            errors=errors or None,  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------
    # Audio transcription
    # ------------------------------------------------------------------

    @app.post("/transcribe", response_model=TranscribeResponse)
    async def transcribe(
        file: UploadFile = File(...),
        onset_threshold: float = Form(0.5),
        frame_threshold: float = Form(0.3),
        min_note_length_ms: float = Form(50.0),
    ) -> TranscribeResponse:
        try:
            from solokit.audio import transcribe_wav
        except ImportError as exc:
            raise HTTPException(
                status_code=501,
                detail="Audio transcription requires `pip install 'solokit[audio]'`",
            ) from exc

        # Save the upload to a temp file
        import tempfile

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.filename or "audio.wav").suffix
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            t = transcribe_wav(
                tmp_path,
                onset_threshold=onset_threshold,
                frame_threshold=frame_threshold,
                min_note_length_ms=min_note_length_ms,
            )
        except Exception as exc:  # noqa: BLE001 — transcribe raises many backend-specific types
            # Catching broadly is correct here: user-uploaded audio is untrusted,
            # and the audio backends (soundfile, audioread, librosa) raise a zoo
            # of unrelated exception types (LibsndfileError, NoBackendError,
            # ZeroDivisionError, etc.) for bad inputs.
            raise HTTPException(
                status_code=400,
                detail=f"Could not transcribe audio: {type(exc).__name__}: {exc}",
            ) from exc
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        from solokit.api.schemas import NoteEventDTO

        return TranscribeResponse(
            notes=[
                NoteEventDTO(
                    pitch=n.pitch,
                    onset_beat=n.onset_beat,
                    duration_beats=n.duration_beats,
                    velocity=n.velocity,
                )
                for n in t.notes
            ],
            tempo_bpm=t.tempo_bpm,
            time_signature=t.time_signature,
            key_signature=t.key_signature,
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @app.post("/features", response_model=FeatureResponse)
    async def features(req: FeatureRequest) -> FeatureResponse:
        from solokit.core.solo import Solo, SoloMetadata
        from solokit.core.transcription import NoteEvent, Transcription
        from solokit.features import FeatureMachine

        notes = tuple(
            NoteEvent(
                pitch=n.pitch,
                onset_beat=n.onset_beat,
                duration_beats=n.duration_beats,
                velocity=n.velocity,
            )
            for n in req.notes
        )
        transcription = Transcription.from_note_sequence(
            notes, tempo_bpm=req.tempo_bpm
        )
        solo = Solo(
            metadata=SoloMetadata(
                melid="ad-hoc",
                title="ad-hoc",
                performer="ad-hoc",
            ),
            transcription=transcription,
        )

        config_path = _config_path(req.config)
        if config_path is None:
            raise HTTPException(status_code=404, detail=f"Unknown config: {req.config}")
        machine = FeatureMachine.from_yaml(config_path)
        results = machine.extract(solo)

        # numpy arrays → lists for JSON serialization
        serializable: dict[str, float | list[float] | dict[str, float]] = {}
        import numpy as np

        for k, v in results.items():
            if isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            elif isinstance(v, (int, float, str, list, dict)):
                serializable[k] = v
            else:
                serializable[k] = str(v)

        return FeatureResponse(features=serializable, config=req.config)

    return app


# Fallback HTML served when the bundled frontend/ dir is missing.
# Lets `pip install solokit` users still get a landing page that points at the API docs.
_FALLBACK_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>solokit API</title>
    <style>
        body { font-family: -apple-system, system-ui, sans-serif; max-width: 720px;
               margin: 60px auto; padding: 0 20px; color: #2c3e50; }
        h1 { color: #2c3e50; }
        code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }
        pre { background: #f4f4f4; padding: 12px; border-radius: 6px; overflow-x: auto; }
    </style>
</head>
<body>
    <h1>solokit API</h1>
    <p>Pattern search and analysis for jazz solo transcriptions.</p>
    <p>See <a href="/docs">/docs</a> for the OpenAPI schema.</p>
    <h2>Try it</h2>
    <pre>curl -X POST http://localhost:8000/search \\
  -H 'Content-Type: application/json' \\
  -d '{"pattern": [-1, -1, 4, -5, -2], "transformation": "interval", "min_similarity": 0.8}'</pre>
    <p style="margin-top: 32px; color: #888; font-size: 0.9em;">
        The bundled frontend is not installed in this environment. To enable the UI,
        clone the repo and run <code>solokit serve</code> from the project root.
    </p>
</body>
</html>
"""


def _fallback_index_path() -> str:
    """Return a writable temp file path containing the fallback HTML.

    Used when the bundled frontend/ dir is missing but FastAPI still wants
    a FileResponse for the index.
    """
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".html")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(_FALLBACK_INDEX_HTML)
    return path


def _config_path(name: str):
    """Resolve a feature config name to a YAML file path.

    Looks in this order:
      1. <project_root>/features/<name>.yaml   (dev workflow)
      2. <project_root>/configs/<name>.yaml    (alternative dev location)
    Returns None if neither exists.
    """
    # Project root: three parents up from this file (src/solokit/api/server.py).
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    for sub in ("features", "configs"):
        candidate = project_root / sub / f"{name}.yaml"
        if candidate.exists():
            return str(candidate)
    return None


# Module-level instance for `uvicorn solokit.api.server:app`
app = create_app()
