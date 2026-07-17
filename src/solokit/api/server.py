"""FastAPI server.

Endpoints:
- GET  /              — simple HTML index
- GET  /healthz       — liveness check
- GET  /corpora       — list available corpora
- POST /search        — pattern search (DTL-compatible)
- POST /transcribe    — audio → transcription (requires [audio] extra)
- POST /features      — extract features from a transcription

Run with:  uvicorn solokit.api.server:app --reload
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from solokit import __version__
from solokit.api.schemas import (
    FeatureRequest,
    FeatureResponse,
    HealthResponse,
    MatchResult,
    SearchRequest,
    SearchResponse,
    TranscribeRequest,
    TranscribeResponse,
)
from solokit.corpora import DTLCorpus
from solokit.patterns.ngram import NGramExtractor
from solokit.patterns.similarity import search_patterns


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize a DTL corpus client at startup, close on shutdown."""
    app.state.dtl = DTLCorpus()
    yield
    app.state.dtl.close()


def create_app() -> FastAPI:
    """Build the FastAPI app. Allows custom configuration in tests."""
    app = FastAPI(
        title="solokit API",
        version=__version__,
        description="Pattern search and analysis for jazz solo transcriptions.",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # Health & metadata
    # ------------------------------------------------------------------

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            corpora=["dtl"],
        )

    @app.get("/corpora")
    async def list_corpora() -> dict[str, list[str]]:
        return {"corpora": ["dtl"]}

    # ------------------------------------------------------------------
    # Pattern search
    # ------------------------------------------------------------------

    @app.post("/search", response_model=SearchResponse)
    async def search(req: SearchRequest) -> SearchResponse:
        t0 = time.perf_counter()
        dtl: DTLCorpus = app.state.dtl

        try:
            results = dtl.search(
                req.pattern,
                transformation=req.transformation,
                databases=req.databases,
                min_similarity=req.min_similarity,
                max_length_difference=req.max_length_difference,
                max_edit_distance=req.max_edit_distance,
                min_frequency=req.min_frequency,
                limit=req.limit,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Upstream error: {exc}") from exc

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
        from pathlib import Path

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

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return """
<!DOCTYPE html>
<html>
<head>
    <title>solokit API</title>
    <style>
        body { font-family: -apple-system, system-ui, sans-serif; max-width: 720px; margin: 60px auto; padding: 0 20px; }
        h1 { color: #2c3e50; }
        code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }
        pre { background: #f4f4f4; padding: 12px; border-radius: 6px; overflow-x: auto; }
        a { color: #2c3e50; }
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
</body>
</html>
"""

    return app


def _config_path(name: str):
    """Resolve a config name to a YAML file path."""
    from importlib.resources import files

    try:
        # Python 3.12+ has files() returning Traversable
        cfg = files("solokit").joinpath(f"../../features/{name}.yaml")  # type: ignore[arg-type]
        if cfg.is_file():  # type: ignore[union-attr]
            return str(cfg)  # type: ignore[arg-type]
    except Exception:
        pass
    # Fallback: relative to this file
    from pathlib import Path

    candidate = Path(__file__).parent.parent.parent.parent / "features" / f"{name}.yaml"
    if candidate.exists():
        return str(candidate)
    return None


# Module-level instance for `uvicorn solokit.api.server:app`
app = create_app()
