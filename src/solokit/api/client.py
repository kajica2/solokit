"""HTTP client for the solokit API.

Use this when you want to talk to a remote solokit server from Python
without going through the CLI.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from solokit.api.schemas import (
    FeatureResponse,
    SearchRequest,
    SearchResponse,
    TranscribeResponse,
)


class SolokitClient:
    """Synchronous client for the solokit API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        *,
        timeout: float = 60.0,
        api_key: str | None = None,
    ) -> None:
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(base_url=base_url, headers=headers, timeout=timeout)
        self.base_url = base_url

    def __enter__(self) -> SolokitClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        resp = self._client.get("/healthz")
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        pattern: Sequence[int],
        transformation: str = "interval",
        databases: Sequence[str] = ("dtl",),
        min_similarity: float = 0.8,
        max_length_difference: int = 0,
        max_edit_distance: int | None = None,
        min_frequency: int = 1,
        limit: int = 50,
    ) -> SearchResponse:
        """Search a corpus via the API. Returns parsed SearchResponse."""
        req = SearchRequest(
            pattern=list(pattern),
            transformation=transformation,  # type: ignore[arg-type]
            databases=list(databases),  # type: ignore[arg-type]
            min_similarity=min_similarity,
            max_length_difference=max_length_difference,
            max_edit_distance=max_edit_distance,
            min_frequency=min_frequency,
            limit=limit,
        )
        resp = self._client.post("/search", json=req.model_dump())
        resp.raise_for_status()
        return SearchResponse.model_validate(resp.json())

    # ------------------------------------------------------------------
    # Transcribe
    # ------------------------------------------------------------------

    def transcribe(self, audio_path: str | Path) -> TranscribeResponse:
        """Upload an audio file for transcription."""
        p = Path(audio_path)
        with p.open("rb") as f:
            resp = self._client.post(
                "/transcribe",
                files={"file": (p.name, f, "audio/wav")},
            )
        resp.raise_for_status()
        return TranscribeResponse.model_validate(resp.json())

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    def features(self, transcription_dict: dict[str, Any], config: str = "basic") -> FeatureResponse:
        """Run a feature extraction over a transcription dict."""
        req = {"notes": transcription_dict.get("notes", []), "config": config, **transcription_dict}
        resp = self._client.post("/features", json=req)
        resp.raise_for_status()
        return FeatureResponse.model_validate(resp.json())
