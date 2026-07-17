"""Pydantic request/response schemas for the API.

The shapes are intentionally compatible with what a TypeScript / React
client would expect. Use snake_case in Python, camelCase in the JSON
wire format via `Field(alias=...)` if needed.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Transformation = Literal["pitch", "interval", "fuzzyinterval", "cdpcx"]
Database = Literal["dtl", "wjazzd", "omnibook", "esac"]


class SearchRequest(BaseModel):
    """Request body for POST /search."""

    pattern: list[int] = Field(
        ...,
        min_length=2,
        max_length=20,
        description="Pattern values (already transformed).",
    )
    transformation: Transformation = "interval"
    databases: list[Database] = Field(
        default_factory=lambda: ["dtl"],
        description="Which corpora to search.",
    )
    min_similarity: float = Field(default=0.8, ge=0.5, le=1.0)
    max_length_difference: int = Field(default=0, ge=0, le=5)
    max_edit_distance: int | None = Field(default=None, ge=0, le=20)
    min_frequency: int = Field(default=1, ge=1)
    limit: int = Field(default=50, ge=1, le=500)

    @field_validator("databases")
    @classmethod
    def _at_least_one(cls, v: list[str]) -> list[str]:
        if not v:
            msg = "at least one database is required"
            raise ValueError(msg)
        return v


class MatchResult(BaseModel):
    """A single search match."""

    melid: str
    performer: str
    title: str
    year: int | None
    database: str
    instrument: str | None = None
    similarity: float
    edit_distance: int
    start_position: float | None = None
    duration: float | None = None
    audio_url: str | None = None


class SearchResponse(BaseModel):
    """Response body for POST /search."""

    query: SearchRequest
    matches: list[MatchResult]
    total: int
    took_ms: float
    errors: dict[str, str] | None = Field(
        default=None,
        description="Per-corpus errors (e.g. DTL down). Partial results still returned.",
    )


class TranscribeRequest(BaseModel):
    """Request body for POST /transcribe.

    Send as multipart/form-data with the audio file as 'file'.
    """

    onset_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    frame_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    min_note_length_ms: float = Field(default=50.0, ge=0.0)
    min_frequency: float | None = None
    max_frequency: float | None = None


class NoteEventDTO(BaseModel):
    """A note event in the transcription response."""

    pitch: int | None
    onset_beat: float
    duration_beats: float
    velocity: int | None = None


class TranscribeResponse(BaseModel):
    """Response body for POST /transcribe."""

    notes: list[NoteEventDTO]
    tempo_bpm: float
    time_signature: tuple[int, int]
    key_signature: str | None = None


class FeatureRequest(BaseModel):
    """Request body for POST /features.

    Send the transcription (or a path) and a YAML config name.
    """

    notes: list[NoteEventDTO]
    config: str = "basic"  # name under solokit/configs/
    tempo_bpm: float = 120.0


class FeatureResponse(BaseModel):
    """Response body for POST /features."""

    features: dict[str, float | list[float] | dict[str, float]]
    config: str


class HealthResponse(BaseModel):
    """Response body for GET /healthz."""

    status: Literal["ok", "degraded"]
    version: str
    corpora: list[str]
