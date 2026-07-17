"""HTTP API — FastAPI server + Python client.

The server exposes a DTL-compatible /search endpoint plus a few
solokit-specific ones for transcription and feature extraction.
The client is a thin httpx wrapper for programmatic access.
"""

from solokit.api.schemas import (
    FeatureRequest,
    FeatureResponse,
    SearchRequest,
    SearchResponse,
    TranscribeRequest,
    TranscribeResponse,
)
from solokit.api.server import create_app

__all__ = [
    "FeatureRequest",
    "FeatureResponse",
    "SearchRequest",
    "SearchResponse",
    "TranscribeRequest",
    "TranscribeResponse",
    "create_app",
]
