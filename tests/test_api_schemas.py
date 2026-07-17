"""Tests for the API Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from solokit.api.schemas import SearchRequest, TranscribeRequest


class TestSearchRequest:
    def test_minimal_valid(self) -> None:
        req = SearchRequest(pattern=[-1, -1, 4, -5, -2])
        assert req.transformation == "interval"
        assert req.min_similarity == 0.8
        assert req.databases == ["dtl"]  # Pydantic v2 normalizes tuple → list

    def test_pattern_too_short(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest(pattern=[1])

    def test_pattern_too_long(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest(pattern=list(range(25)))

    def test_similarity_bounds(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest(pattern=[1, 2, 3], min_similarity=0.4)
        with pytest.raises(ValidationError):
            SearchRequest(pattern=[1, 2, 3], min_similarity=1.5)

    def test_at_least_one_database(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest(pattern=[1, 2, 3], databases=[])

    def test_roundtrip_json(self) -> None:
        req = SearchRequest(
            pattern=[-1, -1, 4, -5, -2],
            transformation="interval",
            databases=["dtl", "omnibook"],
            min_similarity=0.7,
        )
        data = req.model_dump()
        restored = SearchRequest.model_validate(data)
        assert restored == req


class TestTranscribeRequest:
    def test_defaults(self) -> None:
        req = TranscribeRequest()
        assert req.onset_threshold == 0.5
        assert req.min_note_length_ms == 50.0
