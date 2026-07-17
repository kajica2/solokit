"""End-to-end tests for the FastAPI server.

These hit the real app via FastAPI's TestClient (no uvicorn needed).
For /search we patch the DTL corpus so the test never touches the network.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from solokit.api.server import AVAILABLE_CORPORA, create_app
from solokit.corpora.base import SearchResult
from solokit.patterns.similarity import Match


@pytest.fixture
def app():
    """Build a fresh app. DTLCorpus is patched in lifespan to avoid network."""
    return create_app()


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


# ----------------------------------------------------------------------------
# Health & metadata
# ----------------------------------------------------------------------------


class TestHealthAndMetadata:
    def test_healthz_returns_ok(self, client: TestClient) -> None:
        r = client.get("/healthz")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert set(data["corpora"]) == set(AVAILABLE_CORPORA)

    def test_corpora_lists_all_three(self, client: TestClient) -> None:
        r = client.get("/corpora")
        assert r.status_code == 200
        data = r.json()
        assert "dtl" in data["corpora"]
        assert "omnibook" in data["corpora"]
        assert "wjazzd" in data["corpora"]
        assert "default" in data

    def test_cors_preflight_allows_post(self, client: TestClient) -> None:
        r = client.options(
            "/search",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert r.status_code == 200
        assert "access-control-allow-origin" in {k.lower() for k in r.headers.keys()}
        assert r.headers["access-control-allow-origin"] == "http://localhost:5173"

    def test_cors_preflight_allows_localhost_8000(self, client: TestClient) -> None:
        r = client.options(
            "/search",
            headers={
                "Origin": "http://localhost:8000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert r.status_code == 200


# ----------------------------------------------------------------------------
# Frontend
# ----------------------------------------------------------------------------


class TestFrontend:
    def test_index_serves_html(self, client: TestClient) -> None:
        r = client.get("/")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")
        assert "solokit" in r.text.lower()

    def test_static_css_served(self, client: TestClient) -> None:
        r = client.get("/static/css/styles.css")
        assert r.status_code == 200
        assert "text/css" in r.headers["content-type"]
        assert ":root" in r.text  # the CSS variable block

    def test_static_js_served(self, client: TestClient) -> None:
        r = client.get("/static/js/app.js")
        assert r.status_code == 200
        assert "javascript" in r.headers["content-type"]
        assert "fetch" in r.text


# ----------------------------------------------------------------------------
# Pattern search
# ----------------------------------------------------------------------------


def _fake_match(similarity: float, edit_distance: int) -> Match:
    ngram = MagicMock()
    ngram.source_id = "fake"
    ngram.values = [-1, -1, 4, -5, -2]
    return Match(
        source=ngram,
        edit_distance=edit_distance,
        similarity=similarity,
    )


def _fake_result(melid: str, performer: str, title: str, database: str, sim: float, ed: int) -> SearchResult:
    return SearchResult(
        match=_fake_match(sim, ed),
        performer=performer,
        title=title,
        year=1990,
        melid=melid,
        audio_url=None,
        database=database,
        instrument="sax",
        start_position=0.0,
        duration=2.0,
    )


class TestSearchEndpoint:
    """The /search endpoint dispatches to per-corpus backends and aggregates.

    The corpus classes are imported into the server module's namespace, so
    we patch `solokit.api.server.WJAZDCorpus` etc. (NOT solokit.corpora.wjazzd
    which only patches the original definition site).
    """

    def test_search_wjazzd_only(self, client: TestClient) -> None:
        with patch("solokit.api.server.WJAZDCorpus") as MockWJ:
            instance = MockWJ.return_value
            instance.search.return_value = [
                _fake_result("w-1", "Bird", "Anthropology", "wjazzd", 0.95, 1),
                _fake_result("w-2", "Dizzy", "Groovin' High", "wjazzd", 0.88, 2),
            ]
            r = client.post(
                "/search",
                json={
                    "pattern": [-1, -1, 4, -5, -2],
                    "transformation": "interval",
                    "databases": ["wjazzd"],
                    "min_similarity": 0.5,
                    "limit": 10,
                },
            )
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 2
        assert data["matches"][0]["performer"] == "Bird"
        assert data["matches"][0]["database"] == "wjazzd"
        assert data["errors"] is None

    def test_search_omnibook_only(self, client: TestClient) -> None:
        with patch("solokit.api.server.OmnibookCorpus") as MockOmni:
            instance = MockOmni.return_value
            instance.search.return_value = [
                _fake_result("omni-1", "Charlie Parker", "Confirmation", "omnibook", 1.0, 0),
            ]
            r = client.post(
                "/search",
                json={
                    "pattern": [-1, -1, 4, -5, -2],
                    "transformation": "interval",
                    "databases": ["omnibook"],
                },
            )
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 1
        assert data["matches"][0]["performer"] == "Charlie Parker"

    def test_search_aggregates_multiple_corpora(self, client: TestClient) -> None:
        with patch("solokit.api.server.WJAZDCorpus") as MockWJ, \
             patch("solokit.api.server.OmnibookCorpus") as MockOmni:
            MockWJ.return_value.search.return_value = [
                _fake_result("w-1", "Bird", "Anthropology", "wjazzd", 0.85, 2),
            ]
            MockOmni.return_value.search.return_value = [
                _fake_result("omni-1", "Charlie Parker", "Au Private", "omnibook", 0.95, 1),
            ]
            r = client.post(
                "/search",
                json={
                    "pattern": [-1, -1, 4, -5, -2],
                    "transformation": "interval",
                    "databases": ["wjazzd", "omnibook"],
                },
            )
        data = r.json()
        assert data["total"] == 2
        # Higher similarity first
        assert data["matches"][0]["similarity"] == 0.95
        assert data["matches"][1]["similarity"] == 0.85
        assert data["errors"] is None

    def test_search_continues_when_one_corpus_fails(self, client: TestClient) -> None:
        """If DTL is down, the local corpora should still return results."""
        with patch("solokit.api.server.WJAZDCorpus") as MockWJ, \
             patch.object(client.app.state, "dtl") as MockDTLState:
            MockDTLState.search.side_effect = RuntimeError("DTL is down")
            MockWJ.return_value.search.return_value = [
                _fake_result("w-1", "Bird", "Anthropology", "wjazzd", 0.9, 1),
            ]
            r = client.post(
                "/search",
                json={
                    "pattern": [-1, -1, 4, -5, -2],
                    "databases": ["dtl", "wjazzd"],
                },
            )
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 1
        assert "dtl" in data["errors"]
        assert "DTL is down" in data["errors"]["dtl"]

    def test_search_rejects_short_pattern(self, client: TestClient) -> None:
        r = client.post(
            "/search",
            json={
                "pattern": [60],  # too short
                "databases": ["wjazzd"],
            },
        )
        assert r.status_code == 422  # Pydantic validation error

    def test_search_rejects_no_databases(self, client: TestClient) -> None:
        r = client.post(
            "/search",
            json={"pattern": [-1, -1], "databases": []},
        )
        assert r.status_code == 422

    def test_search_respects_limit(self, client: TestClient) -> None:
        with patch("solokit.api.server.WJAZDCorpus") as MockWJ:
            MockWJ.return_value.search.return_value = [
                _fake_result(f"w-{i}", f"Performer {i}", f"Title {i}", "wjazzd", 0.9 - i * 0.01, i)
                for i in range(20)
            ]
            r = client.post(
                "/search",
                json={
                    "pattern": [-1, -1, 4, -5, -2],
                    "databases": ["wjazzd"],
                    "limit": 5,
                },
            )
        data = r.json()
        assert data["total"] == 5


# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------


class TestFeaturesEndpoint:
    def test_features_basic(self, client: TestClient) -> None:
        r = client.post(
            "/features",
            json={
                "config": "basic",
                "tempo_bpm": 120,
                "notes": [
                    {"pitch": 60, "onset_beat": 0, "duration_beats": 1.0},
                    {"pitch": 62, "onset_beat": 1, "duration_beats": 1.0},
                    {"pitch": 64, "onset_beat": 2, "duration_beats": 1.0},
                    {"pitch": 65, "onset_beat": 3, "duration_beats": 1.0},
                    {"pitch": 67, "onset_beat": 4, "duration_beats": 1.0},
                ],
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["config"] == "basic"
        assert "pitch_range" in data["features"]
        assert "pitch_class_histogram" in data["features"]

    def test_features_unknown_config_returns_404(self, client: TestClient) -> None:
        r = client.post(
            "/features",
            json={
                "config": "this-does-not-exist",
                "notes": [{"pitch": 60, "onset_beat": 0, "duration_beats": 1}],
            },
        )
        assert r.status_code == 404


# ----------------------------------------------------------------------------
# Transcribe
# ----------------------------------------------------------------------------


class TestTranscribeEndpoint:
    def test_transcribe_rejects_non_wav(self, client: TestClient) -> None:
        """A non-audio file should get a 400, not an uncaught 500."""
        from io import BytesIO

        r = client.post(
            "/transcribe",
            files={"file": ("test.txt", BytesIO(b"not a wav"), "text/plain")},
        )
        # We accept 200 (unlikely with a non-wav), 400 (rejected), 422 (validation),
        # or 501 (audio deps missing). The bug we're fixing is an uncaught 500.
        assert r.status_code in (200, 400, 422, 501)
        if r.status_code == 400:
            assert "Could not transcribe" in r.json()["detail"] or "transcribe" in r.json()["detail"].lower()

    @pytest.mark.slow
    def test_transcribe_runs_when_audio_deps_present(self, client: TestClient, tmp_path) -> None:
        """End-to-end smoke: upload a tiny wav, get back note events."""
        import numpy as np
        import soundfile as sf

        sr = 22050
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        y = 0.3 * np.sin(2 * np.pi * 440 * t)
        wav_path = tmp_path / "tone.wav"
        sf.write(str(wav_path), y, sr)

        r = client.post(
            "/transcribe",
            files={"file": ("tone.wav", wav_path.read_bytes(), "audio/wav")},
        )
        # The transcribe path is finicky (depends on audio backend heuristics);
        # 200 with notes is the happy path, 400 if our 0.5s tone is too short
        # for pYIN to find a stable F0. Either is fine for this smoke test.
        assert r.status_code in (200, 400), f"got {r.status_code}: {r.text}"
        if r.status_code == 200:
            data = r.json()
            assert "notes" in data
            assert "tempo_bpm" in data
