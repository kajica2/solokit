"""Tests for the data downloader.

We don't actually hit the network in tests (slow, flaky). Instead
we monkeypatch `urllib.request.urlopen` and `shutil.copyfileobj` to
return fake bytes, and verify the right files get created.
"""

from __future__ import annotations

import sqlite3
import tempfile
import zipfile
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest

from solokit.data import (
    CORPORA,
    data_status,
    download_all,
    download_omnibook,
    download_wjazzd,
)


# A 1KB fake SQLite file (just the magic header — enough to pass our
# verify() check). The real wjazzd.db is 41MB and we don't want to
# download it in tests.
FAKE_SQLITE_HEADER = b"SQLite format 3\x00" + b"\x00" * (4096 - 16)


def make_fake_omnibook_zip() -> bytes:
    """Build a minimal Omnibook-shaped zip in memory."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        # The real zip has a "Omnibook xml/" prefix and a __MACOSX/ folder
        zf.writestr("Omnibook xml/._.DS_Store", b"junk")
        zf.writestr("Omnibook xml/Anthropology.xml", "<xml>fake</xml>")
        zf.writestr("Omnibook xml/Au_Private_1.xml", "<xml>fake</xml>")
    return buf.getvalue()


class MockResponse:
    """Minimal mock of urllib.response.addinfourl for urlopen().

    Tracks read position so copyfileobj() can drain the data correctly.
    """

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0
        self.length = len(data)

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            chunk = self._data[self._pos :]
            self._pos = len(self._data)
            return chunk
        chunk = self._data[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk

    def __enter__(self) -> "MockResponse":
        return self

    def __exit__(self, *args: object) -> None:
        pass


@pytest.fixture
def temp_data_dir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


class TestDataStatus:
    def test_reports_missing_corpora(self, temp_data_dir: Path) -> None:
        status = data_status(temp_data_dir)
        assert status["wjazzd"]["present"] is False
        assert status["omnibook"]["present"] is False

    def test_reports_present_wjazzd(self, temp_data_dir: Path) -> None:
        (temp_data_dir / "wjazzd.db").write_bytes(FAKE_SQLITE_HEADER)
        status = data_status(temp_data_dir)
        assert status["wjazzd"]["present"] is True
        assert status["wjazzd"]["size_bytes"] == len(FAKE_SQLITE_HEADER)

    def test_reports_present_omnibook(self, temp_data_dir: Path) -> None:
        d = temp_data_dir / "omnibook"
        d.mkdir()
        (d / "Anthropology.xml").write_text("<x/>")
        (d / "Au_Private_1.xml").write_text("<x/>")
        status = data_status(temp_data_dir)
        assert status["omnibook"]["present"] is True
        assert status["omnibook"]["file_count"] == 2


class TestDownloadWjazzd:
    def test_skip_if_present(self, temp_data_dir: Path) -> None:
        existing = temp_data_dir / "wjazzd.db"
        existing.write_bytes(b"already here")
        result = download_wjazzd(temp_data_dir)
        assert result == existing
        assert existing.read_bytes() == b"already here"  # not touched

    def test_force_redownloads(self, temp_data_dir: Path) -> None:
        existing = temp_data_dir / "wjazzd.db"
        existing.write_bytes(b"old data")
        with patch("solokit.data._download_with_progress") as mock_dl:
            mock_dl.return_value = None
            # Mock the verify step by writing a valid SQLite file
            def fake_dl(url, dest):
                dest.write_bytes(FAKE_SQLITE_HEADER)
            mock_dl.side_effect = fake_dl
            download_wjazzd(temp_data_dir, force=True)
        assert existing.read_bytes() == FAKE_SQLITE_HEADER

    def test_rejects_non_sqlite(self, temp_data_dir: Path) -> None:
        with patch("solokit.data._download_with_progress") as mock_dl:
            # Simulate a download that wrote a 1KB non-SQLite file
            def fake_dl(url, dest):
                dest.write_bytes(b"X" * 1024 + b"definitely not sqlite")
            mock_dl.side_effect = fake_dl
            with pytest.raises(RuntimeError, match="not a SQLite"):
                download_wjazzd(temp_data_dir)
        # The corrupt file is left on disk (we don't delete it — the user
        # might want to inspect). But verify raised so the function failed.


class TestDownloadOmnibook:
    def test_skip_if_present(self, temp_data_dir: Path) -> None:
        d = temp_data_dir / "omnibook"
        d.mkdir()
        (d / "Anthropology.xml").write_text("<x/>")
        result = download_omnibook(temp_data_dir)
        assert result == d

    def test_force_redownloads(self, temp_data_dir: Path) -> None:
        d = temp_data_dir / "omnibook"
        d.mkdir()
        (d / "old.xml").write_text("stale")
        zip_bytes = make_fake_omnibook_zip()

        # Patch urlopen to return our fake zip
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MockResponse(zip_bytes)
            download_omnibook(temp_data_dir, force=True)

        # Old file gone, new files present
        assert not (d / "old.xml").exists()
        assert (d / "Anthropology.xml").exists()
        assert (d / "Au_Private_1.xml").exists()
        # __MACOSX/ and .DS_Store should NOT be extracted
        assert not (d / "._.DS_Store").exists()


class TestDownloadAll:
    def test_downloads_all_by_default(self, temp_data_dir: Path) -> None:
        with (
            patch("urllib.request.urlopen") as mock_urlopen,
            patch("solokit.data._download_with_progress") as mock_dl,
        ):
            zip_bytes = make_fake_omnibook_zip()

            def fake_dl(url, dest):
                if url.endswith(".zip"):
                    dest.write_bytes(zip_bytes)
                else:
                    dest.write_bytes(FAKE_SQLITE_HEADER)
            mock_dl.side_effect = fake_dl

            results = download_all(temp_data_dir)
            assert "wjazzd" in results
            assert "omnibook" in results

    def test_only_specified(self, temp_data_dir: Path) -> None:
        with (
            patch("urllib.request.urlopen") as mock_urlopen,
            patch("solokit.data._download_with_progress") as mock_dl,
        ):
            mock_dl.side_effect = lambda url, dest: dest.write_bytes(FAKE_SQLITE_HEADER)
            results = download_all(temp_data_dir, only=["wjazzd"])
            assert "wjazzd" in results
            assert "omnibook" not in results

    def test_unknown_corpus_raises(self, temp_data_dir: Path) -> None:
        with pytest.raises(ValueError, match="Unknown corpora"):
            download_all(temp_data_dir, only=["fake_corpus"])


class TestCorpusSpecIntegrity:
    def test_all_corpora_have_required_fields(self) -> None:
        for name, spec in CORPORA.items():
            assert spec.name == name
            assert spec.url.startswith("https://")
            assert spec.display_name
            assert spec.target_path
            assert spec.expected_size_bytes > 0
