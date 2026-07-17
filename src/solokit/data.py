"""Local data downloader for the bundled solokit corpora.

The two local corpora (WJAZD SQLite, Omnibook MusicXML) are NOT
included in the git repo or the PyPI sdist — they total ~51MB and
have license attribution requirements. New users run:

    solokit download-corpora
    solokit download-corpora --corpus wjazzd
    solokit download-corpora --corpus omnibook
    solokit download-corpora --force

After download, the data lives in `<solokit>/data/`:

    data/
    ├── wjazzd.db                  # 41MB, ODbL 1.0
    ├── omnibook/                  # 10MB unpacked
    │   ├── Anthropology.xml
    │   ├── Au_Private_1.xml
    │   └── ... (50 files)
    └── ATTRIBUTION.md

Sources (verified 2026-07-17):
- WJAZD: https://jazzomat.hfm-weimar.de/download/downloads/wjazzd.db
- Omnibook: https://homepages.loria.fr/evincent/omnibook/omnibook_xml.zip
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Default download sources. Update CHECKSUMS if these change.
WJAZD_URL = "https://jazzomat.hfm-weimar.de/download/downloads/wjazzd.db"
OMNIBOOK_URL = "https://homepages.loria.fr/evincent/omnibook/omnibook_xml.zip"


@dataclass(frozen=True)
class CorpusSpec:
    name: str
    display_name: str
    url: str
    target_path: Path  # file or directory
    expected_size_bytes: int
    sha256: str | None  # optional; None = skip checksum verification

    @property
    def is_directory(self) -> bool:
        return self.target_path.is_dir() or self.target_path.suffix == ""


# Verified sizes (approximate; from a successful 2026-07-17 download)
CORPORA: dict[str, CorpusSpec] = {
    "wjazzd": CorpusSpec(
        name="wjazzd",
        display_name="Weimar Jazz Database (456 solos, SQLite)",
        url=WJAZD_URL,
        target_path=Path("wjazzd.db"),
        expected_size_bytes=41_000_000,  # ~41MB
        sha256=None,  # jazzomat doesn't publish checksums; we rely on size + SQLite magic
    ),
    "omnibook": CorpusSpec(
        name="omnibook",
        display_name="Charlie Parker Omnibook (50 solos, MusicXML)",
        url=OMNIBOOK_URL,
        target_path=Path("omnibook"),
        expected_size_bytes=400_000,  # ~400KB zipped
        sha256=None,
    ),
}


def _progress_hook(chunk_size: int, total: int | None) -> Callable[[int, int, None], None]:
    """Returns a urllib hook that prints download progress."""
    last_pct = -1

    def hook(block_num: int, block_size: int, _total: int) -> None:
        nonlocal last_pct
        if total is None or total == 0:
            return
        downloaded = block_num * block_size
        pct = min(100, int(downloaded * 100 / total))
        if pct >= last_pct + 5:  # update every 5%
            last_pct = pct
            mb = downloaded / 1_000_000
            total_mb = total / 1_000_000
            print(f"\r  {pct:3d}%  {mb:.1f}/{total_mb:.1f} MB", end="", flush=True)
    return hook


def _download_with_progress(url: str, dest: Path) -> None:
    """Download a URL to dest, with a simple progress bar to stdout."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            total = response.length if hasattr(response, "length") else None
            hook = _progress_hook(4096, total)
            with dest.open("wb") as f:
                shutil.copyfileobj(response, f, length=4096)
            # Call hook one final time to show 100%
            if total:
                hook(total // 4096 + 1, 4096, total)
        print()  # newline after progress
    except urllib.error.URLError as exc:
        msg = f"Download failed: {url} ({exc})"
        raise RuntimeError(msg) from exc


def _verify(spec: CorpusSpec, path: Path) -> None:
    """Verify a downloaded file is what we expect.

    Checks:
    1. File exists and is non-empty (>100 bytes — clearly truncated)
    2. File magic bytes match a known format (SQLite, ZIP, MusicXML)
    3. If a SHA256 is provided, the file matches

    This catches: 404 HTML pages, truncated downloads, redirects
    to other content. We intentionally don't fail on size being
    smaller than expected (the upstream may have grown/shrunk) — we
    leave that as a warning in the future.
    """
    if not path.exists():
        msg = f"Download succeeded but file missing: {path}"
        raise RuntimeError(msg)
    size = path.stat().st_size
    if size < 100:
        msg = f"{path} is only {size} bytes — download likely truncated"
        raise RuntimeError(msg)

    with path.open("rb") as f:
        magic = f.read(16)

    # Magic-byte check. Each format has a known signature.
    if spec.name == "wjazzd":
        if not magic.startswith(b"SQLite format 3"):
            # The SQLite header is 16 bytes: "SQLite format 3\x00"
            # Common failure: a redirect to an HTML error page
            with path.open("rb") as f:
                head = f.read(512)
            if head.lstrip().startswith((b"<!DOCTYPE", b"<html", b"<HTML")):
                msg = f"{path} is an HTML page, not a SQLite file (check the URL)"
            else:
                msg = f"{path} is not a SQLite file (magic: {magic!r})"
            raise RuntimeError(msg)
    elif spec.name == "omnibook":
        # Both ZIP and uncompressed XML should be detected
        if not (magic.startswith(b"PK\x03\x04") or magic.startswith(b"<?xml") or magic.startswith(b"<!DOC")):
            with path.open("rb") as f:
                head = f.read(512)
            if head.lstrip().startswith((b"<!DOCTYPE", b"<html", b"<HTML")):
                msg = f"{path} is an HTML page, not a zip/xml file (check the URL)"
            else:
                msg = f"{path} is not a zip or xml file (magic: {magic!r})"
            raise RuntimeError(msg)

    if spec.sha256:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != spec.sha256:
            msg = f"SHA256 mismatch for {path}: expected {spec.sha256}, got {actual}"
            raise RuntimeError(msg)


def download_wjazzd(data_dir: Path, *, force: bool = False) -> Path:
    """Download the WJAZD SQLite database. Returns the local path."""
    spec = CORPORA["wjazzd"]
    dest = data_dir / spec.target_path
    if dest.exists() and not force:
        return dest
    if dest.exists() and force:
        dest.unlink()
    print(f"Downloading {spec.display_name}...")
    print(f"  from: {spec.url}")
    print(f"  to:   {dest}")
    _download_with_progress(spec.url, dest)
    _verify(spec, dest)
    # Smoke test: confirm it's a real SQLite file
    with dest.open("rb") as f:
        magic = f.read(16)
    if not magic.startswith(b"SQLite format 3"):
        msg = f"{dest} is not a SQLite database (got magic: {magic!r})"
        raise RuntimeError(msg)
    return dest


def download_omnibook(data_dir: Path, *, force: bool = False) -> Path:
    """Download the Omnibook MusicXML zip and unpack. Returns the dir path."""
    spec = CORPORA["omnibook"]
    dest = data_dir / spec.target_path
    if dest.exists() and any(dest.glob("*.xml")) and not force:
        return dest
    if dest.exists() and force:
        shutil.rmtree(dest)
    print(f"Downloading {spec.display_name}...")
    print(f"  from: {spec.url}")
    print(f"  to:   {dest}/")

    # Download to a temp file first
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        _download_with_progress(spec.url, tmp_path)
        _verify(spec, tmp_path)
        # Unpack
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(tmp_path) as zf:
            # The zip contains "Omnibook xml/" — strip that prefix
            for member in zf.namelist():
                if member.endswith("/") or "Omnibook" in member and ".DS_Store" in member:
                    continue
                # Extract just the basename
                basename = Path(member).name
                if not basename.endswith(".xml"):
                    continue
                target = dest / basename
                with zf.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    finally:
        tmp_path.unlink(missing_ok=True)

    xml_count = len(list(dest.glob("*.xml")))
    print(f"  unpacked {xml_count} MusicXML files")
    return dest


def download_all(data_dir: Path, *, force: bool = False, only: list[str] | None = None) -> dict[str, Path]:
    """Download all (or a subset of) corpora. Returns {name: path}."""
    targets = list(CORPORA.keys()) if only is None else only
    if only is not None:
        unknown = set(only) - set(CORPORA)
        if unknown:
            msg = f"Unknown corpora: {unknown}. Available: {list(CORPORA)}"
            raise ValueError(msg)

    results: dict[str, Path] = {}
    for name in targets:
        spec = CORPORA[name]
        if name == "wjazzd":
            results[name] = download_wjazzd(data_dir, force=force)
        elif name == "omnibook":
            results[name] = download_omnibook(data_dir, force=force)
    return results


def data_status(data_dir: Path) -> dict[str, dict[str, object]]:
    """Return a status report for each corpus: present? size? file count?"""
    status: dict[str, dict[str, object]] = {}
    for name, spec in CORPORA.items():
        path = data_dir / spec.target_path
        info: dict[str, object] = {
            "name": name,
            "display_name": spec.display_name,
            "url": spec.url,
            "expected_path": str(path),
        }
        if not path.exists():
            info["present"] = False
        elif path.is_dir():
            files = list(path.glob("*.xml"))
            info["present"] = True
            info["file_count"] = len(files)
            info["size_bytes"] = sum(f.stat().st_size for f in files)
        else:
            info["present"] = True
            info["size_bytes"] = path.stat().st_size
        status[name] = info
    return status
