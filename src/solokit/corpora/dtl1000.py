"""Dig That Lick (DTL) corpus client.

Talks to the DTL HTTP API at https://dig-that-lick.hfm-weimar.de.
The API requires:
- A real browser User-Agent (curl gets "Access denied")
- A CSRF token from the initial GET (form value, not the cookie)
- Form-encoded POST body

The DTL API has no proper schema, so the field names here were
reverse-engineered from the HTML form. They're stable.

Available databases (passed as individual checkboxes):
- "dtl" — DTL1000 (1736 autotranscribed solos, ~6M n-grams)
- "wjazzd" — Weimar Jazz Database (456 hand-transcribed solos)
- "omnibook" — Parker Omnibook (small but canonical)
- "esac" — Essen Folk Song Collection
"""

from __future__ import annotations

import urllib.parse
from collections.abc import Sequence
from typing import Any

import httpx

from solokit.corpora.base import Corpus, CorpusError, SearchResult
from solokit.patterns.ngram import NGram
from solokit.patterns.similarity import Match, search_patterns

# Default browser User-Agent — DTL blocks bare curl/python-requests
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

DATABASE_KEYS = ("dtl", "wjazzd", "omnibook", "esac")


class DTLCorpus(Corpus):
    """Client for the Dig That Lick pattern-search API.

    The corpus is remote — `search()` hits the DTL server and parses
    the HTML / CSV response. There is no `__iter__` for DTL itself;
    you can only search it.

    Example::

        corpus = DTLCorpus()
        results = corpus.search(
            [-1, -1, 4, -5, -2],   # a 5-interval pattern
            transformation="interval",
            min_similarity=0.8,
        )
        for r in results:
            print(f"{r.similarity:.2f}  {r.year}  {r.performer} — {r.title}")
    """

    name = "dtl"

    BASE_URL = "https://dig-that-lick.hfm-weimar.de"
    SEARCH_URL = f"{BASE_URL}/similarity_search/search"
    LANDING_URL = f"{BASE_URL}/similarity_search/"
    EXPORT_URL = f"{BASE_URL}/similarity_search/export_result_as_csv"
    AUDIO_BASE = "https://production-dtl-pattern-api.hfm-weimar.de/static/audio/n_grams"

    def __init__(
        self,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._ua = user_agent
        self._timeout = timeout
        self._client = client or httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=timeout,
            follow_redirects=True,
        )
        self._csrf: str | None = None

    def _ensure_csrf(self) -> str:
        """Fetch the landing page and extract the CSRF form value."""
        if self._csrf is not None:
            return self._csrf
        resp = self._client.get(self.LANDING_URL)
        resp.raise_for_status()
        html = resp.text
        # Find the hidden CSRF input
        marker = 'name="csrfmiddlewaretoken" value="'
        idx = html.find(marker)
        if idx < 0:
            msg = "CSRF token not found in DTL landing page (WAF block?)"
            raise CorpusError(msg)
        start = idx + len(marker)
        end = html.find('"', start)
        self._csrf = html[start:end]
        return self._csrf

    def __enter__(self) -> DTLCorpus:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def __iter__(self):  # type: ignore[override]  # not iterable
        msg = "DTLCorpus is remote and not iterable; use .search() instead"
        raise CorpusError(msg)

    def search(
        self,
        pattern: Sequence[int],
        *,
        transformation: str = "interval",
        databases: Sequence[str] = ("dtl",),
        min_similarity: float = 0.8,
        max_length_difference: int = 0,
        max_edit_distance: int | None = None,
        min_frequency: int = 1,
        limit: int = 50,
    ) -> list[SearchResult]:
        """Search the DTL corpus for matches.

        Args:
            pattern: The pattern values (already transformed).
            databases: Which of {dtl, wjazzd, omnibook, esac} to search.
            Other args: passed through to the DTL form.

        Returns:
            SearchResults with performer / title / year / melid / audio_url.
        """
        csrf = self._ensure_csrf()

        # The DTL "max_length_difference" is 0-5; default 0 (exact length).
        # "max_edit_distance" defaults to 0 (no substitutions). Be careful
        # with high values — search becomes expensive.
        if max_edit_distance is None:
            # Default: allow 0 edits (only matches at given similarity level)
            max_edit_distance = max(1, len(pattern) - int(len(pattern) * min_similarity))

        # Build form payload. DTL expects "n_gram" as comma-separated values.
        form: dict[str, str] = {
            "csrfmiddlewaretoken": csrf,
            "n_gram": ",".join(str(v) for v in pattern),
            "transformation": transformation,
            "minimum_similarity": str(min_similarity),
            "max_length_difference": str(max_length_difference),
            "max_edit_distance": str(max_edit_distance),
            "minimum_frequency": str(min_frequency),
            "filter_category": "3",  # 3 = keep all overlaps
            "within_single_phrase": "True",
            "preserve_contour": "True",
            "preserve_pitch_range": "True",
            "preserve_pattern": "True",
            "pin_first": "True",
            "pin_last": "True",
            "keep_overlapping_instances": "True",
            "group_by": "n_gram",
            "search_id": "",
            "created_at": "",
            "n_gram_pitch": "",
            "target_layout": "",
        }

        # Per-database checkboxes (independent, not a list)
        for db in DATABASE_KEYS:
            form[f"database-{db}"] = "on" if db in databases else ""
            form[f"filter_metadata_{db}"] = "false"

        headers = {
            "Referer": self.LANDING_URL,
            "X-CSRFToken": csrf,
            "User-Agent": self._ua,
            "Accept": "text/html,application/xhtml+xml",
        }

        # POST returns 302 → /similarity_search/search?id=...&...
        try:
            resp = self._client.post(self.SEARCH_URL, data=form, headers=headers)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # DTL's backend sometimes 500s (current known issue: JSONDecodeError
            # on the upstream audio API). Surface a clean error.
            msg = (
                f"DTL search backend returned HTTP {exc.response.status_code}. "
                "The DTL service is currently having a backend issue — "
                "try again later or use a local corpus."
            )
            raise CorpusError(msg) from exc

        # If the POST didn't 302, the response is the result page directly
        results_html = resp.text

        # Parse the results table. DTL renders one <tr> per match.
        # We use a tolerant HTML parser to avoid lxml/BeautifulSoup dep.
        results = self._parse_results_html(results_html, pattern, transformation)

        # Sort by similarity desc (DTL usually returns in this order already,
        # but be defensive)
        results.sort(key=lambda r: -r.match.similarity)
        return results[:limit]

    def _parse_results_html(
        self,
        html: str,
        pattern: Sequence[int],
        transformation: str,
    ) -> list[SearchResult]:
        """Parse the DTL result HTML into SearchResult objects.

        The DTL result page is messy; we look for a table with class
        "search-results" and extract rows. As a fallback, we extract any
        <tr> whose cells contain a "Performer" / "Title" pair.
        """
        results: list[SearchResult] = []
        # Naive but robust row extraction
        import re

        # Rows: <tr ...>...</tr> with cells
        row_re = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
        cell_re = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL | re.IGNORECASE)
        tag_re = re.compile(r"<[^>]+>")
        ws_re = re.compile(r"\s+")

        for row_match in row_re.finditer(html):
            cells = cell_re.findall(row_match.group(1))
            if len(cells) < 4:
                continue
            # Strip tags
            clean = [ws_re.sub(" ", tag_re.sub("", c)).strip() for c in cells]
            try:
                # DTL columns (in order): Database;Melid;N-gram;Frequency;Length;
                # Performer;Title;Recording year;Instrument;Start position;
                # Duration;Pitch range;Contour;Similarity;Edit distance
                if len(clean) < 15:
                    continue
                database = clean[0]
                melid = clean[1]
                frequency = int(clean[3])
                performer = clean[5]
                title = clean[6]
                year_str = clean[7]
                year = int(year_str) if year_str.isdigit() else None
                instrument = clean[8] if len(clean) > 8 else None
                start_str = clean[9] if len(clean) > 9 else ""
                start_pos = float(start_str) if _looks_float(start_str) else None
                duration_str = clean[10] if len(clean) > 10 else ""
                duration = float(duration_str) if _looks_float(duration_str) else None
                similarity = float(clean[13])
                edit_distance = int(clean[14])

                # Build a NGram stub for the source. We don't have the actual
                # matched values from the HTML (they're hidden in the table)
                # so the consumer can re-query the corpus for full details.
                # Edit distance gives us a reasonable length estimate.
                source_gram = NGram(
                    values=tuple(pattern),  # placeholder
                    source_id=melid,
                    onset_beat=start_pos,
                )
                match = Match(
                    source=source_gram,
                    edit_distance=edit_distance,
                    similarity=similarity,
                )
                results.append(
                    SearchResult(
                        match=match,
                        performer=performer,
                        title=title,
                        year=year,
                        melid=melid,
                        audio_url=f"{self.AUDIO_BASE}/{melid}",
                        database=database,
                        instrument=instrument,
                        start_position=start_pos,
                        duration=duration,
                    )
                )
            except (ValueError, IndexError):
                # Skip malformed rows
                continue
        return results


def _looks_float(s: str) -> bool:
    """Cheap check: does this string look like a number?"""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False
