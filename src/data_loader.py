"""Download StatsBomb Open Data with an on-disk cache and retries.

Files are fetched from raw.githubusercontent.com and stored gzip-compressed
under ``cache_dir`` using the same relative path as the upstream repository
(``competitions.json``, ``matches/<comp>/<season>.json``,
``events/<match>.json``). A cached file is never downloaded again, so a second
run of the pipeline works fully offline.
"""
from __future__ import annotations

import gzip
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
import requests

import config

log = logging.getLogger(__name__)


class DataNotAvailableError(RuntimeError):
    """Raised when the requested data does not exist (or is not cached offline)."""


class StatsBombClient:
    def __init__(
        self,
        base_url: str = config.BASE_URL,
        cache_dir: Path | str = config.CACHE_DIR,
        timeout: float = config.HTTP_TIMEOUT,
        retries: int = config.HTTP_RETRIES,
        backoff: float = config.HTTP_BACKOFF,
        offline: bool = False,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.retries = max(1, retries)
        self.backoff = backoff
        self.offline = offline
        self.session = session or requests.Session()
        self.downloads = 0  # number of network fetches (useful in tests)

    # ------------------------------------------------------------------ core IO
    def cache_path(self, rel: str) -> Path:
        return self.cache_dir / (rel + ".gz")

    def get_json(self, rel: str):
        path = self.cache_path(rel)
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                return json.load(fh)
        if self.offline:
            raise DataNotAvailableError(
                f"'{rel}' is not in the cache ({path}) and --offline is set."
            )
        data = self._download(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(data, fh)
        tmp.replace(path)  # atomic: a crash never leaves a half-written cache file
        return data

    def _download(self, rel: str):
        url = f"{self.base_url}/{rel}"
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = self.session.get(url, timeout=self.timeout)
                if resp.status_code == 404:
                    raise DataNotAvailableError(f"404 Not Found: {url}")
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {resp.status_code} for {url}")
                resp.raise_for_status()
                data = resp.json()
                self.downloads += 1
                return data
            except DataNotAvailableError:
                raise
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < self.retries:
                    wait = self.backoff * 2 ** (attempt - 1)
                    log.warning("Attempt %d/%d failed for %s (%s); retrying in %.1fs",
                                attempt, self.retries, url, exc, wait)
                    time.sleep(wait)
        raise ConnectionError(
            f"Could not download {url} after {self.retries} attempts: {last_error}"
        )

    # ------------------------------------------------------------- convenience
    def competitions(self) -> list[dict]:
        return self.get_json("competitions.json")

    def matches(self, competition_id: int, season_id: int) -> list[dict]:
        return self.get_json(f"matches/{competition_id}/{season_id}.json")

    def events(self, match_id: int) -> list[dict]:
        return self.get_json(f"events/{match_id}.json")


# ---------------------------------------------------------------- resolution
def _season_variants(name: str) -> set[str]:
    """'2015/2016' -> {'2015/2016', '2015/16', '2015-2016', '2015-16'}."""
    name = name.strip().lower()
    out = {name, name.replace("/", "-")}
    m = re.fullmatch(r"(\d{4})[/-](\d{2,4})", name)
    if m:
        a, b = m.groups()
        for second in {b[-2:], b if len(b) == 4 else str(int(a[:2] + b[-2:]))}:
            out |= {f"{a}/{second}", f"{a}-{second}"}
    return out


def resolve_competition(competitions: list[dict], competition: str | int,
                        season: str | int) -> dict:
    """Find one competition-season by name or numeric id.

    Raises DataNotAvailableError listing the valid options when not found.
    """
    comp_key = str(competition).strip().lower()
    comps = [c for c in competitions
             if str(c["competition_id"]) == comp_key or c["competition_name"].lower() == comp_key]
    if not comps:
        names = sorted({f"{c['competition_name']} (id {c['competition_id']})" for c in competitions})
        raise DataNotAvailableError(
            f"Competition '{competition}' is not in StatsBomb Open Data. Available: "
            + "; ".join(names)
        )
    season_key = str(season).strip().lower()
    for c in comps:
        if str(c["season_id"]) == season_key or season_key in _season_variants(c["season_name"]):
            return {
                "competition_id": int(c["competition_id"]),
                "season_id": int(c["season_id"]),
                "competition_name": c["competition_name"],
                "season_name": c["season_name"],
            }
    seasons = sorted({f"{c['season_name']} (id {c['season_id']})" for c in comps})
    raise DataNotAvailableError(
        f"Season '{season}' of {comps[0]['competition_name']} is not in StatsBomb Open Data. "
        f"Available seasons: {', '.join(seasons)}"
    )


def matches_frame(raw_matches: list[dict]) -> pd.DataFrame:
    rows = [{
        "match_id": int(m["match_id"]),
        "match_date": m["match_date"],
        "kick_off": m.get("kick_off") or "",
        "match_week": m.get("match_week"),
        "home_team": m["home_team"]["home_team_name"],
        "away_team": m["away_team"]["away_team_name"],
        "home_score": int(m["home_score"]),
        "away_score": int(m["away_score"]),
    } for m in raw_matches]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["match_date", "kick_off", "match_id"]).reset_index(drop=True)


def iter_events(client: StatsBombClient, match_ids: Iterable[int],
                workers: int = config.DOWNLOAD_WORKERS) -> Iterator[tuple[int, list[dict]]]:
    """Yield (match_id, events) in input order, downloading in parallel."""
    match_ids = list(match_ids)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, (mid, events) in enumerate(
            zip(match_ids, pool.map(client.events, match_ids)), start=1
        ):
            if i % 25 == 0 or i == len(match_ids):
                log.info("  events loaded: %d/%d matches", i, len(match_ids))
            yield mid, events
