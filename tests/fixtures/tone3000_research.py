#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fabio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
"""TONE3000 catalog research / fixture-acquisition tool.

Purpose
-------
Survey the public TONE3000 catalog (https://www.tone3000.com) via the official
REST API v1 to build a complete panorama of available **architectures** and
downloadable **.nam** models — with emphasis on **Architecture 2 (A2)** — and to
suggest, per architecture/size, ONE model that combines a high download count
with high user approval, suitable for seeding ``tests/fixtures/models/``.

Why a script (and its limits)
------------------------------
The TONE3000 API is **authenticated** (OAuth 2.0 + PKCE) and rate-limited
(100 req/min; the ``/tones/search`` endpoint is *heavily* rate-limited). There is
**no anonymous crawl** and **no public bulk dump**. Therefore:

* You must obtain an OAuth **access token** for a logged-in TONE3000 account and
  pass it via ``--token`` or the ``T3K_ACCESS_TOKEN`` environment variable. See
  https://www.tone3000.com/api (Authentication) and the reference client at
  https://github.com/tone-3000/api . The PKCE browser dance is intentionally
  *not* implemented here — paste the resulting access token instead.
* The API exposes ``downloads_count`` and ``favorites_count`` per tone, plus a
  per-model ``size`` and ``architecture_version``. **There is no star-rating
  field.** We therefore use ``favorites_count`` as the user-approval proxy and
  rank candidates by a transparent score (see ``rank_score``).

Licensing caveat (read before committing any .nam!)
---------------------------------------------------
Each tone carries a ``license`` (``t3k``, ``cc-by``, ``cc-by-sa``, ``cc0`` …).
The **default T3K license forbids redistribution** of the data file without the
author's permission. Models destined for the *committed* fixtures tree
(``tests/fixtures/models/``) MUST be CC0/CC-BY(-SA) or have explicit permission.
``--redistributable-only`` enforces this filter so the survey never proposes a
fixture we cannot legally vendor.

Architecture mapping (TONE3000 → nam-rs)
----------------------------------------
* ``architecture=1`` (A1)  → classic two-array WaveNet *and* LSTM/Linear/etc.
  On TONE3000 the A1 "size" enum (standard/lite/feather/nano) maps to the
  nam-rs WaveNet A1 catalog (CH=16/12/8/4).
* ``architecture=2`` (A2)  → the single-array 23-layer A2 WaveNet. A single A2
  ``.nam`` runs as either **A2-Full** (CH=8) or **A2-Lite** (CH=3).
* ``architecture=custom``  → arbitrary WaveNet/LSTM geometries (generic
  topologies the nam-rs dispatcher currently rejects — useful negative fixtures).

Usage
-----
    export T3K_ACCESS_TOKEN="...."          # OAuth access token (Bearer)
    # Survey only (no downloads): write a manifest of top candidates.
    python3 tests/fixtures/tone3000_research.py survey \
        --out tests/fixtures/tone3000_survey.json

    # Acquire suggested fixtures (redistributable only) into models/.
    python3 tests/fixtures/tone3000_research.py acquire \
        --redistributable-only \
        --dest tests/fixtures/models \
        --manifest tests/fixtures/models/TONE3000_PROVENANCE.json

Dependencies: Python 3.10+, ``requests`` (``pip install requests``). No other
third-party packages. The script degrades gracefully (clear error) when the
token is missing or the rate limit is hit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
    import requests
except ImportError:  # pragma: no cover - dependency hint
    sys.stderr.write(
        "error: this tool requires the 'requests' package "
        "(pip install requests)\n"
    )
    raise SystemExit(2)

API_BASE = "https://www.tone3000.com/api/v1"

# TONE3000 enums (mirrored from https://www.tone3000.com/api#enums).
ARCHITECTURES = ("1", "2", "custom")  # A1, A2, Custom
SIZES = ("standard", "lite", "feather", "nano", "custom")
# Licenses that permit vendoring the data file into a public repo.
REDISTRIBUTABLE_LICENSES = frozenset(
    {"cc0", "cco", "cc-by", "cc-by-sa"}
)

# Politeness: stay well under the 100 req/min cap and the stricter /search cap.
REQUEST_INTERVAL_S = 1.2


@dataclass
class ToneCandidate:
    """A ranked tone with the per-architecture model breakdown."""

    tone_id: int
    title: str
    url: str
    gear: str
    license: str
    architecture: str
    downloads_count: int
    favorites_count: int
    a1_models_count: int = 0
    a2_models_count: int = 0
    custom_models_count: int = 0
    models: list[dict[str, Any]] = field(default_factory=list)

    @property
    def rank_score(self) -> float:
        """Transparent popularity×approval score.

        Downloads dominate (raw reach); favorites act as a quality multiplier.
        log-free, monotonic, and easy to audit. NOT an official TONE3000 metric.
        """
        return self.downloads_count + 25.0 * self.favorites_count

    @property
    def redistributable(self) -> bool:
        return self.license.lower() in REDISTRIBUTABLE_LICENSES


class Tone3000Client:
    """Thin authenticated wrapper over the TONE3000 REST API v1."""

    def __init__(self, token: str, *, verbose: bool = False) -> None:
        if not token:
            raise SystemExit(
                "error: no access token. Set T3K_ACCESS_TOKEN or pass --token.\n"
                "Obtain one via the OAuth PKCE flow at https://www.tone3000.com/api"
            )
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "User-Agent": "nam-rs-fixture-research/1.0 (+https://www.tone3000.com/api)",
            }
        )
        self._verbose = verbose
        self._last_request = 0.0

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        # Simple client-side throttle to respect the documented rate limits.
        elapsed = time.monotonic() - self._last_request
        if elapsed < REQUEST_INTERVAL_S:
            time.sleep(REQUEST_INTERVAL_S - elapsed)
        url = f"{API_BASE}{path}"
        if self._verbose:
            sys.stderr.write(f"GET {url} {params or ''}\n")
        resp = self._session.get(url, params=params, timeout=30)
        self._last_request = time.monotonic()
        if resp.status_code == 429:
            retry = int(resp.headers.get("Retry-After", "30"))
            sys.stderr.write(f"rate limited; sleeping {retry}s\n")
            time.sleep(retry)
            return self._get(path, params)
        resp.raise_for_status()
        return resp.json()

    def search_tones(
        self,
        *,
        architecture: str,
        sizes: Iterable[str] | None = None,
        gears: Iterable[str] | None = None,
        sort: str = "downloads-all-time",
        page: int = 1,
        page_size: int = 25,
    ) -> dict[str, Any]:
        """GET /tones/search — platform locked to NAM."""
        params: dict[str, Any] = {
            "platform": "nam",
            "architecture": architecture,
            "sort": sort,
            "page": page,
            "page_size": page_size,
        }
        if sizes:
            params["sizes"] = "_".join(sizes)
        if gears:
            params["gears"] = "_".join(gears)
        return self._get("/tones/search", params)

    def list_models(self, tone_id: int, *, architecture: str) -> list[dict[str, Any]]:
        """GET /models?tone_id=..&architecture=.. (paginated)."""
        out: list[dict[str, Any]] = []
        page = 1
        while True:
            data = self._get(
                "/models",
                {
                    "tone_id": tone_id,
                    "architecture": architecture,
                    "page": page,
                    "page_size": 100,
                },
            )
            out.extend(data.get("data", []))
            if page >= data.get("total_pages", 1):
                break
            page += 1
        return out

    def download_model(self, model_url: str, dest: Path) -> None:
        """Stream a pre-signed ``model_url`` to ``dest`` (Bearer required)."""
        with self._session.get(model_url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    fh.write(chunk)


def sanitize_filename(name: str | None, *, fallback: str) -> str:
    """Reduce a remote-supplied model name to a safe ``.nam`` basename.

    TONE3000 model names are user-uploaded content, so they must never be
    trusted to build an on-disk path: strip any directory components (``/``,
    ``\\``) and reject traversal sentinels before use.
    """
    candidate = Path((name or "").replace("\\", "/")).name.strip()
    if not candidate or candidate in {".", ".."}:
        candidate = fallback
    if not candidate.endswith(".nam"):
        candidate += ".nam"
    return candidate


def collect_candidates(
    client: Tone3000Client,
    *,
    architecture: str,
    sizes: Iterable[str] | None,
    pages: int,
    redistributable_only: bool,
) -> list[ToneCandidate]:
    """Survey the top tones for one architecture and rank them."""
    candidates: list[ToneCandidate] = []
    for page in range(1, pages + 1):
        result = client.search_tones(
            architecture=architecture, sizes=sizes, page=page
        )
        rows = result.get("data", [])
        if not rows:
            break
        for tone in rows:
            lic = (tone.get("license") or "").lower()
            cand = ToneCandidate(
                tone_id=tone["id"],
                title=tone.get("title", ""),
                url=tone.get("url", ""),
                gear=tone.get("gear", ""),
                license=lic,
                architecture=architecture,
                downloads_count=tone.get("downloads_count", 0),
                favorites_count=tone.get("favorites_count", 0),
                a1_models_count=tone.get("a1_models_count", 0),
                a2_models_count=tone.get("a2_models_count", 0),
                custom_models_count=tone.get("custom_models_count", 0),
            )
            if redistributable_only and not cand.redistributable:
                continue
            candidates.append(cand)
    candidates.sort(key=lambda c: c.rank_score, reverse=True)
    return candidates


def suggest_one_per_size(
    client: Tone3000Client,
    candidates: list[ToneCandidate],
    *,
    architecture: str,
) -> dict[str, ToneCandidate]:
    """Pick the single highest-ranked tone per declared size bucket.

    Resolves each chosen tone's concrete model list so the manifest carries the
    downloadable ``model_url`` + ``size`` per file.
    """
    by_size: dict[str, ToneCandidate] = {}
    for cand in candidates:
        models = client.list_models(cand.tone_id, architecture=architecture)
        cand.models = models
        sizes_present = {m.get("size", "custom") for m in models}
        for size in sizes_present:
            if size not in by_size:
                by_size[size] = cand
    return by_size


def cmd_survey(args: argparse.Namespace) -> int:
    client = Tone3000Client(args.token, verbose=args.verbose)
    report: dict[str, Any] = {
        "source": "tone3000.com REST API v1",
        "ranking": "score = downloads + 25*favorites (favorites as approval proxy; "
        "TONE3000 exposes no star rating)",
        "architectures": {},
    }
    archs = [args.architecture] if args.architecture else list(ARCHITECTURES)
    for arch in archs:
        cands = collect_candidates(
            client,
            architecture=arch,
            sizes=None,
            pages=args.pages,
            redistributable_only=args.redistributable_only,
        )
        suggestions = suggest_one_per_size(client, cands[: args.top], architecture=arch)
        report["architectures"][arch] = {
            "top_candidates": [asdict(c) for c in cands[: args.top]],
            "suggested_per_size": {
                size: {
                    "tone_id": c.tone_id,
                    "title": c.title,
                    "url": c.url,
                    "license": c.license,
                    "downloads_count": c.downloads_count,
                    "favorites_count": c.favorites_count,
                    "redistributable": c.redistributable,
                }
                for size, c in suggestions.items()
            },
        }
    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    sys.stderr.write(f"wrote survey → {out}\n")
    return 0


def cmd_acquire(args: argparse.Namespace) -> int:
    client = Tone3000Client(args.token, verbose=args.verbose)
    dest = Path(args.dest)
    provenance: list[dict[str, Any]] = []
    archs = [args.architecture] if args.architecture else list(ARCHITECTURES)
    for arch in archs:
        cands = collect_candidates(
            client,
            architecture=arch,
            sizes=None,
            pages=args.pages,
            redistributable_only=True,  # never auto-download non-redistributable
        )
        suggestions = suggest_one_per_size(client, cands[: args.top], architecture=arch)
        for size, cand in suggestions.items():
            for model in cand.models:
                if model.get("size") != size:
                    continue
                name = sanitize_filename(
                    model.get("name"),
                    fallback=f"tone{cand.tone_id}_{size}.nam",
                )
                target = dest / name
                # Defense-in-depth: never escape the destination directory.
                if dest.resolve() not in target.resolve().parents:
                    raise ValueError(
                        f"refusing unsafe model path {target!r} (from name "
                        f"{model.get('name')!r})"
                    )
                client.download_model(model["model_url"], target)
                provenance.append(
                    {
                        "file": name,
                        "tone_id": cand.tone_id,
                        "tone_title": cand.title,
                        "tone_url": cand.url,
                        "architecture": arch,
                        "size": size,
                        "license": cand.license,
                        "downloads_count": cand.downloads_count,
                        "favorites_count": cand.favorites_count,
                        "model_id": model.get("id"),
                    }
                )
                sys.stderr.write(f"downloaded {name} (tone {cand.tone_id})\n")
                break  # one file per size bucket
    manifest = Path(args.manifest)
    manifest.write_text(json.dumps(provenance, indent=2, ensure_ascii=False))
    sys.stderr.write(f"wrote provenance manifest → {manifest}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Research / acquire TONE3000 NAM fixtures (A1/A2/custom).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--token",
        default=os.environ.get("T3K_ACCESS_TOKEN", ""),
        help="OAuth access token (or set T3K_ACCESS_TOKEN).",
    )
    p.add_argument(
        "--architecture",
        choices=ARCHITECTURES,
        help="Restrict to one architecture (default: all of 1, 2, custom).",
    )
    p.add_argument("--pages", type=int, default=2, help="Search pages to scan (×25).")
    p.add_argument("--top", type=int, default=10, help="Top-N candidates to resolve.")
    p.add_argument("--verbose", action="store_true", help="Log each API request.")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("survey", help="Write a ranked candidate manifest (no download).")
    s.add_argument("--out", default="tests/fixtures/tone3000_survey.json")
    s.add_argument(
        "--redistributable-only",
        action="store_true",
        help="Only consider CC0/CC-BY(-SA) licensed tones.",
    )
    s.set_defaults(func=cmd_survey)

    a = sub.add_parser("acquire", help="Download suggested fixtures (redistributable).")
    a.add_argument("--dest", default="tests/fixtures/models")
    a.add_argument("--manifest", default="tests/fixtures/models/TONE3000_PROVENANCE.json")
    a.add_argument(
        "--redistributable-only",
        action="store_true",
        default=True,
        help="(Forced on for acquire.)",
    )
    a.set_defaults(func=cmd_acquire)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
