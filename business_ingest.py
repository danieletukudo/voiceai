"""Collect Maps and/or website text and ingest into Qdrant RAG (with optional files)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ingest_document import ingest_file, ingest_text
from maps_profile import (
    fetch_profile,
    format_text,
    is_maps_url,
    merge_profiles,
)
from website_profile import scrape_website

MAPS_DOC_NAME = "maps-profile.txt"
WEBSITE_DOC_NAME = "website-profile.txt"


def normalize_source_urls(
    maps_url: str | None,
    website_url: str | None,
) -> tuple[str | None, str | None]:
    """Accept either field, and swap if a Maps link was pasted in the website box."""
    maps = (maps_url or "").strip() or None
    website = (website_url or "").strip() or None
    if website and is_maps_url(website) and not maps:
        maps, website = website, None
    elif maps and not is_maps_url(maps) and not website:
        website, maps = maps, None
    elif maps and not is_maps_url(maps) and website and is_maps_url(website):
        maps, website = website, maps
    return maps, website


def collect_business_profile(
    maps_url: str | None = None,
    website_url: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    """Fetch Maps and/or website. Either source is enough."""
    maps_url, website_url = normalize_source_urls(maps_url, website_url)
    if not maps_url and not website_url:
        raise ValueError("Provide a Google Maps URL, a website URL, or both")

    warnings: list[str] = []
    maps_profile = None
    website_profile = None

    if maps_url:
        try:
            maps_profile = fetch_profile(maps_url)
        except Exception as exc:
            warnings.append(f"Google Maps could not be read: {exc}")

    if website_url:
        try:
            website_profile = scrape_website(website_url)
        except Exception as exc:
            warnings.append(f"Website could not be read: {exc}")

    if maps_profile is None and website_profile is None:
        detail = "; ".join(warnings) or "No business data returned"
        raise RuntimeError(detail)

    return maps_profile, website_profile, warnings


def ingest_business_sources(
    maps_url: str | None = None,
    website_url: str | None = None,
    extra_files: list[Path] | None = None,
    docs_dir: Path | None = None,
) -> dict[str, Any]:
    """Pass Maps text, website text, and uploaded files straight into RAG."""
    ingested: list[str] = []
    warnings: list[str] = []
    maps_profile = None
    website_profile = None
    name = None
    total_chunks = 0

    if maps_url or website_url:
        maps_profile, website_profile, warnings = collect_business_profile(
            maps_url,
            website_url,
        )
        if maps_profile:
            text = format_text(merge_profiles(maps_profile, None), full_pages=True)
            if text.strip():
                total_chunks += ingest_text(
                    text,
                    source_file=MAPS_DOC_NAME,
                    source_type="maps",
                    source_url=(maps_profile.get("source_url") or maps_url),
                )
                ingested.append(MAPS_DOC_NAME)
                name = maps_profile.get("name") or name
                if docs_dir is not None:
                    (docs_dir / MAPS_DOC_NAME).write_text(text, encoding="utf-8")
        if website_profile:
            text = format_text(merge_profiles(None, website_profile), full_pages=True)
            if text.strip():
                total_chunks += ingest_text(
                    text,
                    source_file=WEBSITE_DOC_NAME,
                    source_type="website",
                    source_url=(website_profile.get("source_url") or website_url),
                )
                ingested.append(WEBSITE_DOC_NAME)
                name = name or website_profile.get("name")
                if docs_dir is not None:
                    (docs_dir / WEBSITE_DOC_NAME).write_text(text, encoding="utf-8")

    for path in extra_files or []:
        total_chunks += ingest_file(path)
        ingested.append(path.name)

    if not ingested:
        raise ValueError(
            "Provide a Google Maps URL, a website URL, and/or a document"
        )

    return {
        "ok": True,
        "name": name,
        "profile_file": None,
        "ingested": ingested,
        "chunks": total_chunks,
        "warnings": warnings,
        "sources": {
            "maps": (maps_profile or {}).get("source_url") or maps_url,
            "website": (website_profile or {}).get("source_url") or website_url,
        },
    }
