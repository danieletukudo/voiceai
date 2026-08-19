"""Collect Maps and/or website text and ingest it into RAG (no profile file)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llama_index.core import Document

from ingest import ingest_documents, load_file_documents
from maps_profile import (
    fetch_profile,
    format_text,
    is_maps_url,
    merge_profiles,
)
from website_profile import scrape_website

MAPS_DOC_NAME = "maps-profile"
WEBSITE_DOC_NAME = "website-profile"


def normalize_source_urls(
    maps_url: str | None,
    website_url: str | None,
) -> tuple[str | None, str | None]:
    """Accept either field, and swap if a Maps link was pasted
    in the website box.
    """
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


def _text_document(
    text: str,
    *,
    name: str,
    source: str | None,
    kind: str,
) -> Document | None:
    body = (text or "").strip()
    if not body:
        return None
    return Document(
        text=body,
        metadata={
            "file_name": name,
            "filename": name,
            "source": source,
            "source_type": kind,
        },
    )


def ingest_business_sources(
    maps_url: str | None = None,
    website_url: str | None = None,
    extra_files: list[Path] | None = None,
    docs_dir: Path | None = None,
) -> dict[str, Any]:
    """Pass Maps text, website text, and uploaded files straight into RAG."""
    del docs_dir  # kept so the API call site does not need to change
    documents: list[Document] = []
    ingested: list[str] = []
    warnings: list[str] = []
    maps_profile = None
    website_profile = None
    name = None

    if maps_url or website_url:
        maps_profile, website_profile, warnings = collect_business_profile(
            maps_url,
            website_url,
        )
        if maps_profile:
            doc = _text_document(
                format_text(merge_profiles(maps_profile, None), full_pages=True),
                name=MAPS_DOC_NAME,
                source=(maps_profile.get("source_url") or maps_url),
                kind="maps",
            )
            if doc:
                documents.append(doc)
                ingested.append(MAPS_DOC_NAME)
                name = maps_profile.get("name") or name
        if website_profile:
            doc = _text_document(
                format_text(
                    merge_profiles(None, website_profile),
                    full_pages=True,
                ),
                name=WEBSITE_DOC_NAME,
                source=(website_profile.get("source_url") or website_url),
                kind="website",
            )
            if doc:
                documents.append(doc)
                ingested.append(WEBSITE_DOC_NAME)
                name = name or website_profile.get("name")

    for path in extra_files or []:
        documents.extend(load_file_documents(path))
        ingested.append(path.name)

    if not documents:
        raise ValueError(
            "Provide a Google Maps URL, a website URL, and/or a document"
        )

    ingest_documents(documents)
    return {
        "ok": True,
        "name": name,
        "profile_file": None,
        "ingested": ingested,
        "warnings": warnings,
        "sources": {
            "maps": (maps_profile or {}).get("source_url") or maps_url,
            "website": (website_profile or {}).get("source_url") or website_url,
        },
    }
