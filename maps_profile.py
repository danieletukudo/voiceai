"""Read a Google Maps business profile, a business website, or both.

Reviews are ignored on Maps. Review-like website pages are skipped.

Example:
    python src/maps_profile.py
    python src/maps_profile.py "https://www.google.com/maps/place/..."
    python src/maps_profile.py "https://theboroughlagos.com/"
    python src/maps_profile.py "https://www.google.com/maps/place/..." "https://theboroughlagos.com/"
"""

from __future__ import annotations

import argparse
import html as htmlmod
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit, urlunsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from website_profile import scrape_website

DEFAULT_URL = (
    "https://www.google.com/maps/place/CIRCA+LAGOS/@6.4498536,3.4735446,17z/data=!4m8!3m7!1s0x103bf50d7d7f3c87:0x917a07ce588c2cfa!8m2!3d6.4498536!4d3.4735446!9m1!1b1!16s%2Fg%2F11hdkdxfgd?entry=ttu&g_ep=EgoyMDI2MDgxMi4wIKXMDSoASAFQAw%3D%3D"



)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

WEEKDAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
PLUS_CODE_RE = re.compile(r"\b[A-Z0-9]{4}\+[A-Z0-9]{2,3}\b")
TEL_RE = re.compile(r"tel:([+\d][\d\s\-()]{6,})")
DATA_ID_RE = re.compile(r"^0x[0-9a-f]+:0x[0-9a-f]+$", re.I)
TRACKING_PREFIXES = ("0ahUK", "AOvVaw", ",AOvVaw")
NOISE_HOSTS = (
    "gstatic.com",
    "googleusercontent.com",
    "support.google.com",
    "business.google.com",
    "fonts.gstatic.com",
)


def at(node: Any, *idxs: int | str, default: Any = None) -> Any:
    cur = node
    for idx in idxs:
        if cur is None:
            return default
        try:
            cur = cur[idx]
        except (TypeError, IndexError, KeyError):
            return default
    return cur


def _fetch(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def clean_maps_url(url: str) -> str:
    """Drop Maps tracking query params. Keep the data= protobuf intact."""
    parts = urlsplit(url)
    query = parse_qs(parts.query, keep_blank_values=True)
    for key in ("entry", "g_ep", "utm_source", "utm_medium", "utm_campaign"):
        query.pop(key, None)
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query, doseq=True), "")
    )


def _preview_url_from_html(html: str, page_url: str) -> str | None:
    match = re.search(r'href="(/maps/preview/place[^"]+)"', html)
    if not match:
        return None
    return urljoin(page_url, htmlmod.unescape(match.group(1)))


def _load_xssi_json(raw: str) -> Any:
    text = raw.lstrip()
    if text.startswith(")]}'"):
        text = text[4:].lstrip()
    return json.loads(text)


def _is_noise_string(value: str) -> bool:
    if not value or not isinstance(value, str):
        return True
    if any(value.startswith(prefix) for prefix in TRACKING_PREFIXES):
        return True
    if any(host in value for host in NOISE_HOSTS):
        return True
    if value.startswith("/geo/type/"):
        return False
    if value.startswith("/") and " " not in value and len(value) < 8:
        return True
    return False


def _unique_links(links: list[dict[str, str]], skip_url: str | None = None) -> list[dict[str, str]]:
    by_url: dict[str, dict[str, str]] = {}
    for link in links:
        url = link.get("url")
        if not url or url == skip_url:
            continue
        existing = by_url.get(url)
        if not existing or (not existing.get("label") and link.get("label")):
            by_url[url] = link
    return list(by_url.values())


def _unique(items: list[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for item in items:
        key = json.dumps(item, sort_keys=True, ensure_ascii=False) if not isinstance(item, str) else item
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _walk(node: Any):
    if isinstance(node, (str, int, float, bool)) or node is None:
        yield node
        return
    if isinstance(node, list):
        yield node
        for child in node:
            yield from _walk(child)
    elif isinstance(node, dict):
        yield node
        for child in node.values():
            yield from _walk(child)


def _find_place_blob(payload: Any) -> Any:
    """Locate the listing array even if Google shifts top-level indexes."""
    candidate = at(payload, 6)
    if isinstance(candidate, list) and isinstance(at(candidate, 11), str):
        return candidate
    for node in _walk(payload):
        if not isinstance(node, list) or len(node) < 14:
            continue
        data_id = at(node, 10)
        name = at(node, 11)
        if isinstance(data_id, str) and DATA_ID_RE.match(data_id) and isinstance(name, str):
            return node
    return candidate


def _looks_like_review_blob(node: Any) -> bool:
    """Heuristic: skip arrays that are user reviews, not listing facts."""
    if not isinstance(node, list) or len(node) < 3:
        return False
    texts = [x for x in node if isinstance(x, str)]
    joined = " ".join(texts).lower()
    review_markers = (
        "translated by google",
        " liked this review",
        "local guide",
        "photos of this place",
    )
    if any(marker in joined for marker in review_markers):
        return True
    return False


def _collect_hours(place: Any) -> list[dict[str, Any]]:
    hours: list[dict[str, Any]] = []
    for node in _walk(place):
        if not isinstance(node, list) or not node:
            continue
        if _looks_like_review_blob(node):
            continue
        day = node[0]
        if day not in WEEKDAYS:
            continue
        ranges: list[str] = []
        if len(node) > 3 and isinstance(node[3], list):
            for block in node[3]:
                if isinstance(block, list) and block and isinstance(block[0], str):
                    ranges.append(block[0])
        hours.append({"day": day, "hours": ranges or None})
    # Keep one entry per weekday (first complete one wins).
    by_day: dict[str, dict[str, Any]] = {}
    for row in hours:
        if row["day"] not in by_day or (row["hours"] and not by_day[row["day"]]["hours"]):
            by_day[row["day"]] = row
    ordered = [by_day[day] for day in WEEKDAYS if day in by_day]
    return ordered


def _collect_attributes(place: Any) -> list[dict[str, str]]:
    attrs: list[dict[str, str]] = []
    for node in _walk(place):
        if not isinstance(node, list) or len(node) < 3:
            continue
        group_id, group_label, items = node[0], node[1], node[2]
        if not isinstance(group_id, str) or not isinstance(group_label, str):
            continue
        if not isinstance(items, list) or not items:
            continue
        if not all(isinstance(item, list) and item and isinstance(item[0], str) for item in items[:1]):
            continue
        if not str(items[0][0]).startswith("/geo/type/"):
            continue
        values = []
        for item in items:
            if isinstance(item, list) and len(item) > 1 and isinstance(item[1], str):
                values.append(item[1])
        if values:
            attrs.append({"group": group_label, "values": _unique(values)})
    return _unique(attrs)


def parse_place(payload: Any) -> dict[str, Any]:
    place = _find_place_blob(payload)
    if not isinstance(place, list) or not isinstance(at(place, 11), str):
        raise ValueError("Unexpected Maps payload; could not find place data")

    address_parts = [p for p in (at(place, 2) or []) if isinstance(p, str)]
    website_url = at(place, 7, 0)
    website_label = at(place, 7, 1)
    lat = at(place, 9, 2)
    lng = at(place, 9, 3)
    name = at(place, 11)
    categories = [c for c in (at(place, 13) or []) if isinstance(c, str)]
    neighborhood = at(place, 14)
    full_address = at(place, 18) or at(place, 39)
    timezone = at(place, 30)

    descriptions: list[str] = []
    about = at(place, 32)
    if isinstance(about, list):
        for block in about:
            if isinstance(block, list) and len(block) > 1 and isinstance(block[1], str):
                if not _is_noise_string(block[1]):
                    descriptions.append(block[1])
    owner_blurb = at(place, 154, 0, 0)
    if isinstance(owner_blurb, str) and not _is_noise_string(owner_blurb):
        descriptions.insert(0, owner_blurb)

    phones: list[str] = []
    phone_block = at(place, 178, 0)
    if isinstance(phone_block, list):
        if isinstance(phone_block[0], str):
            phones.append(phone_block[0])
        variants = at(phone_block, 1) or []
        if isinstance(variants, list):
            for variant in variants:
                if isinstance(variant, list) and variant and isinstance(variant[0], str):
                    phones.append(variant[0])
        if isinstance(at(phone_block, 3), str):
            phones.append(at(phone_block, 3))

    plus_codes: list[str] = []
    status_lines: list[str] = []
    action_links: list[dict[str, str]] = []
    for node in _walk(place):
        if _looks_like_review_blob(node):
            continue
        if isinstance(node, str):
            if _is_noise_string(node):
                continue
            plus_codes.extend(PLUS_CODE_RE.findall(node))
            tel = TEL_RE.search(node)
            if tel:
                phones.append(tel.group(1))
            if node.startswith(("Closed", "Open")) and "·" in node:
                status_lines.append(node)
            continue
        if isinstance(node, list) and len(node) >= 2 and isinstance(node[0], str):
            url = node[0]
            label = node[1] if isinstance(node[1], str) else None
            if url.startswith("http") and not _is_noise_string(url):
                if "google.com/maps" in url or "google.com/local" in url:
                    continue
                action_links.append({"url": url, "label": label})

    profile = {
        "name": name,
        "categories": _unique(categories),
        "neighborhood": neighborhood,
        "address": full_address,
        "address_lines": address_parts,
        "plus_code": next(iter(_unique(plus_codes)), None),
        "phone": next(iter(_unique(phones)), None),
        "phones": _unique(phones),
        "website": website_url if isinstance(website_url, str) else None,
        "website_label": website_label if isinstance(website_label, str) else None,
        "latitude": lat,
        "longitude": lng,
        "timezone": timezone,
        "rating": at(place, 4, 7),
        "status": next(iter(_unique(status_lines)), None),
        "hours": _collect_hours(place),
        "description": _unique(descriptions),
        "attributes": _collect_attributes(place),
        "links": _unique_links(action_links, skip_url=website_url if isinstance(website_url, str) else None),
        "data_id": at(place, 10),
    }
    return profile


def fetch_profile(maps_url: str) -> dict[str, Any]:
    page_url = clean_maps_url(maps_url)
    html = _fetch(page_url)
    preview_url = _preview_url_from_html(html, page_url)
    if not preview_url:
        raise RuntimeError("Could not find Maps place payload in the page HTML")
    raw = _fetch(preview_url)
    payload = _load_xssi_json(raw)
    profile = parse_place(payload)
    profile["source_url"] = page_url
    return profile


def is_maps_url(url: str) -> bool:
    parts = urlsplit(url)
    host = parts.netloc.lower()
    path = parts.path.lower()
    return "google." in host and ("/maps" in path or host.startswith("maps.google."))


def _first(*values: Any) -> Any:
    for value in values:
        if value in (None, "", [], {}):
            continue
        return value
    return None


def merge_profiles(maps: dict[str, Any] | None, website: dict[str, Any] | None) -> dict[str, Any]:
    """Fold Maps and website facts into one profile."""
    maps = maps or {}
    website = website or {}
    descriptions = _unique(
        list(maps.get("description") or []) + list(website.get("description") or [])
    )
    phones = _unique(list(maps.get("phones") or []) + list(website.get("phones") or []))
    links = list(maps.get("links") or [])
    for url in (website.get("social") or []) + (website.get("whatsapp") or []):
        links.append({"url": url, "label": None})
    website_url = _first(maps.get("website"), website.get("website"))
    combined = {
        "name": _first(maps.get("name"), website.get("name")),
        "categories": maps.get("categories") or website.get("types") or [],
        "neighborhood": maps.get("neighborhood"),
        "address": _first(maps.get("address"), website.get("address")),
        "address_lines": maps.get("address_lines") or [],
        "plus_code": maps.get("plus_code"),
        "phone": _first(maps.get("phone"), next(iter(phones), None)),
        "phones": phones,
        "emails": website.get("emails") or [],
        "whatsapp": website.get("whatsapp") or [],
        "social": website.get("social") or [],
        "website": website_url,
        "website_label": maps.get("website_label"),
        "latitude": maps.get("latitude"),
        "longitude": maps.get("longitude"),
        "timezone": maps.get("timezone"),
        "rating": maps.get("rating"),
        "status": maps.get("status"),
        "hours": maps.get("hours") or [],
        "website_hours": website.get("hours") or [],
        "description": descriptions,
        "attributes": maps.get("attributes") or [],
        "links": _unique_links(links, skip_url=website_url),
        "pages": website.get("pages") or [],
        "data_id": maps.get("data_id"),
        "sources": {
            "maps": maps.get("source_url"),
            "website": website.get("source_url"),
        },
    }
    return combined


def classify_urls(positional: list[str], maps_url: str | None, website_url: str | None) -> tuple[str | None, str | None]:
    maps = maps_url
    website = website_url
    leftover: list[str] = []
    for url in positional:
        if is_maps_url(url):
            if maps:
                leftover.append(url)
            else:
                maps = url
        else:
            if website:
                leftover.append(url)
            else:
                website = url
    if leftover:
        raise ValueError(
            "Pass one Google Maps URL, one website URL, or both. Extra URL: "
            + leftover[0]
        )
    return maps, website


def format_text(profile: dict[str, Any], *, full_pages: bool = False) -> str:
    lines = [
        profile.get("name") or "(unknown place)",
        "=" * len(profile.get("name") or "(unknown place)"),
    ]
    if profile.get("categories"):
        lines.append("Category: " + ", ".join(profile["categories"]))
    if profile.get("neighborhood"):
        lines.append(f"Area: {profile['neighborhood']}")
    if profile.get("address"):
        lines.append(f"Address: {profile['address']}")
    if profile.get("plus_code"):
        lines.append(f"Plus code: {profile['plus_code']}")
    if profile.get("phone"):
        lines.append(f"Phone: {profile['phone']}")
    if profile.get("website"):
        lines.append(f"Website: {profile['website']}")
    if profile.get("emails"):
        lines.append("Email: " + ", ".join(profile["emails"]))
    if profile.get("whatsapp"):
        lines.append("WhatsApp: " + ", ".join(profile["whatsapp"]))
    if profile.get("social"):
        lines.append("Social: " + ", ".join(profile["social"]))
    if profile.get("status"):
        lines.append(f"Status: {profile['status']}")
    if profile.get("timezone"):
        lines.append(f"Timezone: {profile['timezone']}")
    if profile.get("rating") is not None:
        lines.append(f"Rating: {profile['rating']}")
    if profile.get("latitude") is not None and profile.get("longitude") is not None:
        lines.append(f"Coordinates: {profile['latitude']}, {profile['longitude']}")
    if profile.get("hours"):
        lines.append("Hours (Google Maps):" if profile.get("website_hours") else "Hours:")
        for row in profile["hours"]:
            hours = ", ".join(row["hours"] or ["(not listed)"])
            lines.append(f"  {row['day']}: {hours}")
    if profile.get("website_hours"):
        lines.append("Hours (website):")
        for row in profile["website_hours"]:
            lines.append(f"  {row}")
    if profile.get("description"):
        lines.append("About:")
        for text in profile["description"]:
            lines.append(f"  {text}")
    if profile.get("attributes"):
        lines.append("Attributes:")
        for group in profile["attributes"]:
            lines.append(f"  {group['group']}: {', '.join(group['values'])}")
    extra_links = [link for link in profile.get("links") or [] if link.get("url")]
    if extra_links:
        lines.append("Other links:")
        for link in extra_links:
            label = link.get("label") or "link"
            lines.append(f"  {label}: {link['url']}")
    if profile.get("pages"):
        lines.append("From the website:")
        for page in profile["pages"]:
            title = page.get("title") or page.get("url")
            lines.append(f"  {title}")
            if page.get("url"):
                lines.append(f"    {page['url']}")
            if page.get("description"):
                lines.append(f"    {page['description']}")
            if page.get("text"):
                if full_pages:
                    lines.append("")
                    lines.append(page["text"])
                    lines.append("")
                else:
                    snippet = page["text"].replace("\n", " ")
                    if len(snippet) > 400:
                        snippet = snippet[:400] + "…"
                    lines.append(f"    {snippet}")
    sources = profile.get("sources") or {}
    used = [f"{kind}: {url}" for kind, url in sources.items() if url]
    if used:
        lines.append("Sources:")
        for row in used:
            lines.append(f"  {row}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch a Google Maps profile and/or scrape a business website (reviews are not read)."
    )
    parser.add_argument(
        "urls",
        nargs="*",
        help="Google Maps place URL and/or website URL",
    )
    parser.add_argument("--maps", help="Google Maps place URL")
    parser.add_argument("--website", help="Business website URL")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON instead of a readable summary",
    )
    args = parser.parse_args(argv)

    try:
        maps_url, website_url = classify_urls(args.urls, args.maps, args.website)
        if not maps_url and not website_url:
            maps_url = DEFAULT_URL

        maps_profile = fetch_profile(maps_url) if maps_url else None
        website_profile = scrape_website(website_url) if website_url else None
        if maps_profile is None and website_profile is None:
            raise RuntimeError("Provide a Google Maps URL, a website URL, or both")
        if maps_profile is not None and website_profile is not None:
            profile = merge_profiles(maps_profile, website_profile)
        elif maps_profile is not None:
            profile = merge_profiles(maps_profile, None)
        else:
            profile = merge_profiles(None, website_profile)
    except Exception as exc:
        print(f"Failed to read profile: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(profile, indent=2, ensure_ascii=False))
    else:
        print(format_text(profile))
        print()
        print("--- JSON ---")
        print(json.dumps(profile, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
