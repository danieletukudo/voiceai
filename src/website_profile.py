"""Scrape a business website for listing facts. Review pages are skipped.

Example:
    python src/website_profile.py https://theboroughlagos.com/
"""

from __future__ import annotations

import argparse
import html as htmlmod
import json
import re
import sys
import urllib.error
import urllib.request
from html.parser import HTMLParser
from typing import Any
from urllib.parse import unquote, urldefrag, urljoin, urlsplit, urlunsplit

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(
    r"""
    (?<!\d)
    (?:
        \+234[\s\-]?[0-9]{10}
        |
        \+234[\s\-]?[0-9]{3}[\s\-]?[0-9]{3}[\s\-]?[0-9]{4}
        |
        0[7-9][0-1][0-9][\s\-]?[0-9]{3}[\s\-]?[0-9]{4}
    )
    (?!\d)
    """,
    re.VERBOSE,
)
KEEP_LD_TYPES = {
    "Hotel",
    "Organization",
    "Restaurant",
    "LocalBusiness",
    "BarOrPub",
    "FoodEstablishment",
    "LodgingBusiness",
    "CafeOrCoffeeShop",
    "Store",
    "Place",
}
SKIP_EMAIL_PARTS = (
    "example.com",
    "sentry.io",
    "wixpress.com",
    "wordpress.com",
    "schema.org",
    "godaddy.com",
)
SKIP_PATH_PARTS = (
    "/review",
    "/reviews",
    "/testimonial",
    "/cart",
    "/checkout",
    "/account",
    "/login",
    "/wp-json",
    "/wp-content",
    "/wp-admin",
    "/wp-includes",
    "/author/",
    "/tag/",
    "/feed",
    "/privacy",
    "/terms",
    "/cookie",
    "/product/",
    "/shop/",
)
PRIORITY_HINTS = (
    ("contact", 90),
    ("about", 85),
    ("menu", 80),
    ("hour", 80),
    ("location", 75),
    ("find-us", 75),
    ("reserv", 70),
    ("book", 65),
    ("circa", 85),
    ("dining", 70),
    ("restaurant", 70),
    ("lobby", 60),
    ("discover", 60),
    ("event", 50),
    ("faq", 50),
    ("room", 40),
    ("stay", 40),
)
SOCIAL_HOSTS = (
    "instagram.com",
    "facebook.com",
    "fb.com",
    "twitter.com",
    "x.com",
    "tiktok.com",
    "youtube.com",
    "linkedin.com",
    "wa.me",
    "api.whatsapp.com",
)
SKIP_TAGS = {"script", "style", "noscript", "svg", "path"}
MAX_PAGES = 8
MAX_TEXT_CHARS = 4000


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.meta: dict[str, str] = {}
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.json_ld_parts: list[str] = []
        self._in_title = False
        self._in_ld = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        data = {k.lower(): (v or "") for k, v in attrs}
        if tag == "script" and data.get("type", "").lower() == "application/ld+json":
            self._in_ld = True
            return
        if tag in SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
        if tag == "meta":
            name = (data.get("name") or data.get("property") or "").lower()
            content = data.get("content", "").strip()
            if name and content and name not in self.meta:
                self.meta[name] = content
        if tag == "a":
            href = data.get("href", "").strip()
            if href:
                self.links.append(href)
        if tag == "link" and "canonical" in data.get("rel", "").lower():
            href = data.get("href", "").strip()
            if href:
                self.meta["canonical"] = href

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._in_ld:
            self._in_ld = False
            return
        if tag in SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return
        if self._in_ld:
            self.json_ld_parts.append(data)
            return
        if self._skip_depth:
            return
        if self._in_title:
            self.title_parts.append(text)
            return
        self.text_parts.append(text)


def _fetch(url: str) -> tuple[str, str]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
        charset = resp.headers.get_content_charset() or "utf-8"
        html = raw.decode(charset, errors="replace")
        return html, resp.geturl()


def _normalize_url(url: str, base: str | None = None) -> str | None:
    if not url or url.startswith(("#", "javascript:", "data:", "mailto:")):
        return None
    absolute = urljoin(base or url, url)
    absolute, _frag = urldefrag(absolute)
    parts = urlsplit(absolute)
    if parts.scheme not in {"http", "https"}:
        return None
    path = parts.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return urlunsplit((parts.scheme, parts.netloc.lower(), path, "", ""))


def _same_host(a: str, b: str) -> bool:
    host_a = urlsplit(a).netloc.lower().removeprefix("www.")
    host_b = urlsplit(b).netloc.lower().removeprefix("www.")
    return host_a == host_b


def _should_skip_path(url: str) -> bool:
    path = urlsplit(url).path.lower()
    return any(part in path for part in SKIP_PATH_PARTS)


def _page_score(url: str) -> int:
    blob = (urlsplit(url).path + " " + urlsplit(url).query).lower()
    score = 10
    for hint, value in PRIORITY_HINTS:
        if hint in blob:
            score = max(score, value)
    if blob.rstrip("/") in {"", "/"}:
        score = max(score, 95)
    return score


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


def _clean_text(parts: list[str]) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for part in parts:
        line = re.sub(r"\s+", " ", part).strip()
        if len(line) < 2:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(line)
    text = "\n".join(lines)
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS].rsplit("\n", 1)[0] + "\n…"
    return text


def _parse_json_ld(chunks: list[str]) -> list[Any]:
    blobs: list[Any] = []
    for chunk in chunks:
        raw = htmlmod.unescape(chunk).strip()
        if not raw:
            continue
        try:
            blobs.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return blobs


def _walk_ld(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk_ld(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_ld(value)


def _ld_facts(blobs: list[Any]) -> dict[str, Any]:
    names: list[str] = []
    types: list[str] = []
    hours: list[str] = []
    phones: list[str] = []
    emails: list[str] = []
    addresses: list[str] = []
    for obj in _walk_ld(blobs):
        if not isinstance(obj, dict):
            continue
        raw_type = obj.get("@type")
        type_list: list[str] = []
        if isinstance(raw_type, str):
            type_list = [raw_type]
        elif isinstance(raw_type, list):
            type_list = [str(t) for t in raw_type]
        types.extend(t for t in type_list if t in KEEP_LD_TYPES)
        is_business = not type_list or any(t in KEEP_LD_TYPES for t in type_list)
        if not is_business:
            continue
        name = obj.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
        for key in ("openingHours", "openingHoursSpecification"):
            value = obj.get(key)
            if isinstance(value, str):
                hours.append(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        hours.append(item)
                    elif isinstance(item, dict):
                        days = item.get("dayOfWeek") or ""
                        opens = item.get("opens") or ""
                        closes = item.get("closes") or ""
                        if opens or closes:
                            hours.append(f"{days} {opens}-{closes}".strip())
        telephone = obj.get("telephone")
        if isinstance(telephone, str):
            phones.append(telephone)
        email = obj.get("email")
        if isinstance(email, str):
            emails.append(email)
        address = obj.get("address")
        if isinstance(address, str):
            addresses.append(address)
        elif isinstance(address, dict):
            parts = [
                address.get("streetAddress"),
                address.get("addressLocality"),
                address.get("addressRegion"),
                address.get("postalCode"),
                address.get("addressCountry"),
            ]
            joined = ", ".join(str(p) for p in parts if p)
            if joined:
                addresses.append(joined)
    return {
        "names": _unique(names),
        "types": _unique(types),
        "hours": _unique(hours),
        "phones": _unique(phones),
        "emails": _unique(emails),
        "addresses": _unique(addresses),
    }


def _normalize_phone(value: str) -> str:
    return re.sub(r"\s+", " ", unquote(value)).strip()


def _extract_contacts(text: str, html: str, base_url: str) -> dict[str, list[str]]:
    emails = [
        e for e in EMAIL_RE.findall(text)
        if not any(part in e.lower() for part in SKIP_EMAIL_PARTS)
    ]
    phones = [_normalize_phone(p) for p in PHONE_RE.findall(text)]
    social: list[str] = []
    whatsapp: list[str] = []
    for match in re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.I):
        lower = match.strip().lower()
        if lower.startswith("mailto:"):
            emails.append(match.split(":", 1)[1])
            continue
        if lower.startswith("tel:"):
            phones.append(_normalize_phone(match.split(":", 1)[1]))
            continue
        url = _normalize_url(match, base_url)
        if not url:
            continue
        host = urlsplit(url).netloc.lower()
        if any(host.endswith(s) or s in host for s in SOCIAL_HOSTS):
            if "wa.me" in host or "whatsapp" in host:
                whatsapp.append(url)
                digits = re.sub(r"\D", "", urlsplit(url).path)
                if digits.startswith("234") and len(digits) >= 13:
                    phones.append("+" + digits)
            else:
                social.append(url)
    return {
        "emails": _unique(emails),
        "phones": _unique(phones),
        "social": _unique(social),
        "whatsapp": _unique(whatsapp),
    }


def parse_page(html: str, page_url: str) -> dict[str, Any]:
    parser = PageParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        pass
    title = " ".join(parser.title_parts).strip()
    description = (
        parser.meta.get("description")
        or parser.meta.get("og:description")
        or ""
    ).strip()
    text = _clean_text(parser.text_parts)
    json_ld = _parse_json_ld(parser.json_ld_parts)
    contacts = _extract_contacts(text, html, page_url)
    links = []
    for href in parser.links:
        url = _normalize_url(href, page_url)
        if url:
            links.append(url)
    return {
        "url": page_url,
        "title": title or None,
        "description": description or None,
        "text": text or None,
        "links": _unique(links),
        "json_ld": json_ld,
        **contacts,
    }


def _discover_urls(start_url: str, first_page: dict[str, Any], max_pages: int) -> list[str]:
    ranked: list[tuple[int, str]] = []
    seen = {start_url}
    for url in first_page.get("links") or []:
        if url in seen or not _same_host(start_url, url) or _should_skip_path(url):
            continue
        seen.add(url)
        ranked.append((_page_score(url), url))
    ranked.sort(key=lambda item: item[0], reverse=True)
    urls = [start_url]
    for _score, url in ranked:
        if len(urls) >= max_pages:
            break
        urls.append(url)
    return urls


def _is_fact_line(line: str) -> bool:
    lower = line.lower().strip()
    if lower in {"address", "phone", "email", "socials", "contact"}:
        return True
    if EMAIL_RE.search(line) or PHONE_RE.search(line):
        return True
    if re.search(r"\d", line) and any(
        word in lower for word in ("close", "street", "road", "avenue", "lekki", "lagos", "phase")
    ):
        return True
    return False


def _strip_boilerplate(pages: list[dict[str, Any]]) -> None:
    from collections import Counter

    counts: Counter[str] = Counter()
    for page in pages:
        for line in (page.get("text") or "").split("\n"):
            if len(line) < 48 and not _is_fact_line(line):
                counts[line] += 1
    threshold = max(2, len(pages) // 2)
    boilerplate = {line for line, n in counts.items() if n >= threshold}
    for page in pages:
        text = page.get("text") or ""
        kept = [line for line in text.split("\n") if line not in boilerplate]
        page["text"] = "\n".join(kept).strip() or None


def _address_from_pages(pages: list[dict[str, Any]]) -> str | None:
    for page in pages:
        lines = [ln.strip() for ln in (page.get("text") or "").split("\n")]
        for i, line in enumerate(lines):
            if line.lower() == "address" and i + 1 < len(lines):
                chunk = [lines[i + 1]]
                if i + 2 < len(lines) and len(lines[i + 2]) < 80:
                    chunk.append(lines[i + 2])
                return ", ".join(chunk)
    return None


def scrape_website(url: str, *, max_pages: int = MAX_PAGES) -> dict[str, Any]:
    """Fetch a website and a few useful subpages; skip review-like paths."""
    start = _normalize_url(url) or url
    html, final_url = _fetch(start)
    start = _normalize_url(final_url) or final_url
    first = parse_page(html, start)
    targets = _discover_urls(start, first, max_pages=max_pages)

    pages = [first]
    seen_final = {start}
    for target in targets[1:]:
        try:
            page_html, page_final = _fetch(target)
        except (urllib.error.URLError, TimeoutError, ValueError):
            continue
        final = _normalize_url(page_final) or page_final
        if final in seen_final:
            continue
        seen_final.add(final)
        pages.append(parse_page(page_html, final))

    _strip_boilerplate(pages)
    ld = _ld_facts([blob for page in pages for blob in page.get("json_ld") or []])
    emails = _unique([e for page in pages for e in page.get("emails") or []] + ld["emails"])
    phones = _unique([p for page in pages for p in page.get("phones") or []] + ld["phones"])
    social = _unique([s for page in pages for s in page.get("social") or []])
    whatsapp = _unique([w for page in pages for w in page.get("whatsapp") or []])
    descriptions = _unique(
        [page.get("description") for page in pages if page.get("description")]
    )
    extra_hours = [
        d
        for d in descriptions
        if d
        and re.search(r"(?i)(?:\d\s*(?:am|pm)|(?:am|pm)\b)", d)
        and re.search(
            r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|–|-",
            d,
            re.I,
        )
    ]
    hours = _unique(list(ld["hours"]) + extra_hours)
    name = next(iter(ld["names"]), None)
    if not name:
        title = first.get("title") or ""
        name = re.sub(r"\s*[-|].*$", "", title).strip() or title or None

    slim_pages = [
        {
            "url": page["url"],
            "title": page.get("title"),
            "description": page.get("description"),
            "text": page.get("text"),
        }
        for page in pages
    ]
    return {
        "name": name,
        "website": start,
        "types": ld["types"],
        "description": descriptions[:8],
        "address": next(iter(ld["addresses"]), None) or _address_from_pages(pages),
        "emails": emails,
        "phones": phones,
        "whatsapp": whatsapp,
        "social": social,
        "hours": hours,
        "pages": slim_pages,
        "source_url": start,
    }


def format_text(profile: dict[str, Any]) -> str:
    lines = [
        profile.get("name") or "(website)",
        "=" * len(profile.get("name") or "(website)"),
    ]
    if profile.get("website"):
        lines.append(f"Website: {profile['website']}")
    if profile.get("types"):
        lines.append("Type: " + ", ".join(profile["types"]))
    if profile.get("address"):
        lines.append(f"Address: {profile['address']}")
    if profile.get("phones"):
        lines.append("Phone: " + ", ".join(profile["phones"]))
    if profile.get("emails"):
        lines.append("Email: " + ", ".join(profile["emails"]))
    if profile.get("whatsapp"):
        lines.append("WhatsApp: " + ", ".join(profile["whatsapp"]))
    if profile.get("social"):
        lines.append("Social: " + ", ".join(profile["social"]))
    if profile.get("hours"):
        lines.append("Hours:")
        for row in profile["hours"]:
            lines.append(f"  {row}")
    if profile.get("description"):
        lines.append("About:")
        for text in profile["description"]:
            lines.append(f"  {text}")
    if profile.get("pages"):
        lines.append("Pages:")
        for page in profile["pages"]:
            title = page.get("title") or page.get("url")
            lines.append(f"  {title}")
            lines.append(f"    {page.get('url')}")
            if page.get("description"):
                lines.append(f"    {page['description']}")
            if page.get("text"):
                snippet = page["text"].replace("\n", " ")
                if len(snippet) > 400:
                    snippet = snippet[:400] + "…"
                lines.append(f"    {snippet}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scrape a business website (reviews are skipped).")
    parser.add_argument("url", help="Website URL, e.g. https://theboroughlagos.com/")
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES)
    args = parser.parse_args(argv)
    try:
        profile = scrape_website(args.url, max_pages=args.max_pages)
    except Exception as exc:
        print(f"Failed to scrape website: {exc}", file=sys.stderr)
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
