"""Generic company knowledge pack: built at ingest, used first on /ask."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PACK_PATH = ROOT / "chat-engine-storage" / "company_pack.json"
LLM_MODEL = "gemini-3.5-flash"
MAX_SOURCE_CHARS = 24000
MAX_CORPUS_CHARS = 70000
FAQ_SCORE_MIN = 0.36

_STOP = {
    "a", "an", "the", "and", "or", "to", "of", "for", "in", "on", "at", "is",
    "are", "do", "you", "your", "we", "our", "me", "please", "what", "whats",
    "when", "where", "how", "can", "i", "about", "this", "that", "with",
    "have", "has", "had", "must", "does", "did", "will", "would", "should",
    "could", "may", "might", "not", "any", "all", "from", "into", "who",
    "which", "their", "they", "them", "its", "be", "been", "being", "as",
    "by", "if", "it", "my", "no", "yes", "also", "only", "than", "then",
    "there", "these", "those", "was", "were", "so", "such", "just", "like",
    "under", "over", "out", "up", "down", "other", "through", "during",
    "after", "before", "between", "each", "few", "more", "most", "some",
    "too", "very", "own", "same", "both", "but",
}

_ALIASES: dict[str, frozenset[str]] = {
    "cto": frozenset({"cto", "chief", "technology", "officer"}),
    "ceo": frozenset({"ceo", "chief", "executive", "officer", "founder"}),
    "cfo": frozenset({"cfo", "chief", "financial", "officer"}),
    "coo": frozenset({"coo", "chief", "operating", "officer"}),
    "task": frozenset({
        "task", "tasks", "duty", "duties", "responsibility",
        "responsibilities", "deliverable", "deliverables",
    }),
    "tasks": frozenset({
        "task", "tasks", "duty", "duties", "responsibility",
        "responsibilities", "deliverable", "deliverables",
    }),
    "duty": frozenset({"task", "tasks", "duty", "duties", "responsibility", "responsibilities"}),
    "duties": frozenset({"task", "tasks", "duty", "duties", "responsibility", "responsibilities"}),
}

_DUTY_WORDS = frozenset({
    "task", "tasks", "duty", "duties", "responsibility", "responsibilities",
    "deliverable", "deliverables",
})

_TITLE_PHRASES = (
    ("chief technology officer", "cto"),
    ("chief executive officer", "ceo"),
    ("chief financial officer", "cfo"),
    ("chief operating officer", "coo"),
    ("technical development lead", "lead"),
)

_DUTY_PHRASES = (
    ("what is the task of", "what does"),
    ("what are the tasks of", "what does"),
    ("what is the duty of", "what does"),
    ("what are the duties of", "what does"),
    ("what are the responsibilities of", "what does"),
    ("what is the role of", "what does"),
)

_INTENT_FACTS: list[tuple[str, tuple[str, ...]]] = [
    ("hours", ("hour", "hours", "open", "opens", "opening", "close",
               "closes", "closing", "time", "times", "schedule")),
    ("address", ("where", "address", "located", "location", "place",
                 "direction", "directions", "find", "map")),
    ("phones", ("phone", "call", "number", "telephone", "whatsapp",
                "mobile", "contact")),
    ("emails", ("email", "e-mail", "mail")),
    ("website", ("website", "site", "web", "url", "online")),
    ("offerings", ("offer", "offers", "offering", "service", "services",
                   "product", "products", "available", "price", "prices",
                   "menu", "account", "accounts", "course", "courses")),
]

_pack_cache: dict[str, Any] | None = None

EXTRACT_PROMPT = """You are preparing a phone receptionist's brief from source text.
The source may be a company website, Google listing, contract, job description,
policy, handbook, school brochure, bank FAQ, or mixed documents.
Do not assume it is a hotel. Infer the organization type only from the text.

Return JSON only, no markdown, with this shape:
{
  "name": "organization or document subject name",
  "kind": "short label such as bank, school, restaurant, hotel, contract, employer",
  "summary": "2-4 spoken sentences describing who they are",
  "facts": {
    "address": "full address or null",
    "hours": "opening hours as one string or null",
    "phones": ["phone numbers"],
    "emails": ["emails"],
    "website": "url or null",
    "offerings": ["each distinct product, room, course, account, service, or deliverable"],
    "other": ["roles, duties, parties, fees, dates, and other caller-useful facts"]
  },
  "faqs": [
    {"q": "a natural spoken question", "a": "1-3 short spoken sentences", "topics": ["hours"]}
  ]
}

Rules:
- Use only the source text. Never invent facts.
- Write 15-40 FAQs a real caller or staff member would ask about THIS text.
- Cover every important named role or title (CTO, CEO, lead, teacher, and so on).
- For each role, include both "What does the CTO do?" style questions AND the
  full title. Spell out duties, tasks, and responsibilities from the text.
  If a title is not actually appointed, say that, then state whose duties apply.
- Also cover hours, location, offerings, booking, and reservations when those facts appear.
- Answers must be ready to speak on a phone call, in first person as staff
  (we / our / I). Never mention documents, sources, files, or a knowledge base.
- If something is unknown, use null or omit it.
"""


def _api_key() -> str | None:
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if w not in _STOP and len(w) > 1}


def _expand(tokens: set[str]) -> set[str]:
    out = set(tokens)
    for word in tokens:
        aliases = _ALIASES.get(word)
        if aliases:
            out |= aliases
    if {"chief", "technology", "officer"} <= out:
        out.add("cto")
    if {"chief", "executive", "officer"} <= out:
        out.add("ceo")
    return out


def _norm_question(text: str) -> str:
    t = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
    t = re.sub(r"\s+", " ", t).strip()
    for long, short in _TITLE_PHRASES:
        t = t.replace(long, short)
    for long, short in _DUTY_PHRASES:
        t = t.replace(long, short)
    return t


def invalidate_pack() -> None:
    global _pack_cache
    _pack_cache = None


def load_pack() -> dict[str, Any] | None:
    global _pack_cache
    if _pack_cache is not None:
        return _pack_cache
    if not PACK_PATH.is_file():
        return None
    try:
        data = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    _pack_cache = data
    return data


def save_pack(pack: dict[str, Any]) -> None:
    global _pack_cache
    PACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    pack["updated_at"] = datetime.now(timezone.utc).isoformat()
    PACK_PATH.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")
    _pack_cache = pack


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Model did not return JSON")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("Pack JSON must be an object")
    return data


def _concat_corpus(corpus: list[dict[str, Any]]) -> str:
    parts = []
    used = 0
    for item in corpus:
        name = item.get("name") or "source"
        body = (item.get("text") or "").strip()
        if not body:
            continue
        chunk = f"\n\n## {name}\n{body}"
        if used + len(chunk) > MAX_CORPUS_CHARS:
            remain = MAX_CORPUS_CHARS - used
            if remain > 500:
                parts.append(chunk[:remain])
            break
        parts.append(chunk)
        used += len(chunk)
    return "".join(parts).strip()


def extract_pack_from_text(source_text: str) -> dict[str, Any]:
    api_key = _api_key()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is required to build the company pack")
    body = (source_text or "").strip()
    if len(body) < 40:
        raise ValueError("Not enough company text to build a pack")

    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=(
            EXTRACT_PROMPT
            + "\n\nSOURCE TEXT:\n"
            + body[:MAX_CORPUS_CHARS]
        ),
    )
    raw = (response.text or "").strip()
    pack = _parse_json_object(raw)
    facts = pack.get("facts") if isinstance(pack.get("facts"), dict) else {}
    faqs = pack.get("faqs") if isinstance(pack.get("faqs"), list) else []
    clean_faqs = []
    for item in faqs:
        if not isinstance(item, dict):
            continue
        q = str(item.get("q") or "").strip()
        a = str(item.get("a") or "").strip()
        if q and a:
            topics = item.get("topics") if isinstance(item.get("topics"), list) else []
            clean_faqs.append({"q": q, "a": a, "topics": [str(t) for t in topics]})
    pack["facts"] = facts
    pack["faqs"] = clean_faqs[:40]
    return pack


def rebuild_corpus(entries: list[tuple[str, str]]) -> dict[str, Any]:
    """Replace the pack corpus with these texts and rebuild FAQs via the LLM."""
    corpus: list[dict[str, Any]] = []
    for name, text in entries:
        body = (text or "").strip()
        if not name or not body:
            continue
        corpus.append({"name": name, "text": body[:MAX_SOURCE_CHARS]})
    if not corpus:
        raise ValueError("No text available to build the company pack")
    combined = _concat_corpus(corpus)
    pack = extract_pack_from_text(combined)
    pack["corpus"] = corpus
    save_pack(pack)
    return pack


def upsert_corpus_and_rebuild(entries: list[tuple[str, str]]) -> dict[str, Any]:
    """Rebuild the pack from the provided sources only (no stale leftover company)."""
    return rebuild_corpus(entries)


def rebuild_from_docs_dir(docs_dir: Path) -> dict[str, Any]:
    """Read every file in the library and rebuild the pack from that text."""
    from ingest import load_file_documents

    entries: list[tuple[str, str]] = []
    for path in sorted(docs_dir.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        docs = load_file_documents(path)
        text = "\n".join((d.text or "") for d in docs)
        entries.append((path.name, text))
    return rebuild_corpus(entries)


def drop_corpus_entry(name: str) -> None:
    current = load_pack()
    if not current:
        return
    corpus = [
        item
        for item in (current.get("corpus") or [])
        if isinstance(item, dict) and item.get("name") != name
    ]
    if not corpus:
        if PACK_PATH.is_file():
            PACK_PATH.unlink()
        invalidate_pack()
        return
    combined = _concat_corpus(corpus)
    try:
        pack = extract_pack_from_text(combined)
    except Exception:
        current["corpus"] = corpus
        save_pack(current)
        return
    pack["corpus"] = corpus
    save_pack(pack)


def _fact_value(facts: dict[str, Any], key: str) -> str | None:
    value = facts.get(key)
    if value in (None, "", [], {}):
        return None
    if isinstance(value, list):
        items = [str(v).strip() for v in value if str(v).strip()]
        return ", ".join(items[:8]) if items else None
    text = str(value).strip()
    return text or None


def _spoken_fact(key: str, value: str) -> str:
    complete = value[:1].isupper() and value.endswith((".", "!"))
    if key == "hours":
        return value if complete else f"Our hours are {value}."
    if key == "address":
        return value if complete else f"We are at {value}."
    if key == "phones":
        return f"You can call us on {value}."
    if key == "emails":
        return f"Our email is {value}."
    if key == "website":
        return f"Our website is {value}."
    if key == "offerings":
        return value if complete else f"We offer {value}."
    return value


def _faq_score(question: str, faq: dict[str, Any]) -> float:
    q_raw = _tokens(question)
    if not q_raw:
        return 0.0
    q_tokens = _expand(q_raw)
    faq_q = _expand(_tokens(str(faq.get("q") or "")))
    faq_a = _expand(_tokens(str(faq.get("a") or "")))
    if not faq_q:
        return 0.0

    overlap_q = q_tokens & faq_q
    distinctive = {t for t in q_raw if len(t) >= 4}
    if len(overlap_q) < 2 and not (distinctive & faq_q):
        return 0.0

    sim = SequenceMatcher(
        None,
        _norm_question(question),
        _norm_question(str(faq.get("q") or "")),
    ).ratio()
    recall = len(overlap_q) / max(len(q_raw), 1)
    precision = len(overlap_q) / max(len(faq_q), 1)
    score = (0.55 * sim) + (0.30 * min(recall, 1.0)) + (0.15 * precision)

    overlap_a = q_tokens & faq_a
    if overlap_q and overlap_a:
        score += 0.03 * min(len(overlap_a), 3)

    q_duty = bool(q_raw & _DUTY_WORDS) or bool(q_tokens & _DUTY_WORDS)
    f_duty = bool((faq_q | faq_a) & _DUTY_WORDS)
    if q_duty and f_duty:
        score += 0.12
    elif q_duty and not f_duty:
        score -= 0.2

    return max(0.0, min(score, 1.0))


def unknown_answer() -> dict[str, Any]:
    """Spoken miss for a live call: staff voice, no model, no document talk."""
    pack = load_pack()
    name = str((pack or {}).get("name") or "").strip()
    kind = str((pack or {}).get("kind") or "").lower()
    looks_like_doc = kind in {"contract", "agreement", "document"} or any(
        hint in name.lower()
        for hint in ("agreement", "contract", "policy", "handbook", ".pdf")
    )
    if name and not looks_like_doc:
        answer = (
            "We do not have that information. "
            f"I can connect you with someone at {name} who can help."
        )
    else:
        answer = (
            "We do not have that information. "
            "I can connect you with a colleague who can help."
        )
    return {"answer": answer, "via": "pack", "sources": []}


def answer_from_pack(question: str) -> dict[str, Any] | None:
    """Return a fast spoken answer from the pack, or None if nothing matches."""
    pack = load_pack()
    if not pack:
        return None
    q = (question or "").strip()
    if not q:
        return None
    q_tokens = _tokens(q)
    facts = pack.get("facts") if isinstance(pack.get("facts"), dict) else {}

    best: dict[str, Any] | None = None
    best_score = 0.0
    for faq in pack.get("faqs") or []:
        if not isinstance(faq, dict):
            continue
        score = _faq_score(q, faq)
        if score > best_score:
            best_score = score
            best = faq
    if best and best_score >= FAQ_SCORE_MIN:
        return {
            "answer": str(best.get("a") or "").strip(),
            "via": "pack",
            "sources": [
                {
                    "text": str(best.get("q") or "")[:500],
                    "score": best_score,
                    "document": "company-pack",
                    "page": "faq",
                }
            ],
        }

    for fact_key, cues in _INTENT_FACTS:
        if not (q_tokens & set(cues)):
            continue
        value = _fact_value(facts, fact_key)
        if not value:
            continue
        if fact_key == "offerings":
            generic = q_tokens & {
                "offer", "offers", "offering", "service", "services",
                "product", "products",
            }
            if not generic and not (_tokens(value) & q_tokens):
                continue
        return {
            "answer": _spoken_fact(fact_key, value),
            "via": "pack",
            "sources": [
                {
                    "text": value[:500],
                    "score": 1.0,
                    "document": "company-pack",
                    "page": fact_key,
                }
            ],
        }
    return None
