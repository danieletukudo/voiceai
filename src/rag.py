"""Knowledge helpers used by the FastAPI Knowledge API."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from company_pack import answer_from_pack, unknown_answer

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env.local")
load_dotenv(ROOT / ".env")
load_dotenv(ROOT.parent / ".env.local")
load_dotenv(ROOT.parent / ".env")

_index = None


def invalidate_index() -> None:
    """Drop any cached index so the next ingest reloads from disk."""
    global _index
    _index = None


async def ask(question: str) -> dict:
    """Answer from the company pack. Unknown questions stay fast and in staff voice."""
    question = (question or "").strip()
    if not question:
        return {
            "answer": "I did not catch a question. Could you please repeat that?",
            "sources": [],
            "via": "pack",
        }

    packed = answer_from_pack(question)
    if packed and packed.get("answer"):
        return packed

    return unknown_answer()
