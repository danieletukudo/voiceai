"""Knowledge / RAG helpers used by the FastAPI Knowledge API."""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

logger = logging.getLogger("rag")

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env.local")
load_dotenv(ROOT / ".env")
load_dotenv(ROOT.parent / ".env.local")
load_dotenv(ROOT.parent / ".env")

PERSIST_DIR = ROOT / "chat-engine-storage"
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

_index = None


def invalidate_index() -> None:
    """Drop the in-memory index so the next ask()/ingest reload from disk."""
    global _index
    _index = None


def get_index():
    """Load the persisted vector index once per process."""
    global _index
    if _index is None:
        if not (PERSIST_DIR / "docstore.json").exists():
            raise FileNotFoundError(
                f"No RAG index at {PERSIST_DIR}. Upload documents via POST /api/files first."
            )
        logger.info("Loading RAG index from %s", PERSIST_DIR)
        Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL)
        Settings.llm = OpenAI(model=LLM_MODEL)
        storage_context = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
        _index = load_index_from_storage(storage_context)
    return _index


async def ask(question: str, *, top_k: int = 4) -> dict:
    """Retrieve context and generate a final voice-friendly answer."""
    question = (question or "").strip()
    if not question:
        return {
            "answer": "I did not catch a question. Could you please repeat that?",
            "sources": [],
        }

    index = get_index()
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = await retriever.aretrieve(question)
    sources = []
    for node in nodes:
        meta = node.node.metadata or {}
        sources.append(
            {
                "text": node.node.get_content()[:500],
                "score": float(node.score) if node.score is not None else None,
                "document": meta.get("file_name") or meta.get("filename"),
                "page": meta.get("page_label") or meta.get("page"),
            }
        )

    if not nodes:
        return {
            "answer": (
                "I could not find that in the company knowledge base. "
                "Please ask about something covered in our documents, "
                "or I can connect you with a human agent."
            ),
            "sources": [],
        }

    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        use_async=True,
    )
    response = await query_engine.aquery(
        (
            "Answer clearly in one to three short spoken sentences. "
            "Use only the retrieved company documents. "
            "If the documents do not contain the answer, say so briefly.\n\n"
            f"Question: {question}"
        )
    )
    return {"answer": str(response).strip(), "sources": sources}
