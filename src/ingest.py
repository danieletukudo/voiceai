"""Ingest a document into the RAG vector store.

Example:
    from ingest import ingest
    ingest("docs/company_handbook.pdf")
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env.local")
load_dotenv(ROOT / ".env")
load_dotenv(ROOT.parent / ".env.local")
load_dotenv(ROOT.parent / ".env")

PERSIST_DIR = ROOT / "chat-engine-storage"
EMBED_MODEL = "gemini-embedding-001"
VISION_MODEL = "gemini-3.5-flash"
# Below this, treat extract as empty / image-based and use Gemini vision.
MIN_TEXT_CHARS = 50

MIME_BY_SUFFIX = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}

Settings.embed_model = GoogleGenAIEmbedding(
    model_name=EMBED_MODEL,
    api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
)


def _api_key() -> str | None:
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


def _combined_text(documents: list) -> str:
    return "\n".join((d.text or "").strip() for d in documents).strip()


def _transcribe_with_gemini(path: Path) -> str:
    """Use Gemini vision to turn an image/PDF chart into embeddable text."""
    from google import genai
    from google.genai import types

    api_key = _api_key()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY is required for vision ingest")

    mime = MIME_BY_SUFFIX.get(path.suffix.lower())
    if not mime:
        raise ValueError(f"No vision MIME type for {path.suffix}")

    client = genai.Client(api_key=api_key)
    prompt = (
        "Transcribe this document completely into plain text. "
        "If it is a chart or table, convert every row and column into markdown tables. "
        "Preserve all food names, cuts, doneness levels, temperatures, times, and notes exactly. "
        "Do not invent values that are not visible. Do not summarize — extract all readable content."
    )
    response = client.models.generate_content(
        model=VISION_MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=path.read_bytes(), mime_type=mime),
                    types.Part.from_text(text=prompt),
                ],
            )
        ],
    )
    text = (response.text or "").strip()
    if len(text) < MIN_TEXT_CHARS:
        raise RuntimeError(
            f"Vision model returned too little text ({len(text)} chars) for {path.name}"
        )
    return text


def _ensure_text_documents(path: Path, documents: list) -> list:
    """If text extract is empty, replace with Gemini vision transcription."""
    if len(_combined_text(documents)) >= MIN_TEXT_CHARS:
        return documents

    if path.suffix.lower() not in MIME_BY_SUFFIX:
        return documents

    print(f"Sparse text in {path.name}; transcribing with {VISION_MODEL}...")
    text = _transcribe_with_gemini(path)
    meta = dict(documents[0].metadata) if documents else {}
    meta.update(
        {
            "file_name": path.name,
            "file_path": str(path),
            "transcription": VISION_MODEL,
        }
    )
    return [Document(text=text, metadata=meta)]


def _purge_same_file(index: VectorStoreIndex, file_name: str) -> int:
    """Drop prior chunks for this filename so re-ingest replaces, not duplicates."""
    to_delete = []
    for ref_doc_id, info in (index.ref_doc_info or {}).items():
        meta = info.metadata or {}
        if meta.get("file_name") == file_name or meta.get("filename") == file_name:
            to_delete.append(ref_doc_id)
    for ref_doc_id in to_delete:
        index.delete_ref_doc(ref_doc_id, delete_from_docstore=True)
    return len(to_delete)


def remove_from_index(file_name: str) -> int:
    """Remove all RAG chunks for a document and persist. Returns how many refs deleted."""
    if not (PERSIST_DIR / "docstore.json").exists():
        return 0

    storage = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
    index = load_index_from_storage(storage)
    removed = _purge_same_file(index, file_name)
    index.storage_context.persist(persist_dir=str(PERSIST_DIR))
    try:
        from rag import invalidate_index

        invalidate_index()
    except Exception:
        pass
    return removed


def ingest(file_path: str | Path) -> None:
    """Read one file (PDF, CSV, TXT, …) and add it to the vector store."""
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    documents = SimpleDirectoryReader(input_files=[str(path)]).load_data()
    documents = _ensure_text_documents(path, documents)

    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    has_index = (PERSIST_DIR / "docstore.json").exists()

    if has_index:
        storage = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
        index = load_index_from_storage(storage)
        _purge_same_file(index, path.name)
        for doc in documents:
            index.insert(doc)
    else:
        index = VectorStoreIndex.from_documents(documents)

    index.storage_context.persist(persist_dir=str(PERSIST_DIR))
    try:
        from rag import invalidate_index

        invalidate_index()
    except Exception:
        pass
    print(f"Ingested {path.name} -> {PERSIST_DIR} ({len(_combined_text(documents))} chars)")


if __name__ == "__main__":
    ingest(
        ROOT
        / "docs"
        / "SousVideCookingTimes_Chart_229dc15b-0dfd-4fbd-bfce-fbad588e986e_2048x2048 (1).pdf"
    )
