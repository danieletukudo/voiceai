import argparse
import os
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pypdf import PdfReader
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from rag_config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    QDRANT_API_KEY,
    QDRANT_URL,
    VECTOR_SIZE,
)

TEXT_SUFFIXES = {".txt", ".md", ".csv", ".json", ".log"}


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    _ = model.encode("warmup")
    print("Embedding model ready.")
    return model


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    if not QDRANT_URL:
        raise RuntimeError("QDRANT_URL is not set in .env")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    ensure_collection(client)
    return client


def extract_text_from_pdf(pdf_path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        if text.strip():
            pages.append((page_num, text))
    return pages


def extract_text_from_file(path: Path) -> list[tuple[int, str]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix in TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        return [(1, text)] if text else []
    raise ValueError(f"Unsupported file type: {suffix or path.name}")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def ensure_collection(client: QdrantClient) -> None:
    try:
        client.get_collection(collection_name=COLLECTION_NAME)
    except Exception:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )

    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="source_file",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    except Exception:
        pass


def delete_existing_chunks(client: QdrantClient, source_file: str) -> int:
    try:
        result = client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_file",
                            match=models.MatchValue(value=source_file),
                        )
                    ]
                )
            ),
        )
        return getattr(result, "operation_id", 0) or 0
    except Exception as e:
        print(f"Could not remove old chunks for '{source_file}': {e}")
        return 0


def ingest_pages(
    pages: list[tuple[int, str]],
    *,
    source_file: str,
    extra_payload: dict[str, Any] | None = None,
    embedding_model: SentenceTransformer | None = None,
    qdrant_client: QdrantClient | None = None,
) -> int:
    if not pages:
        raise ValueError(f"No text to ingest for '{source_file}'.")

    model = embedding_model or get_embedding_model()
    client = qdrant_client or get_qdrant_client()
    delete_existing_chunks(client, source_file)

    points = []
    chunk_index = 0
    for page_num, page_text in pages:
        for page_chunk in chunk_text(page_text):
            embedding = model.encode([page_chunk])[0]
            payload = {
                "text": page_chunk,
                "source_file": source_file,
                "page": page_num,
                "chunk_index": chunk_index,
            }
            if extra_payload:
                payload.update(extra_payload)
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload=payload,
                )
            )
            chunk_index += 1

    if not points:
        raise ValueError(f"No chunks produced for '{source_file}'.")

    client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
    return len(points)


def ingest_text(
    text: str,
    *,
    source_file: str,
    source_type: str | None = None,
    source_url: str | None = None,
) -> int:
    pages = [(1, text.strip())] if text and text.strip() else []
    extra: dict[str, Any] = {}
    if source_type:
        extra["source_type"] = source_type
    if source_url:
        extra["source_url"] = source_url
    return ingest_pages(pages, source_file=source_file, extra_payload=extra or None)


def ingest_file(path: Path) -> int:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    pages = extract_text_from_file(path)
    return ingest_pages(
        pages,
        source_file=path.name,
        extra_payload={"source_type": "file"},
    )


def remove_from_index(source_file: str) -> int:
    client = get_qdrant_client()
    delete_existing_chunks(client, source_file)
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a document into the RAG collection.")
    parser.add_argument("file_path", help="Path to the PDF or text file to ingest")
    args = parser.parse_args()

    path = Path(args.file_path).expanduser().resolve()
    count = ingest_file(path)
    print(f"Ingested '{path.name}' ({count} chunks).")


if __name__ == "__main__":
    main()
