import asyncio
import os
from typing import Any

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from qdrant_client import AsyncQdrantClient
from sentence_transformers import SentenceTransformer

from rag_config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    QDRANT_API_KEY,
    QDRANT_URL,
    TOP_K,
)


async def retrieve_chunks(
    question: str,
    embedding_model: SentenceTransformer,
    qdrant_client: AsyncQdrantClient,
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    loop = asyncio.get_event_loop()
    query_vector = await loop.run_in_executor(
        None,
        lambda: embedding_model.encode([question])[0].tolist(),
    )

    search_result = await qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    hits = search_result if isinstance(search_result, list) else getattr(search_result, "points", [])
    return [hit.payload for hit in hits if hit.payload]


async def main() -> None:
    if not QDRANT_URL:
        raise SystemExit("QDRANT_URL is not set in .env")

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    _ = embedding_model.encode("warmup")

    print("Connecting to Qdrant...")
    qdrant_client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)

    try:
        info = await qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        point_count = info.points_count or 0
        if point_count == 0:
            print(f"Collection '{COLLECTION_NAME}' is empty. Ingest a PDF first:")
            print("  ./venv/bin/python ingest_document.py path/to/file.pdf")
            return

        print(f"Document search ready ({point_count} chunks indexed).")
        print("Ask questions about your ingested documents. Type 'exit' to quit.\n")

        while True:
            question = input("Question: ").strip()
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                break

            chunks = await retrieve_chunks(question, embedding_model, qdrant_client)
            if not chunks:
                print("No relevant document chunks found.\n")
                continue

            print(f"\nFound {len(chunks)} relevant passages:\n")
            for i, chunk in enumerate(chunks, start=1):
                print(f"--- Result {i} ---")
                print(f"Source: {chunk.get('source_file')} (page {chunk.get('page')})")
                print(f"{chunk.get('text', '')}\n")
    finally:
        await qdrant_client.close()


if __name__ == "__main__":
    asyncio.run(main())
