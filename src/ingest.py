"""Ingest a document into the RAG vector store.

Example:
    from ingest import ingest
    ingest("docs/company_handbook.pdf")
"""

from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.openai import OpenAIEmbedding

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env.local")
load_dotenv(ROOT / ".env")
load_dotenv(ROOT.parent / ".env.local")
load_dotenv(ROOT.parent / ".env")

PERSIST_DIR = ROOT / "chat-engine-storage"
EMBED_MODEL = "text-embedding-3-small"
Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL)


def ingest(file_path: str | Path) -> None:
    """Read one file (PDF, CSV, TXT, …) and add it to the vector store."""
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")


    documents = SimpleDirectoryReader(input_files=[str(path)]).load_data()

    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    has_index = (PERSIST_DIR / "docstore.json").exists()

    if has_index:
        storage = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
        index = load_index_from_storage(storage)
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
    print(f"Ingested {path.name} -> {PERSIST_DIR}")


if __name__ == "__main__":
    # Change this path (or call ingest() from another script / REPL)
    ingest(ROOT / "docs" / "week6_api_microservices_mcp.pdf")



