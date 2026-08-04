"""Knowledge API: ingest documents, answer questions via RAG, serve call transcripts."""

import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from ingest import ingest
from rag import ask, invalidate_index

ROOT = Path(__file__).resolve().parent.parent
# Prefer package .env, then monorepo root .env.local
load_dotenv(ROOT / ".env.local")
load_dotenv(ROOT / ".env")
load_dotenv(ROOT.parent / ".env.local")
load_dotenv(ROOT.parent / ".env")

DOCS = ROOT / "docs"
STATIC = ROOT / "static"
DOCS.mkdir(parents=True, exist_ok=True)

# Call transcripts written by the voice agent (Agent: / Person:).
TRANSCRIPTS = Path(
    os.getenv("TRANSCRIPTS_DIR", str(ROOT / "transcripts"))
).expanduser().resolve()
TRANSCRIPTS.mkdir(parents=True, exist_ok=True)

API_HOST = os.getenv("API_HOST", "0.0.0.0")
# Render injects PORT; fall back to API_PORT, then 8000.
API_PORT = int(os.getenv("PORT") or os.getenv("API_PORT", "8000"))

app = FastAPI(title="Knowledge API")


def _bootstrap_rag_from_docs() -> None:
    """On a fresh host (empty chat-engine-storage), ingest every file in docs/.

    Render's disk is ephemeral: after each deploy the vector store is gone even
    when seed documents are copied into the image. Without this, the Library
    lists files but /ask has nothing to retrieve.
    """
    persist = ROOT / "chat-engine-storage"
    if (persist / "docstore.json").exists():
        return
    seed_files = [
        f
        for f in sorted(DOCS.iterdir())
        if f.is_file() and not f.name.startswith(".")
    ]
    if not seed_files:
        return
    for path in seed_files:
        try:
            ingest(path)
        except Exception:
            # Missing API key / bad file should not prevent the API from booting.
            print(f"WARN: bootstrap ingest failed for {path.name}", flush=True)
    invalidate_index()


_bootstrap_rag_from_docs()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    company_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[dict] = Field(default_factory=list)


class TranscriptInfo(BaseModel):
    name: str
    size: int
    modified: str


class TranscriptDetail(TranscriptInfo):
    content: str


def _safe_path(name: str) -> Path:
    path = (DOCS / Path(name).name).resolve()
    if path.parent != DOCS.resolve():
        raise HTTPException(400, "Invalid filename")
    return path


def _safe_transcript_path(name: str) -> Path:
    path = (TRANSCRIPTS / Path(name).name).resolve()
    if path.parent != TRANSCRIPTS.resolve():
        raise HTTPException(400, "Invalid filename")
    return path


def _iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


@app.get("/health")
@app.head("/health")
def health():
    return {"ok": True}


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.head("/")
def home_head():
    return {"ok": True}


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(body: AskRequest):
    """Answer a question using the ingested knowledge base."""
    try:
        result = await ask(body.question)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Knowledge query failed: {exc}") from exc
    return AskResponse(**result)


@app.get("/api/files")
def list_files():
    return [
        {
            "name": f.name,
            "size": f.stat().st_size,
            "modified": _iso_mtime(f),
        }
        for f in sorted(DOCS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if f.is_file() and not f.name.startswith(".")
    ]


@app.post("/api/files")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No filename")
    path = _safe_path(file.filename)
    path.write_bytes(await file.read())
    try:
        ingest(path)
    except Exception as exc:
        raise HTTPException(500, f"Ingest failed: {exc}") from exc
    invalidate_index()
    return {"ok": True, "name": path.name}


@app.get("/api/files/{name}")
def get_file_meta(name: str):
    """Return file metadata (and text content for plain-text docs)."""
    path = _safe_path(name)
    if not path.is_file():
        raise HTTPException(404, "File not found")
    payload: dict = {
        "name": path.name,
        "size": path.stat().st_size,
        "modified": _iso_mtime(path),
    }
    if path.suffix.lower() in {".txt", ".md", ".csv", ".json", ".log"}:
        payload["content"] = path.read_text(encoding="utf-8", errors="replace")
    return payload


@app.get("/api/files/{name}/download")
def download_file(name: str):
    """Download a knowledge-base document."""
    path = _safe_path(name)
    if not path.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(
        path,
        filename=path.name,
        content_disposition_type="attachment",
    )


@app.delete("/api/files/{name}")
def delete_file(name: str):
    path = _safe_path(name)
    if not path.is_file():
        raise HTTPException(404, "File not found")
    path.unlink()
    return {"ok": True, "name": path.name}


@app.get("/api/transcripts", response_model=list[TranscriptInfo])
def list_transcripts():
    """List all call transcript .txt files, newest first."""
    if not TRANSCRIPTS.is_dir():
        return []
    files = [
        f
        for f in TRANSCRIPTS.iterdir()
        if f.is_file() and not f.name.startswith(".") and f.suffix.lower() == ".txt"
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [
        TranscriptInfo(name=f.name, size=f.stat().st_size, modified=_iso_mtime(f))
        for f in files
    ]


@app.post("/api/transcripts")
async def upload_transcript(file: UploadFile = File(...)):
    """Accept a call transcript from the voice agent (or any client)."""
    if not file.filename:
        raise HTTPException(400, "No filename")
    filename = Path(file.filename).name
    if not filename.lower().endswith(".txt"):
        filename = f"{filename}.txt"
    path = _safe_transcript_path(filename)
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty transcript")
    path.write_bytes(data)
    return {
        "ok": True,
        "name": path.name,
        "size": path.stat().st_size,
        "modified": _iso_mtime(path),
    }


@app.get("/api/transcripts/{name}", response_model=TranscriptDetail)
def get_transcript(name: str):
    """Return transcript metadata and full text for display."""
    path = _safe_transcript_path(name)
    if not path.is_file():
        raise HTTPException(404, "Transcript not found")
    return TranscriptDetail(
        name=path.name,
        size=path.stat().st_size,
        modified=_iso_mtime(path),
        content=path.read_text(encoding="utf-8"),
    )


@app.get("/api/transcripts/{name}/download")
def download_transcript(name: str):
    """Download a transcript as a plain-text attachment."""
    path = _safe_transcript_path(name)
    if not path.is_file():
        raise HTTPException(404, "Transcript not found")
    return FileResponse(
        path,
        media_type="text/plain; charset=utf-8",
        filename=path.name,
        content_disposition_type="attachment",
    )


@app.get("/api/transcripts/{name}/raw", response_class=PlainTextResponse)
def raw_transcript(name: str):
    """Raw transcript body (handy for curl / preview)."""
    path = _safe_transcript_path(name)
    if not path.is_file():
        raise HTTPException(404, "Transcript not found")
    return path.read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
