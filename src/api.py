"""Knowledge API: ingest documents and answer questions via RAG."""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
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

API_HOST = os.getenv("API_HOST", "0.0.0.0")
# Render injects PORT; fall back to API_PORT, then 8000.
API_PORT = int(os.getenv("PORT") or os.getenv("API_PORT", "8000"))

app = FastAPI(title="Knowledge API")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    company_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[dict] = Field(default_factory=list)


def _safe_path(name: str) -> Path:
    path = (DOCS / Path(name).name).resolve()
    if path.parent != DOCS.resolve():
        raise HTTPException(400, "Invalid filename")
    return path


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
        {"name": f.name, "size": f.stat().st_size}
        for f in sorted(DOCS.iterdir())
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


@app.delete("/api/files/{name}")
def delete_file(name: str):
    path = _safe_path(name)
    if not path.is_file():
        raise HTTPException(404, "File not found")
    path.unlink()
    return {"ok": True, "name": path.name}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
