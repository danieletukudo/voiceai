"""Knowledge API: ingest documents and answer questions via RAG."""

import json
import os
import re
from datetime import datetime, timezone
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
RESPONSES = ROOT / "responses"
STATIC = ROOT / "static"
DOCS.mkdir(parents=True, exist_ok=True)
RESPONSES.mkdir(parents=True, exist_ok=True)

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


class QAExchange(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)


class CallResponseRequest(BaseModel):
    """JSON payload sent by the voice agent when a call ends."""

    exchanges: list[QAExchange] = Field(..., min_length=1)
    room: str | None = None
    agent_name: str | None = None
    started_at: str | None = None
    ended_at: str | None = None


def _safe_path(name: str) -> Path:
    path = (DOCS / Path(name).name).resolve()
    if path.parent != DOCS.resolve():
        raise HTTPException(400, "Invalid filename")
    return path


def _safe_response_path(name: str) -> Path:
    path = (RESPONSES / Path(name).name).resolve()
    if path.parent != RESPONSES.resolve():
        raise HTTPException(400, "Invalid filename")
    return path


def _slug(value: str | None, fallback: str = "call") -> str:
    text = (value or fallback).strip().lower()
    text = re.sub(r"[^a-z0-9_-]+", "-", text).strip("-")
    return text[:48] or fallback


def _format_transcript_text(body: CallResponseRequest, saved_at: str) -> str:
    lines = [
        "Call transcript",
        f"Saved at: {saved_at}",
    ]
    if body.room:
        lines.append(f"Room: {body.room}")
    if body.agent_name:
        lines.append(f"Agent: {body.agent_name}")
    if body.started_at:
        lines.append(f"Started: {body.started_at}")
    if body.ended_at:
        lines.append(f"Ended: {body.ended_at}")
    lines.append("")
    for i, exchange in enumerate(body.exchanges, start=1):
        lines.append(f"--- Exchange {i} ---")
        lines.append(f"Q: {exchange.question}")
        lines.append(f"A: {exchange.answer}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


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


@app.post("/api/response")
async def save_call_response(body: CallResponseRequest):
    """Accept call Q&A JSON from the voice agent and store it as a downloadable text file."""
    saved_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filename = f"call-{_slug(body.room)}-{stamp}.txt"
    path = _safe_response_path(filename)

    path.write_text(_format_transcript_text(body, saved_at), encoding="utf-8")

    # Also keep the raw JSON next to the text file for debugging / re-import.
    json_path = path.with_suffix(".json")
    json_path.write_text(
        json.dumps(body.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {"ok": True, "name": path.name, "json_name": json_path.name}


@app.get("/api/responses")
def list_responses():
    """List saved call transcripts (text files only)."""
    return [
        {"name": f.name, "size": f.stat().st_size}
        for f in sorted(RESPONSES.iterdir(), reverse=True)
        if f.is_file() and f.suffix == ".txt" and not f.name.startswith(".")
    ]


@app.get("/api/responses/{name}")
def download_response(name: str):
    """Download a saved call transcript."""
    path = _safe_response_path(name)
    if not path.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(
        path,
        media_type="text/plain; charset=utf-8",
        filename=path.name,
    )


@app.delete("/api/responses/{name}")
def delete_response(name: str):
    path = _safe_response_path(name)
    if not path.is_file():
        raise HTTPException(404, "File not found")
    path.unlink()
    json_sidecar = path.with_suffix(".json")
    if json_sidecar.is_file():
        json_sidecar.unlink()
    return {"ok": True, "name": path.name}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
