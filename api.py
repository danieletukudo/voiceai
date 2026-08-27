"""Knowledge API: ingest Maps / website / documents into Qdrant RAG, serve UI + library."""

import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from business_ingest import ingest_business_sources, normalize_source_urls
from ingest_document import ingest_file, remove_from_index

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DOCS = ROOT / "docs"
STATIC = ROOT / "static"
DOCS.mkdir(parents=True, exist_ok=True)
STATIC.mkdir(parents=True, exist_ok=True)

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT") or os.getenv("API_PORT", "8000"))

app = FastAPI(title="Knowledge API")


class BusinessIngestResponse(BaseModel):
    ok: bool = True
    name: str | None = None
    profile_file: str | None = None
    ingested: list[str] = Field(default_factory=list)
    chunks: int = 0
    warnings: list[str] = Field(default_factory=list)
    sources: dict = Field(default_factory=dict)


def _safe_path(name: str) -> Path:
    path = (DOCS / Path(name).name).resolve()
    if path.parent != DOCS.resolve():
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
    index = STATIC / "index.html"
    if not index.is_file():
        raise HTTPException(404, "static/index.html not found")
    return FileResponse(index)


@app.head("/")
def home_head():
    return {"ok": True}


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


@app.post("/api/business", response_model=BusinessIngestResponse)
async def ingest_business(
    maps_url: str | None = Form(None),
    website_url: str | None = Form(None),
    file: UploadFile | None = File(None),
):
    """Ingest Maps text, website text, and/or an uploaded document into RAG.

    Maps-only, website-only, both, or either plus a document are all valid.
    """
    maps_url, website_url = normalize_source_urls(maps_url, website_url)
    extra_paths: list[Path] = []
    if file is not None and file.filename:
        path = _safe_path(file.filename)
        data = await file.read()
        if not data:
            raise HTTPException(400, "Uploaded document is empty")
        path.write_bytes(data)
        extra_paths.append(path)

    if not maps_url and not website_url and not extra_paths:
        raise HTTPException(
            400,
            "Provide a Google Maps URL, a website URL, a document, or any combination",
        )

    try:
        result = ingest_business_sources(
            maps_url=maps_url,
            website_url=website_url,
            extra_files=extra_paths,
            docs_dir=DOCS,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Business ingest failed: {exc}") from exc
    return BusinessIngestResponse(**result)


@app.post("/api/files")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No filename")
    path = _safe_path(file.filename)
    data = await file.read()
    if not data:
        raise HTTPException(400, "Uploaded document is empty")
    path.write_bytes(data)
    try:
        chunks = ingest_file(path)
    except Exception as exc:
        raise HTTPException(500, f"Ingest failed: {exc}") from exc
    return {"ok": True, "name": path.name, "chunks": chunks}


@app.get("/api/files/{name}")
def get_file_meta(name: str):
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
    removed = remove_from_index(path.name)
    path.unlink()
    return {"ok": True, "name": path.name, "chunks_removed": removed}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
