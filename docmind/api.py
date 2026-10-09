"""FastAPI backend.  Run:  uvicorn docmind.api:app --reload"""
import logging
import re
import time
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl

from docmind.chains.extract import DocumentInsights
from docmind.config import get_settings
from docmind.ingest import SUPPORTED_EXTENSIONS
from docmind.models import get_chat_model, get_embeddings
from docmind.service import DocMindService, NoDocumentsError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("docmind")

app = FastAPI(title="DocMind API", version="1.0.0", description="LangChain RAG + agent document assistant")


@lru_cache
def get_service() -> DocMindService:
    settings = get_settings()
    return DocMindService(get_chat_model(settings), get_embeddings(settings), settings)


# ---------- request / response models ----------
class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="default", max_length=100)
    source: str | None = None


class UrlRequest(BaseModel):
    url: HttpUrl


class DocRequest(BaseModel):
    source: str | None = None
    style: str = Field(default="concise bullet-point", max_length=100)


# ---------- middleware / errors ----------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    log.info("%s %s -> %s in %.0fms", request.method, request.url.path, response.status_code,
             (time.perf_counter() - start) * 1000)
    return response


@app.exception_handler(NoDocumentsError)
async def no_documents(_: Request, exc: NoDocumentsError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


# ---------- routes ----------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents")
def list_documents(svc: DocMindService = Depends(get_service)):
    return {"sources": svc.list_sources()}


@app.post("/documents/upload", status_code=201)
async def upload_document(file: UploadFile = File(...), svc: DocMindService = Depends(get_service)):
    name = _safe_filename(file.filename or "")
    if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type. Allowed: {sorted(SUPPORTED_EXTENSIONS)}")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 10 MB)")

    upload_dir = Path(svc.settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / name
    path.write_bytes(content)
    return svc.ingest_file(path)


@app.post("/documents/url", status_code=201)
def ingest_url(body: UrlRequest, svc: DocMindService = Depends(get_service)):
    return svc.ingest_url(str(body.url))


@app.post("/ask")
def ask(body: AskRequest, svc: DocMindService = Depends(get_service)):
    return svc.ask(body.question, body.session_id)


@app.post("/assistant")
def assistant(body: AskRequest, svc: DocMindService = Depends(get_service)):
    return svc.assistant(body.question, body.session_id, body.source)


@app.post("/summarize")
def summarize(body: DocRequest, svc: DocMindService = Depends(get_service)):
    return svc.summarize(body.source, body.style)


@app.post("/extract", response_model=DocumentInsights)
def extract(body: DocRequest, svc: DocMindService = Depends(get_service)):
    return svc.extract(body.source)


@app.post("/agent")
def agent(body: AskRequest, svc: DocMindService = Depends(get_service)):
    return svc.agent(body.question, body.session_id)


@app.delete("/sessions/{session_id}", status_code=204)
def clear_session(session_id: str, svc: DocMindService = Depends(get_service)):
    svc.clear_session(session_id)


def _safe_filename(name: str) -> str:
    # Strip any path components and odd characters so uploads can't escape upload_dir.
    name = Path(name).name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    if not name.strip("._"):
        raise HTTPException(400, "Invalid file name")
    return name
