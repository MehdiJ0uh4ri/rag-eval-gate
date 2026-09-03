"""FastAPI wrapper -- what the perf gate (k6) and the deployed chart serve."""
from __future__ import annotations

import time

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import RagConfig
from .rag import build_pipeline

app = FastAPI(title="acme-rag", version="1.0.0")
_config = RagConfig()
_pipeline = build_pipeline(_config)


class Query(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class Answer(BaseModel):
    answer: str
    context_ids: list[str]
    latency_ms: int


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "chunks": len(_pipeline.index.chunks)}


@app.get("/config")
def config() -> dict:
    return _config.fingerprint()


@app.post("/v1/query", response_model=Answer)
def query(body: Query) -> Answer:
    started = time.perf_counter()
    result = _pipeline.answer(body.question)
    return Answer(
        answer=result.answer,
        context_ids=result.context_ids,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
