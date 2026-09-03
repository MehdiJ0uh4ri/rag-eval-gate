"""Chunking + BM25 retrieval over the local corpus.

Deliberately dependency-light: the point of this lab is the eval gate, not the
vector store. Swapping this for pgvector/Qdrant should not change eval/ at all.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

CORPUS_DIR = Path(__file__).parent / "corpus"
_TOKEN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class Chunk:
    doc: str
    ordinal: int
    text: str

    @property
    def id(self) -> str:
        return f"{self.doc}#{self.ordinal}"


def _split(text: str, size: int, overlap: int) -> list[str]:
    """Paragraph-greedy split with a character budget and overlap tail."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    out: list[str] = []
    buf = ""
    for para in paras:
        if buf and len(buf) + len(para) + 2 > size:
            out.append(buf)
            tail = buf[-overlap:] if overlap else ""
            buf = (tail + "\n\n" + para).strip() if tail else para
        else:
            buf = f"{buf}\n\n{para}".strip() if buf else para
    if buf:
        out.append(buf)
    return out


class BM25Index:
    K1 = 1.5
    B = 0.75

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.docs = [tokenize(c.text) for c in chunks]
        self.lengths = [len(d) for d in self.docs]
        self.avgdl = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.tf = [Counter(d) for d in self.docs]
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    @classmethod
    def build(cls, chunk_chars: int, chunk_overlap: int) -> BM25Index:
        chunks: list[Chunk] = []
        for path in sorted(CORPUS_DIR.glob("*.md")):
            for i, piece in enumerate(_split(path.read_text(encoding="utf-8"), chunk_chars, chunk_overlap)):
                chunks.append(Chunk(doc=path.name, ordinal=i, text=piece))
        if not chunks:
            raise RuntimeError(f"empty corpus at {CORPUS_DIR}")
        return cls(chunks)

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        q = tokenize(query)
        scored: list[tuple[Chunk, float]] = []
        for i, tf in enumerate(self.tf):
            dl = self.lengths[i] or 1
            score = 0.0
            for term in q:
                if term not in tf:
                    continue
                f = tf[term]
                denom = f + self.K1 * (1 - self.B + self.B * dl / (self.avgdl or 1))
                score += self.idf.get(term, 0.0) * f * (self.K1 + 1) / denom
            if score > 0:
                scored.append((self.chunks[i], score))
        scored.sort(key=lambda p: p[1], reverse=True)
        return scored[:top_k]
