"""The RAG pipeline under test: retrieve -> prompt -> Claude -> answer."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

import anthropic

from .config import GENERATOR_MODEL, RagConfig
from .index import BM25Index


@dataclass
class RagAnswer:
    question: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    context_ids: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


@lru_cache(maxsize=4)
def _index(chunk_chars: int, chunk_overlap: int) -> BM25Index:
    return BM25Index.build(chunk_chars, chunk_overlap)


@lru_cache(maxsize=1)
def _client() -> anthropic.Anthropic:
    # Zero-arg constructor: resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
    # or an `ant auth login` profile. CI injects the key as a secret.
    return anthropic.Anthropic(max_retries=4, timeout=120.0)


class RagPipeline:
    def __init__(self, config: RagConfig | None = None):
        self.config = config or RagConfig()
        self.index = _index(self.config.chunk_chars, self.config.chunk_overlap)

    def retrieve(self, question: str) -> list[tuple[str, str]]:
        hits = self.index.search(question, self.config.top_k)
        return [(c.id, c.text) for c, _ in hits]

    def answer(self, question: str) -> RagAnswer:
        retrieved = self.retrieve(question)
        if not retrieved:
            return RagAnswer(question=question, answer="I do not have that in the documentation.")

        block = "\n\n".join(
            f"[{i + 1}] ({cid})\n{text}" for i, (cid, text) in enumerate(retrieved)
        )
        user = f"Context passages:\n\n{block}\n\n---\n\nQuestion: {question}"

        resp = _client().messages.create(
            model=GENERATOR_MODEL,
            max_tokens=self.config.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": self.config.system_prompt,
                    # The system prompt is byte-stable across the whole golden
                    # run; the per-question context goes after the breakpoint.
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            output_config={"effort": self.config.effort},
            messages=[{"role": "user", "content": user}],
        )

        if resp.stop_reason == "refusal":
            detail = getattr(resp.stop_details, "category", None)
            text = f"[refused: {detail}]"
        else:
            text = "".join(b.text for b in resp.content if b.type == "text").strip()

        return RagAnswer(
            question=question,
            answer=text,
            contexts=[t for _, t in retrieved],
            context_ids=[cid for cid, _ in retrieved],
            usage={
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0),
            },
        )


class OfflinePipeline(RagPipeline):
    """Retrieval-only stub for `RAG_OFFLINE=1`.

    Used by the unit tests and by the perf gate's warm-up so that pipeline
    plumbing can be exercised without spending judge/generator tokens. It is
    never used by the quality gate -- an offline run cannot score faithfulness.
    """

    def answer(self, question: str) -> RagAnswer:
        retrieved = self.retrieve(question)
        first = retrieved[0][1] if retrieved else ""
        return RagAnswer(
            question=question,
            answer=first[:400] or "I do not have that in the documentation.",
            contexts=[t for _, t in retrieved],
            context_ids=[cid for cid, _ in retrieved],
            usage={"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0},
        )


def build_pipeline(config: RagConfig | None = None) -> RagPipeline:
    cls = OfflinePipeline if os.getenv("RAG_OFFLINE") == "1" else RagPipeline
    return cls(config)
