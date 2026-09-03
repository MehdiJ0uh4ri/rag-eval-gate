"""Runtime knobs for the RAG service.

Everything the pipeline can regress on is an env var so a PR can change
retrieval or prompting behaviour without touching eval code -- and so the
gate can be demonstrated failing (`make demo-regression`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# Judge and generator default to the same model family. Keep them pinned:
# a silent model bump is a silent baseline shift.
GENERATOR_MODEL = os.getenv("RAG_GENERATOR_MODEL", "claude-opus-5")
JUDGE_MODEL = os.getenv("RAG_JUDGE_MODEL", "claude-opus-5")

# Anthropic has no embeddings endpoint; RAGAS needs one for answer_relevancy
# and semantic similarity. We run a local sentence-transformers model so CI
# has exactly one LLM vendor and one offline dependency. See docs/ci-quirks.md.
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

PROMPT_VARIANTS = {
    # The grounded prompt is the shipped behaviour.
    "grounded": (
        "You answer questions about the Acme Payments API using ONLY the numbered "
        "context passages below.\n"
        "Rules:\n"
        "- Every factual claim must be supported by a passage. Do not add rules, "
        "limits, timings, or error codes that are not written there.\n"
        "- If the context does not contain the answer, reply exactly: "
        "\"I do not have that in the documentation.\"\n"
        "- Be concise: at most four sentences. No preamble."
    ),
    # Deliberately ungrounded -- used by the regression demo to prove the gate
    # actually blocks. Do not ship.
    "loose": (
        "You are a helpful expert on payments APIs. Use the context below if it "
        "helps, and fall back on your general knowledge of payment platforms to "
        "give the user a complete, confident answer."
    ),
}


@dataclass(frozen=True)
class RagConfig:
    # default_factory, not a bare default: a bare default is evaluated once at
    # import time, which would silently ignore any env var set afterwards --
    # including the ones `make demo-regression` and the tests rely on.
    top_k: int = field(default_factory=lambda: int(os.getenv("RAG_TOP_K", "4")))
    chunk_chars: int = field(default_factory=lambda: int(os.getenv("RAG_CHUNK_CHARS", "700")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("RAG_CHUNK_OVERLAP", "120")))
    prompt_variant: str = field(default_factory=lambda: os.getenv("RAG_PROMPT_VARIANT", "grounded"))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("RAG_MAX_TOKENS", "1024")))
    effort: str = field(default_factory=lambda: os.getenv("RAG_EFFORT", "low"))

    @property
    def system_prompt(self) -> str:
        try:
            return PROMPT_VARIANTS[self.prompt_variant]
        except KeyError as err:
            raise SystemExit(
                f"unknown RAG_PROMPT_VARIANT={self.prompt_variant!r}; "
                f"expected one of {sorted(PROMPT_VARIANTS)}"
            ) from err

    def fingerprint(self) -> dict:
        """Goes into every eval report. A metric delta without this is unreadable."""
        return {
            "generator_model": GENERATOR_MODEL,
            "judge_model": JUDGE_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "top_k": self.top_k,
            "chunk_chars": self.chunk_chars,
            "chunk_overlap": self.chunk_overlap,
            "prompt_variant": self.prompt_variant,
            "effort": self.effort,
        }
