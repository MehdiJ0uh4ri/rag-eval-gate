"""RAGAS metric wiring, with Claude as the judge.

Two things are non-obvious and both are load-bearing:

1. The judge LLM is Claude via ``langchain_anthropic.ChatAnthropic``, wrapped in
   RAGAS's ``LangchainLLMWrapper``. We never pass ``temperature`` -- sampling
   parameters were removed on Claude Opus 5 / Sonnet 5 / Opus 4.7+ and the API
   returns 400 if you send them. Judge stability comes from ``--repeats`` and a
   median, not from temperature=0. See docs/ci-quirks.md.

2. Embeddings are local (sentence-transformers). Anthropic has no embeddings
   endpoint, and answer_relevancy / semantic similarity need one. Keeping the
   embedder local also keeps the metric numerically stable across runs, which
   matters when you are gating on a 0.03 delta.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from app.config import EMBEDDING_MODEL, JUDGE_MODEL

# Metric keys the rest of the pipeline (thresholds, gate, report, baseline)
# agrees on. Renaming one here without updating eval/thresholds.yaml is caught
# by gate.py at load time.
FAITHFULNESS = "faithfulness"
ANSWER_RELEVANCY = "answer_relevancy"
CONTEXT_PRECISION = "context_precision"
CONTEXT_RECALL = "context_recall"
ANSWER_CORRECTNESS = "answer_correctness"

RAGAS_METRICS = [
    FAITHFULNESS,
    ANSWER_RELEVANCY,
    CONTEXT_PRECISION,
    CONTEXT_RECALL,
    ANSWER_CORRECTNESS,
]

# Derived, non-RAGAS metrics computed in run_eval.py.
HALLUCINATION_RATE = "hallucination_rate"
REFUSAL_ACCURACY = "refusal_accuracy"
RETRIEVAL_HIT_RATE = "retrieval_hit_rate"

DERIVED_METRICS = [HALLUCINATION_RATE, REFUSAL_ACCURACY, RETRIEVAL_HIT_RATE]
ALL_METRICS = RAGAS_METRICS + DERIVED_METRICS

# Metrics where lower is better. The gate flips comparison direction on these.
LOWER_IS_BETTER = {HALLUCINATION_RATE}


@dataclass(frozen=True)
class JudgeSettings:
    model: str = JUDGE_MODEL
    max_tokens: int = int(os.getenv("JUDGE_MAX_TOKENS", "4096"))
    effort: str = os.getenv("JUDGE_EFFORT", "low")
    timeout_s: float = float(os.getenv("JUDGE_TIMEOUT_S", "180"))
    max_retries: int = int(os.getenv("JUDGE_MAX_RETRIES", "4"))
    max_workers: int = int(os.getenv("JUDGE_MAX_WORKERS", "4"))


def build_judge(settings: JudgeSettings | None = None):
    """RAGAS-wrapped Claude judge."""
    from langchain_anthropic import ChatAnthropic
    from ragas.llms import LangchainLLMWrapper

    s = settings or JudgeSettings()
    chat = ChatAnthropic(
        model=s.model,
        max_tokens=s.max_tokens,
        timeout=s.timeout_s,
        max_retries=s.max_retries,
        # No temperature / top_p / top_k: rejected with 400 on current models.
        # Effort is the cost/depth dial; "low" is plenty for scoring one claim
        # against one passage and keeps a 24-item run cheap.
        model_kwargs={"output_config": {"effort": s.effort}},
    )
    return LangchainLLMWrapper(chat)


def build_embeddings():
    """Local embeddings -- no second vendor, no network in the metric path."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    )


def build_metrics(judge, embeddings) -> list:
    """Instantiate the RAGAS metric objects, bound to our judge/embedder.

    Metric classes (not the module-level singletons) so the judge binding is
    explicit -- the singletons silently fall back to OpenAI defaults if the
    global RAGAS config was never set, which is exactly the kind of thing that
    turns into a surprise vendor bill in CI.
    """
    from ragas.metrics import (
        AnswerCorrectness,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    return [
        Faithfulness(llm=judge),
        ResponseRelevancy(llm=judge, embeddings=embeddings),
        LLMContextPrecisionWithReference(llm=judge),
        LLMContextRecall(llm=judge),
        AnswerCorrectness(llm=judge, embeddings=embeddings),
    ]


# RAGAS result column -> our metric key.
RAGAS_COLUMN_MAP = {
    "faithfulness": FAITHFULNESS,
    "answer_relevancy": ANSWER_RELEVANCY,
    "semantic_similarity": None,  # component of answer_correctness; not gated
    "llm_context_precision_with_reference": CONTEXT_PRECISION,
    "context_precision": CONTEXT_PRECISION,
    "context_recall": CONTEXT_RECALL,
    "answer_correctness": ANSWER_CORRECTNESS,
}


def normalize_columns(row: dict) -> dict:
    """Map whatever column names this RAGAS version emitted onto our keys."""
    out: dict[str, float] = {}
    for column, value in row.items():
        key = RAGAS_COLUMN_MAP.get(column)
        if key is None:
            continue
        if isinstance(value, (int, float)):
            out[key] = float(value)
    return out
