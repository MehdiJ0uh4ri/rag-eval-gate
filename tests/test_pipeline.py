"""Offline tests: dataset shape, retrieval, and the deterministic flags.

None of these call the API (RAG_OFFLINE=1), so they run on every push in
seconds and catch the boring breakages -- a renamed corpus file, a golden item
pointing at a doc that no longer exists, a chunker that stopped retrieving.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("RAG_OFFLINE", "1")

from app.config import PROMPT_VARIANTS, RagConfig  # noqa: E402
from app.index import BM25Index  # noqa: E402
from app.rag import build_pipeline  # noqa: E402
from eval.run_eval import derived_metrics, deterministic_flags  # noqa: E402
from golden.loader import load, validate  # noqa: E402


@pytest.fixture(scope="module")
def index() -> BM25Index:
    return BM25Index.build(chunk_chars=700, chunk_overlap=120)


def test_golden_dataset_is_valid():
    assert validate() == []


def test_every_answerable_item_retrieves_its_reference_doc(index):
    """The golden set must be answerable by the retriever we actually ship.

    If this fails, either retrieval regressed or the dataset asks something the
    corpus does not contain -- both are bugs, and neither should be discovered
    by a confusing faithfulness drop three steps later.
    """
    misses = []
    for item in load():
        if not item.reference_ids:
            continue
        hits = index.search(item.question, top_k=4)
        docs = {chunk.doc for chunk, _ in hits}
        if not set(item.reference_ids) & docs:
            misses.append((item.id, sorted(docs)))
    assert not misses, f"retrieval misses: {misses}"


def test_chunking_covers_the_whole_corpus(index):
    assert len(index.chunks) >= 6
    joined = " ".join(c.text for c in index.chunks)
    for needle in ("refund_limit", "whsec_", "idempotency_key_reuse",
                   "rate_limit_exceeded", "debit_not_authorized", "payment_disputed"):
        assert needle in joined, f"{needle} fell out of the index"


def test_offline_pipeline_returns_contexts():
    result = build_pipeline(RagConfig()).answer("How many refunds can a payment have?")
    assert result.contexts
    assert result.context_ids
    assert "01-refunds.md" in result.context_ids[0]


def test_unknown_prompt_variant_fails_loudly(monkeypatch):
    monkeypatch.setenv("RAG_PROMPT_VARIANT", "does-not-exist")
    with pytest.raises(SystemExit):
        _ = RagConfig().system_prompt


def test_both_prompt_variants_exist():
    assert {"grounded", "loose"} <= set(PROMPT_VARIANTS)


def _row(**over):
    row = {
        "id": "x", "question": "q", "answer": "A payment supports at most 20 refunds.",
        "contexts": ["..."], "context_ids": ["01-refunds.md#0"],
        "ground_truth": "At most 20.", "tags": ["factual"],
        "reference_ids": ["01-refunds.md"], "must_not_say": ["unlimited"],
        "latency_ms": 10, "usage": {},
    }
    row.update(over)
    return row


def test_banned_phrase_is_flagged():
    row = _row(answer="You can create unlimited refunds.")
    assert any(f.startswith("banned_phrase:") for f in deterministic_flags(row))


def test_answering_an_unanswerable_question_is_flagged():
    row = _row(tags=["unanswerable"], reference_ids=[], must_not_say=[],
               answer="Acme supports Bitcoin and USDC.")
    assert "answered_unanswerable" in deterministic_flags(row)


def test_refusing_an_answerable_question_is_flagged():
    row = _row(answer="I do not have that in the documentation.")
    assert "refused_answerable" in deterministic_flags(row)


def test_retrieval_miss_is_flagged():
    row = _row(context_ids=["04-rate-limits.md#0"])
    assert "retrieval_miss" in deterministic_flags(row)


def test_hallucination_is_a_union_of_three_signals():
    banned = _row(answer="Unlimited refunds are allowed.")
    banned["flags"] = deterministic_flags(banned)
    judged = _row()
    judged["flags"] = deterministic_flags(judged)

    derived = derived_metrics([banned, judged], {"x": {"faithfulness": 1.0}})
    # Both rows share id "x"; the banned-phrase row must still score 1.0
    # hallucination purely from the deterministic signal, with a perfect
    # judge score. That independence is the point.
    assert derived["hallucination_rate"][0] == 1.0
    assert derived["hallucination_rate"][1] == 0.0


def test_low_faithfulness_alone_counts_as_hallucination():
    row = _row()
    row["flags"] = deterministic_flags(row)
    derived = derived_metrics([row], {"x": {"faithfulness": 0.2}})
    assert derived["hallucination_rate"] == [1.0]
