"""Run the golden dataset through the RAG pipeline and score it with RAGAS.

Output is a single JSON results file consumed by eval/gate.py and eval/report.py.
Nothing here decides pass/fail -- scoring and gating are deliberately separate so
you can re-gate an old artifact after a threshold change without re-spending
judge tokens.

    python -m eval.run_eval --out artifacts/results.json --repeats 3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from app.config import RagConfig
from app.rag import build_pipeline
from eval.metrics import (
    ALL_METRICS,
    ANSWER_RELEVANCY,
    CONTEXT_PRECISION,
    CONTEXT_RECALL,
    FAITHFULNESS,
    HALLUCINATION_RATE,
    REFUSAL_ACCURACY,
    RETRIEVAL_HIT_RATE,
    JudgeSettings,
    build_embeddings,
    build_judge,
    build_metrics,
    normalize_columns,
)
from golden.loader import REFUSAL_SENTINEL, GoldenItem, load, validate

SCHEMA_VERSION = 2
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260826  # fixed: a moving CI is not a gate

# A per-item faithfulness below this counts the item as hallucinating. RAGAS
# faithfulness is (supported claims / total claims), so 0.5 means at least half
# the claims in the answer are not grounded in the retrieved context.
HALLUCINATION_FAITHFULNESS_FLOOR = float(os.getenv("HALLUCINATION_FLOOR", "0.50"))


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------

def generate_answers(items: list[GoldenItem], config: RagConfig, workers: int) -> list[dict]:
    pipeline = build_pipeline(config)

    def one(item: GoldenItem) -> dict:
        started = time.perf_counter()
        result = pipeline.answer(item.question)
        return {
            "id": item.id,
            "question": item.question,
            "answer": result.answer,
            "contexts": result.contexts,
            "context_ids": result.context_ids,
            "ground_truth": item.ground_truth,
            "tags": item.tags,
            "reference_ids": item.reference_ids,
            "must_not_say": item.must_not_say,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "usage": result.usage,
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, items))


# --------------------------------------------------------------------------
# deterministic checks -- no LLM involved, so these never flake
# --------------------------------------------------------------------------

def deterministic_flags(row: dict) -> list[str]:
    flags: list[str] = []
    answer = row["answer"].lower()
    refused = REFUSAL_SENTINEL in answer
    unanswerable = "unanswerable" in row["tags"]

    for banned in row["must_not_say"]:
        if banned in answer:
            flags.append(f"banned_phrase:{banned}")

    if unanswerable and not refused:
        flags.append("answered_unanswerable")
    if not unanswerable and refused:
        flags.append("refused_answerable")

    if row["reference_ids"]:
        retrieved_docs = {cid.split("#", 1)[0] for cid in row["context_ids"]}
        if not set(row["reference_ids"]) & retrieved_docs:
            flags.append("retrieval_miss")
    return flags


def derived_metrics(rows: list[dict], per_item_ragas: dict[str, dict]) -> dict[str, list[float]]:
    """Per-item values (0/1 or continuous) for the metrics RAGAS does not give us."""
    hallucination: list[float] = []
    refusal: list[float] = []
    retrieval: list[float] = []

    for row in rows:
        flags = row["flags"]
        faith = per_item_ragas.get(row["id"], {}).get(FAITHFULNESS)

        # Hallucination is a union of three independent signals: an LLM-judged
        # faithfulness collapse, a hand-curated banned phrase, or answering a
        # question the corpus cannot answer. Any one of them is a hallucination.
        judged_bad = faith is not None and faith < HALLUCINATION_FAITHFULNESS_FLOOR
        banned = any(f.startswith("banned_phrase:") for f in flags)
        overreach = "answered_unanswerable" in flags
        hallucination.append(1.0 if (judged_bad or banned or overreach) else 0.0)

        if "unanswerable" in row["tags"]:
            refusal.append(0.0 if "answered_unanswerable" in flags else 1.0)
        else:
            refusal.append(0.0 if "refused_answerable" in flags else 1.0)

        if row["reference_ids"]:
            retrieval.append(0.0 if "retrieval_miss" in flags else 1.0)

    return {
        HALLUCINATION_RATE: hallucination,
        REFUSAL_ACCURACY: refusal,
        RETRIEVAL_HIT_RATE: retrieval,
    }


# --------------------------------------------------------------------------
# judging
# --------------------------------------------------------------------------

def judge_once(rows: list[dict], settings: JudgeSettings) -> dict[str, dict[str, float]]:
    """One RAGAS pass. Returns {item_id: {metric: value}}."""
    from ragas import EvaluationDataset, RunConfig, evaluate

    samples = [
        {
            "user_input": row["question"],
            "retrieved_contexts": row["contexts"] or [""],
            "response": row["answer"],
            "reference": row["ground_truth"],
        }
        for row in rows
    ]
    dataset = EvaluationDataset.from_list(samples)

    judge = build_judge(settings)
    embeddings = build_embeddings()

    result = evaluate(
        dataset=dataset,
        metrics=build_metrics(judge, embeddings),
        llm=judge,
        embeddings=embeddings,
        run_config=RunConfig(
            timeout=int(settings.timeout_s),
            max_workers=settings.max_workers,
            max_retries=settings.max_retries,
        ),
        show_progress=False,
        raise_exceptions=False,  # a single judge failure becomes NaN, not a crash
    )

    frame = result.to_pandas()
    out: dict[str, dict[str, float]] = {}
    # strict=True: a RAGAS frame that lost a row would otherwise silently
    # misalign every score with the wrong question.
    for row, (_, record) in zip(rows, frame.iterrows(), strict=True):
        values = normalize_columns(record.to_dict())
        # NaN means the judge failed on that item. Dropping it is safer than
        # scoring it 0 (which would fail the build on an API blip) and safer
        # than scoring it 1 (which would hide a real regression); coverage is
        # reported and gated separately.
        out[row["id"]] = {k: v for k, v in values.items() if v == v}
    return out


def median_across_repeats(passes: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    merged: dict[str, dict[str, float]] = {}
    ids = {i for p in passes for i in p}
    for item_id in ids:
        per_metric: dict[str, list[float]] = {}
        for p in passes:
            for metric, value in p.get(item_id, {}).items():
                per_metric.setdefault(metric, []).append(value)
        merged[item_id] = {m: statistics.median(v) for m, v in per_metric.items() if v}
    return merged


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

def summarize(values: list[float]) -> dict:
    """Mean plus a percentile-bootstrap 95% CI over the golden items.

    The gate compares the CI bound, not the point estimate, so a two-question
    wobble on a 24-item set cannot turn into a red build.
    """
    clean = [v for v in values if v == v]
    if not clean:
        return {"mean": None, "n": 0, "ci95_low": None, "ci95_high": None, "stdev": None}
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(clean)
    means = sorted(
        sum(rng.choice(clean) for _ in range(n)) / n for _ in range(BOOTSTRAP_SAMPLES)
    )
    lo = means[int(0.025 * BOOTSTRAP_SAMPLES)]
    hi = means[int(0.975 * BOOTSTRAP_SAMPLES) - 1]
    return {
        "mean": round(statistics.fmean(clean), 4),
        "median": round(statistics.median(clean), 4),
        "stdev": round(statistics.pstdev(clean), 4) if n > 1 else 0.0,
        "ci95_low": round(lo, 4),
        "ci95_high": round(hi, 4),
        "n": n,
    }


def git_sha() -> str:
    for cmd in (["git", "rev-parse", "HEAD"],):
        try:
            return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            pass
    return os.getenv("GITHUB_SHA", "unknown")


def build_results(rows: list[dict], per_item: dict[str, dict[str, float]], config: RagConfig,
                  repeats: int) -> dict:
    derived = derived_metrics(rows, per_item)
    for row in rows:
        row_metrics = dict(per_item.get(row["id"], {}))
        row["metrics"] = row_metrics

    metrics: dict[str, dict] = {}
    for metric in (FAITHFULNESS, ANSWER_RELEVANCY, CONTEXT_PRECISION, CONTEXT_RECALL,
                   "answer_correctness"):
        metrics[metric] = summarize([
            per_item.get(r["id"], {}).get(metric) for r in rows
            if per_item.get(r["id"], {}).get(metric) is not None
        ])
    for metric, values in derived.items():
        metrics[metric] = summarize(values)

    # Judge coverage: fraction of (item, ragas metric) pairs that produced a
    # number. A collapsing judge shows up here before it shows up in the scores.
    expected = len(rows) * 5
    got = sum(len(v) for v in per_item.values())
    coverage = round(got / expected, 4) if expected else 0.0

    per_tag: dict[str, dict[str, float]] = {}
    for tag in sorted({t for r in rows for t in r["tags"]}):
        tagged = [r for r in rows if tag in r["tags"]]
        tag_derived = derived_metrics(tagged, per_item)
        entry: dict[str, float] = {}
        for metric in (FAITHFULNESS, ANSWER_RELEVANCY, CONTEXT_RECALL):
            vals = [per_item.get(r["id"], {}).get(metric) for r in tagged]
            vals = [v for v in vals if v is not None]
            if vals:
                entry[metric] = round(statistics.fmean(vals), 4)
        for metric, vals in tag_derived.items():
            if vals:
                entry[metric] = round(statistics.fmean(vals), 4)
        entry["n"] = len(tagged)
        per_tag[tag] = entry

    usage = {
        "generator_input_tokens": sum(r["usage"].get("input_tokens", 0) for r in rows),
        "generator_output_tokens": sum(r["usage"].get("output_tokens", 0) for r in rows),
        "generator_cache_read_tokens": sum(
            r["usage"].get("cache_read_input_tokens", 0) for r in rows
        ),
        "judge_passes": repeats,
        "items": len(rows),
    }

    return {
        "schema": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "branch": os.getenv("GITHUB_HEAD_REF") or os.getenv("GITHUB_REF_NAME", "local"),
        "pr": os.getenv("PR_NUMBER"),
        "config": config.fingerprint(),
        "repeats": repeats,
        "judge_coverage": coverage,
        "metrics": metrics,
        "per_tag": per_tag,
        "latency_ms_p95": _p95([r["latency_ms"] for r in rows]),
        "usage": usage,
        "per_item": [
            {
                "id": r["id"],
                "tags": r["tags"],
                "question": r["question"],
                "answer": r["answer"],
                "context_ids": r["context_ids"],
                "flags": r["flags"],
                "latency_ms": r["latency_ms"],
                "metrics": {k: round(v, 4) for k, v in r["metrics"].items()},
            }
            for r in rows
        ],
    }


def _p95(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the RAG golden-set evaluation")
    parser.add_argument("--out", default="artifacts/results.json", type=Path)
    parser.add_argument("--repeats", type=int, default=int(os.getenv("EVAL_REPEATS", "1")),
                        help="judge passes over the same answers; scores are the median")
    parser.add_argument("--limit", type=int, default=None, help="first N items (smoke runs only)")
    parser.add_argument("--tag", action="append", default=[], help="restrict to a tag slice")
    parser.add_argument("--workers", type=int, default=int(os.getenv("GEN_WORKERS", "4")))
    parser.add_argument("--answers", type=Path, default=None,
                        help="reuse a previous answers.json instead of regenerating")
    parser.add_argument("--save-answers", type=Path, default=None)
    args = parser.parse_args(argv)

    errors = validate()
    if errors:
        print("golden dataset INVALID -- refusing to run a paid eval:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 2

    items = load()
    if args.tag:
        items = [i for i in items if set(args.tag) & set(i.tags)]
    if args.limit:
        items = items[: args.limit]
    if not items:
        print("no golden items selected", file=sys.stderr)
        return 2

    config = RagConfig()
    print(f"config: {json.dumps(config.fingerprint())}", file=sys.stderr)

    if args.answers and args.answers.exists():
        rows = json.loads(args.answers.read_text(encoding="utf-8"))
        print(f"reusing {len(rows)} cached answers from {args.answers}", file=sys.stderr)
    else:
        print(f"generating {len(items)} answers...", file=sys.stderr)
        rows = generate_answers(items, config, args.workers)
        if args.save_answers:
            args.save_answers.parent.mkdir(parents=True, exist_ok=True)
            args.save_answers.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    for row in rows:
        row["flags"] = deterministic_flags(row)

    settings = JudgeSettings()
    print(f"judging with {settings.model} x{args.repeats}...", file=sys.stderr)
    passes = [judge_once(rows, settings) for _ in range(max(args.repeats, 1))]
    per_item = median_across_repeats(passes)

    results = build_results(rows, per_item, config, max(args.repeats, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nwrote {args.out}", file=sys.stderr)
    for metric in ALL_METRICS:
        summary = results["metrics"].get(metric)
        if summary and summary["mean"] is not None:
            print(
                f"  {metric:<20} {summary['mean']:.3f}  "
                f"[{summary['ci95_low']:.3f}, {summary['ci95_high']:.3f}]  n={summary['n']}",
                file=sys.stderr,
            )
    print(f"  judge_coverage       {results['judge_coverage']:.3f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
