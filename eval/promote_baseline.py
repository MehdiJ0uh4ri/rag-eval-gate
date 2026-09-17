"""Promote a passing results.json to the tracked baseline.

Only ever runs on the default branch, after the gate passed. Two guards:

* the run must have passed its own gate (we re-gate here rather than trusting
  an upstream job's exit code -- artifacts get reused);
* the baseline is stripped down to metrics + provenance. Per-item answers are
  not committed: they are large, they churn every run, and a diff full of
  regenerated prose hides the two numbers that actually moved.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.gate import load_yaml, run_checks, verdict

KEEP = ("schema", "generated_at", "git_sha", "branch", "config", "repeats",
        "judge_coverage", "metrics", "per_tag", "usage")


def strip(results: dict) -> dict:
    return {k: results[k] for k in KEEP if k in results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promote a run to the baseline")
    parser.add_argument("--results", type=Path, default=Path("artifacts/results.json"))
    parser.add_argument("--baseline", type=Path, default=Path("eval/baseline/main.json"))
    parser.add_argument("--thresholds", type=Path, default=Path("eval/thresholds.yaml"))
    parser.add_argument("--force", action="store_true", help="skip the re-gate (break glass)")
    args = parser.parse_args(argv)

    results = json.loads(args.results.read_text(encoding="utf-8"))
    previous = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline.exists() else None

    if not args.force:
        decision = verdict(run_checks(results, previous, load_yaml(args.thresholds)))
        if decision == "FAIL":
            print(f"refusing to promote a {decision} run", file=sys.stderr)
            return 1

    args.baseline.parent.mkdir(parents=True, exist_ok=True)
    args.baseline.write_text(json.dumps(strip(results), indent=2) + "\n", encoding="utf-8")

    faith = results["metrics"]["faithfulness"]["mean"]
    halluc = results["metrics"]["hallucination_rate"]["mean"]
    print(f"baseline updated -> {args.baseline} "
          f"(faithfulness {faith:.3f}, hallucination_rate {halluc:.3f}, "
          f"sha {results.get('git_sha', '?')[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
