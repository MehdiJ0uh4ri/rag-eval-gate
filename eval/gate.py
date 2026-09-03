"""Turn a results.json into a pass/fail decision.

Separate from run_eval.py on purpose: gating is cheap, deterministic, and
re-runnable. You can re-gate last night's artifact after editing thresholds
without spending a single judge token.

    python -m eval.gate --results artifacts/results.json \
        --baseline eval/baseline/main.json --markdown artifacts/report.md

Exit codes:  0 pass (possibly with warnings) | 1 gate failed | 2 bad input
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from eval.metrics import LOWER_IS_BETTER

THRESHOLDS = Path(__file__).parent / "thresholds.yaml"


@dataclass
class Check:
    name: str
    metric: str
    kind: str          # absolute | regression | per_tag | health | waiver
    passed: bool
    blocking: bool
    observed: float | None
    limit: float | None
    detail: str
    waived: str | None = None

    @property
    def status(self) -> str:
        if self.waived:
            return "WAIVED"
        if self.passed:
            return "PASS"
        return "FAIL" if self.blocking else "WARN"


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def observed_value(summary: dict, metric: str, mode: str) -> float | None:
    """Point estimate or the conservative CI bound, per `mode`."""
    if not summary or summary.get("mean") is None:
        return None
    if mode != "ci95":
        return summary["mean"]
    if metric in LOWER_IS_BETTER:
        return summary.get("ci95_high", summary["mean"])
    return summary.get("ci95_low", summary["mean"])


def active_waivers(config: dict) -> tuple[dict[tuple[str, str], str], list[Check]]:
    """Return {(scope, metric): reason} plus failures for malformed/expired ones."""
    active: dict[tuple[str, str], str] = {}
    problems: list[Check] = []
    today = date.today()
    for i, waiver in enumerate(config.get("waivers") or []):
        where = f"waivers[{i}]"
        missing = {"metric", "scope", "until", "owner", "reason"} - set(waiver)
        if missing:
            problems.append(Check(where, "-", "waiver", False, True, None, None,
                                  f"waiver missing keys {sorted(missing)}"))
            continue
        try:
            until = waiver["until"] if isinstance(waiver["until"], date) else date.fromisoformat(
                str(waiver["until"]))
        except ValueError:
            problems.append(Check(where, waiver["metric"], "waiver", False, True, None, None,
                                  f"unparseable until={waiver['until']!r}"))
            continue
        if until < today:
            problems.append(Check(where, waiver["metric"], "waiver", False, True, None, None,
                                  f"waiver expired {until} (owner {waiver['owner']}) -- "
                                  "fix the metric or renew it deliberately"))
            continue
        active[(waiver["scope"], waiver["metric"])] = (
            f"{waiver['reason']} (owner {waiver['owner']}, until {until})"
        )
    return active, problems


def run_checks(results: dict, baseline: dict | None, config: dict) -> list[Check]:
    mode = config.get("mode", "ci95")
    waivers, checks = active_waivers(config)
    metrics = results.get("metrics", {})

    # ---- health ---------------------------------------------------------
    health = config.get("health", {})
    coverage = results.get("judge_coverage")
    cov_min = health.get("judge_coverage_min")
    if cov_min is not None:
        checks.append(Check(
            "judge_coverage", "judge_coverage", "health",
            passed=coverage is not None and coverage >= cov_min,
            blocking=True, observed=coverage, limit=cov_min,
            detail="fraction of (item, metric) pairs the judge actually scored",
        ))
    items = (metrics.get("faithfulness") or {}).get("n") or 0
    min_items = health.get("min_items")
    if min_items is not None:
        checks.append(Check(
            "min_items", "items", "health", passed=items >= min_items, blocking=True,
            observed=float(items), limit=float(min_items),
            detail="items scored in this run",
        ))

    # ---- absolute floors -------------------------------------------------
    for metric, rule in (config.get("absolute") or {}).items():
        # Per-rule override. Rare-event rates cannot be gated on a CI bound at
        # this sample size -- see the note in thresholds.yaml.
        value = observed_value(metrics.get(metric, {}), metric, rule.get("compare", mode))
        blocking = bool(rule.get("blocking", True))
        if value is None:
            checks.append(Check(f"absolute:{metric}", metric, "absolute", False, blocking,
                                None, None, "metric missing from results"))
            continue
        if "min" in rule:
            limit, ok = rule["min"], value >= rule["min"]
            detail = f"{metric} >= {limit}"
        else:
            limit, ok = rule["max"], value <= rule["max"]
            detail = f"{metric} <= {limit}"
        checks.append(Check(f"absolute:{metric}", metric, "absolute", ok, blocking, value,
                            limit, detail, waivers.get(("absolute", metric))))

    # ---- regression vs baseline -----------------------------------------
    if baseline:
        base_metrics = baseline.get("metrics", {})
        for metric, rule in (config.get("regression") or {}).items():
            current = observed_value(metrics.get(metric, {}), metric, "mean")
            previous = observed_value(base_metrics.get(metric, {}), metric, "mean")
            blocking = bool(rule.get("blocking", True))
            if current is None or previous is None:
                checks.append(Check(f"regression:{metric}", metric, "regression", True, blocking,
                                    current, None, "no baseline value; skipped"))
                continue
            if metric in LOWER_IS_BETTER:
                delta, limit = current - previous, rule["max_rise"]
                ok = delta <= limit
                detail = f"{metric} rose {delta:+.3f} vs baseline {previous:.3f} (max +{limit})"
            else:
                delta, limit = previous - current, rule["max_drop"]
                ok = delta <= limit
                detail = f"{metric} dropped {delta:+.3f} vs baseline {previous:.3f} (max {limit})"
            checks.append(Check(f"regression:{metric}", metric, "regression", ok, blocking,
                                current, limit, detail, waivers.get(("regression", metric))))
    else:
        # Deliberately a WARN, not a PASS: a run without a baseline is only
        # half a gate, and that has to be visible in the report rather than
        # looking like a clean bill of health.
        checks.append(Check("regression:baseline", "-", "regression", False, False, None, None,
                            "no baseline available; regression checks skipped, "
                            "absolute floors only"))

    # ---- per-tag slices --------------------------------------------------
    per_tag = results.get("per_tag", {})
    for tag, rules in (config.get("per_tag") or {}).items():
        slice_values = per_tag.get(tag)
        if not slice_values:
            checks.append(Check(f"per_tag:{tag}", "-", "per_tag", True, False, None, None,
                                f"tag {tag!r} not present in this run; skipped"))
            continue
        for metric, rule in rules.items():
            # Slice values are point estimates already; `compare` is accepted
            # here only so a rule reads the same in both sections.
            value = slice_values.get(metric)
            blocking = bool(rule.get("blocking", True))
            if value is None:
                checks.append(Check(f"per_tag:{tag}:{metric}", metric, "per_tag", True, False,
                                    None, None, "metric missing for slice; skipped"))
                continue
            if "min" in rule:
                limit, ok = rule["min"], value >= rule["min"]
                detail = f"[{tag}] {metric} >= {limit} (n={slice_values.get('n')})"
            else:
                limit, ok = rule["max"], value <= rule["max"]
                detail = f"[{tag}] {metric} <= {limit} (n={slice_values.get('n')})"
            checks.append(Check(f"per_tag:{tag}:{metric}", metric, "per_tag", ok, blocking,
                                value, limit, detail, waivers.get((f"per_tag:{tag}", metric))))

    return checks


def verdict(checks: list[Check]) -> str:
    if any(c.status == "FAIL" for c in checks):
        return "FAIL"
    if any(c.status in ("WARN", "WAIVED") for c in checks):
        return "PASS_WITH_WARNINGS"
    return "PASS"


def render_console(checks: list[Check], results: dict, decision: str) -> str:
    lines = [
        "",
        f"RAG quality gate: {decision}",
        f"  sha={results.get('git_sha', '?')[:12]} branch={results.get('branch')} "
        f"repeats={results.get('repeats')} mode-items={(results.get('metrics', {}).get('faithfulness') or {}).get('n')}",
        "",
        f"  {'CHECK':<34} {'STATUS':<7} {'OBSERVED':>9} {'LIMIT':>8}  DETAIL",
    ]
    for c in sorted(checks, key=lambda c: (c.status != "FAIL", c.status != "WARN", c.name)):
        obs = f"{c.observed:.3f}" if isinstance(c.observed, float) else "-"
        lim = f"{c.limit:.3f}" if isinstance(c.limit, float) else "-"
        detail = c.detail if not c.waived else f"{c.detail} -- WAIVED: {c.waived}"
        lines.append(f"  {c.name:<34} {c.status:<7} {obs:>9} {lim:>8}  {detail}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate a RAG eval run")
    parser.add_argument("--results", type=Path, default=Path("artifacts/results.json"))
    parser.add_argument("--baseline", type=Path, default=Path("eval/baseline/main.json"))
    parser.add_argument("--no-baseline", action="store_true",
                        help="skip regression checks entirely (smoke runs, first run on a branch)")
    parser.add_argument("--thresholds", type=Path, default=THRESHOLDS)
    parser.add_argument("--markdown", type=Path, default=None, help="write a PR-comment report")
    parser.add_argument("--json-out", type=Path, default=None, help="write the machine verdict")
    parser.add_argument("--warn-only", action="store_true",
                        help="report but always exit 0 (used by the nightly rescan)")
    args = parser.parse_args(argv)

    if not args.results.exists():
        print(f"no results at {args.results}", file=sys.stderr)
        return 2
    results = json.loads(args.results.read_text(encoding="utf-8"))
    config = load_yaml(args.thresholds)

    baseline = None
    if not args.no_baseline and args.baseline and args.baseline.is_file():
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))

    checks = run_checks(results, baseline, config)
    decision = verdict(checks)
    print(render_console(checks, results, decision))

    if args.markdown:
        from eval.report import render_markdown

        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(
            render_markdown(results, baseline, checks, decision), encoding="utf-8"
        )
        print(f"report -> {args.markdown}", file=sys.stderr)

    payload = {
        "decision": decision,
        "git_sha": results.get("git_sha"),
        "checks": [
            {
                "name": c.name, "status": c.status, "observed": c.observed,
                "limit": c.limit, "detail": c.detail, "waived": c.waived,
            }
            for c in checks
        ],
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary and args.markdown and args.markdown.exists():
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(args.markdown.read_text(encoding="utf-8"))

    if args.warn_only:
        return 0
    return 1 if decision == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
