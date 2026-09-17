"""Markdown report -- posted as the PR comment and the job summary.

Written for the reviewer who has 30 seconds: verdict, what moved, and the worst
five items with their actual text. A gate that only prints "faithfulness 0.83"
gets ignored; one that shows the sentence the model made up gets fixed.
"""
from __future__ import annotations

from eval.metrics import ALL_METRICS, LOWER_IS_BETTER

VERDICT_BADGE = {
    "PASS": "PASS",
    "PASS_WITH_WARNINGS": "PASS (warnings)",
    "FAIL": "BLOCKED",
}
MARKER = "<!-- rag-eval-gate -->"  # lets CI update its comment instead of spamming


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _delta(current, previous, metric) -> str:
    if current is None or previous is None:
        return "-"
    delta = current - previous
    if abs(delta) < 0.0005:
        return "0.000"
    good = (delta < 0) if metric in LOWER_IS_BETTER else (delta > 0)
    return f"{delta:+.3f} {'▲' if good else '▼'}"


def render_markdown(results: dict, baseline: dict | None, checks: list, decision: str) -> str:
    metrics = results.get("metrics", {})
    base_metrics = (baseline or {}).get("metrics", {})
    failures = [c for c in checks if c.status == "FAIL"]
    warnings = [c for c in checks if c.status in ("WARN", "WAIVED")]

    out = [MARKER, f"## RAG quality gate — **{VERDICT_BADGE.get(decision, decision)}**", ""]

    cfg = results.get("config", {})
    out += [
        f"`{results.get('git_sha', '?')[:12]}` · {results.get('metrics', {}).get('faithfulness', {}).get('n', 0)} golden items · "
        f"{results.get('repeats')} judge pass(es) · judge `{cfg.get('judge_model')}` · "
        f"generator `{cfg.get('generator_model')}` · prompt `{cfg.get('prompt_variant')}` · top_k `{cfg.get('top_k')}`",
        "",
    ]

    if failures:
        out += ["### Blocking failures", ""]
        for c in failures:
            out.append(f"- **{c.name}** — observed `{_fmt(c.observed)}`, limit `{_fmt(c.limit)}`. {c.detail}")
        out.append("")

    out += [
        "### Metrics",
        "",
        "| Metric | This run | 95% CI | Baseline | Δ | Floor |",
        "|---|---|---|---|---|---|",
    ]
    absolute = {c.metric: c.limit for c in checks if c.kind == "absolute"}
    for metric in ALL_METRICS:
        summary = metrics.get(metric) or {}
        if summary.get("mean") is None:
            continue
        base = (base_metrics.get(metric) or {}).get("mean")
        ci = f"[{_fmt(summary.get('ci95_low'))}, {_fmt(summary.get('ci95_high'))}]"
        out.append(
            f"| `{metric}` | **{_fmt(summary['mean'])}** | {ci} | {_fmt(base)} | "
            f"{_delta(summary['mean'], base, metric)} | {_fmt(absolute.get(metric))} |"
        )
    out.append("")

    per_tag = results.get("per_tag", {})
    if per_tag:
        out += ["<details><summary>Per-slice breakdown</summary>", "",
                "| Slice | n | faithfulness | hallucination | refusal acc. | ctx recall |",
                "|---|---|---|---|---|---|"]
        for tag, values in sorted(per_tag.items()):
            out.append(
                f"| `{tag}` | {values.get('n', 0)} | {_fmt(values.get('faithfulness'))} | "
                f"{_fmt(values.get('hallucination_rate'))} | {_fmt(values.get('refusal_accuracy'))} | "
                f"{_fmt(values.get('context_recall'))} |"
            )
        out += ["", "</details>", ""]

    worst = sorted(
        results.get("per_item", []),
        key=lambda r: (
            -len(r.get("flags", [])),
            r.get("metrics", {}).get("faithfulness", 1.0),
        ),
    )
    offenders = [r for r in worst if r.get("flags") or r.get("metrics", {}).get("faithfulness", 1.0) < 0.85][:5]
    if offenders:
        out += ["### Worst items", ""]
        for row in offenders:
            flags = ", ".join(f"`{f}`" for f in row.get("flags", [])) or "—"
            faith = row.get("metrics", {}).get("faithfulness")
            out += [
                f"<details><summary><code>{row['id']}</code> — faithfulness {_fmt(faith)} — {flags}</summary>",
                "",
                f"**Q:** {row['question']}",
                "",
                f"**A:** {row['answer'][:700]}",
                "",
                f"**Retrieved:** {', '.join(f'`{c}`' for c in row.get('context_ids', [])) or '—'}",
                "",
                "</details>",
                "",
            ]

    if warnings:
        out += ["<details><summary>Non-blocking warnings and waivers</summary>", ""]
        for c in warnings:
            note = f" — WAIVED: {c.waived}" if c.waived else ""
            out.append(f"- `{c.name}` ({c.status}): {c.detail}{note}")
        out += ["", "</details>", ""]

    usage = results.get("usage", {})
    out += [
        "<details><summary>Run cost & latency</summary>",
        "",
        f"- generator tokens: {usage.get('generator_input_tokens', 0)} in "
        f"({usage.get('generator_cache_read_tokens', 0)} cached) / "
        f"{usage.get('generator_output_tokens', 0)} out",
        f"- judge passes: {usage.get('judge_passes')} over {usage.get('items')} items",
        f"- generation p95 latency: {results.get('latency_ms_p95', 0)} ms",
        f"- judge coverage: {_fmt(results.get('judge_coverage'))}",
        "",
        "</details>",
        "",
        "_Thresholds live in `eval/thresholds.yaml`. To change one, open a PR — "
        "`@platform-quality` owns that file. To ship past a known, tracked "
        "regression, add a dated waiver rather than lowering the floor._",
    ]
    return "\n".join(out)
