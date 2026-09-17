"""Tests for the gate itself.

The gate is the only part of this repo that can block a deploy, so it is the
part that gets deterministic unit tests. Everything upstream of it involves an
LLM and is verified by the nightly rescan instead.
"""
from __future__ import annotations

import copy
from datetime import date, timedelta

import pytest

from eval.gate import run_checks, verdict

BASE_CONFIG = {
    "mode": "ci95",
    "absolute": {
        "faithfulness": {"min": 0.85, "blocking": True},
        "hallucination_rate": {"max": 0.10, "blocking": True},
        "context_precision": {"min": 0.70, "blocking": False},
    },
    "regression": {
        "faithfulness": {"max_drop": 0.03, "blocking": True},
        "hallucination_rate": {"max_rise": 0.05, "blocking": True},
    },
    "per_tag": {
        "unanswerable": {"refusal_accuracy": {"min": 0.90, "blocking": True}},
    },
    "health": {"judge_coverage_min": 0.95, "min_items": 20},
    "waivers": [],
}


def results(**overrides) -> dict:
    base = {
        "git_sha": "abc123",
        "judge_coverage": 1.0,
        "metrics": {
            "faithfulness": {"mean": 0.94, "ci95_low": 0.89, "ci95_high": 0.98, "n": 24},
            "hallucination_rate": {"mean": 0.04, "ci95_low": 0.0, "ci95_high": 0.09, "n": 24},
            "context_precision": {"mean": 0.81, "ci95_low": 0.75, "ci95_high": 0.87, "n": 20},
        },
        "per_tag": {"unanswerable": {"refusal_accuracy": 1.0, "n": 4}},
    }
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and key in merged:
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


BASELINE = {"metrics": {
    "faithfulness": {"mean": 0.94},
    "hallucination_rate": {"mean": 0.04},
}}


def test_healthy_run_passes():
    assert verdict(run_checks(results(), BASELINE, BASE_CONFIG)) == "PASS"


def test_faithfulness_floor_blocks():
    bad = results(metrics={"faithfulness": {"mean": 0.80, "ci95_low": 0.74,
                                            "ci95_high": 0.86, "n": 24}})
    checks = run_checks(bad, BASELINE, BASE_CONFIG)
    assert verdict(checks) == "FAIL"
    failed = [c for c in checks if c.status == "FAIL"]
    assert any(c.name == "absolute:faithfulness" for c in failed)


def test_ci_bound_is_what_gets_compared():
    """Mean above the floor but a wide CI dipping below it still blocks in ci95 mode."""
    wobbly = results(metrics={"faithfulness": {"mean": 0.87, "ci95_low": 0.79,
                                               "ci95_high": 0.95, "n": 24}})
    # No baseline here: this test is about the absolute check only, and a
    # baseline of 0.94 would trip the regression rule for unrelated reasons.
    assert verdict(run_checks(wobbly, None, BASE_CONFIG)) == "FAIL"

    mean_mode = {**BASE_CONFIG, "mode": "mean"}
    assert verdict(run_checks(wobbly, None, mean_mode)) != "FAIL"


def test_hallucination_rise_blocks_even_when_under_the_absolute_cap():
    """0.04 -> 0.095 is under the 0.10 cap but is a +0.055 regression."""
    risen = results(metrics={"hallucination_rate": {"mean": 0.095, "ci95_low": 0.02,
                                                    "ci95_high": 0.099, "n": 24}})
    checks = run_checks(risen, BASELINE, BASE_CONFIG)
    assert verdict(checks) == "FAIL"
    assert any(c.name == "regression:hallucination_rate" and c.status == "FAIL" for c in checks)


def test_non_blocking_metric_warns_only():
    low_precision = results(metrics={"context_precision": {"mean": 0.60, "ci95_low": 0.52,
                                                           "ci95_high": 0.68, "n": 20}})
    checks = run_checks(low_precision, BASELINE, BASE_CONFIG)
    assert verdict(checks) == "PASS_WITH_WARNINGS"
    assert any(c.name == "absolute:context_precision" and c.status == "WARN" for c in checks)


def test_slice_failure_blocks_while_aggregate_is_green():
    """The whole point of per-tag floors: 4 bad items out of 24 hide in the mean."""
    rotten_slice = results(per_tag={"unanswerable": {"refusal_accuracy": 0.50, "n": 4}})
    checks = run_checks(rotten_slice, BASELINE, BASE_CONFIG)
    assert verdict(checks) == "FAIL"
    assert any(c.name == "per_tag:unanswerable:refusal_accuracy" for c in checks
               if c.status == "FAIL")


def test_low_judge_coverage_blocks():
    """A judge that failed half the items is a broken measurement, not a green build."""
    partial = results(judge_coverage=0.62)
    checks = run_checks(partial, BASELINE, BASE_CONFIG)
    assert verdict(checks) == "FAIL"
    assert any(c.name == "judge_coverage" and c.status == "FAIL" for c in checks)


def test_truncated_run_blocks():
    small = results(metrics={"faithfulness": {"mean": 0.99, "ci95_low": 0.97,
                                              "ci95_high": 1.0, "n": 6}})
    assert verdict(run_checks(small, BASELINE, BASE_CONFIG)) == "FAIL"


def test_missing_metric_is_a_failure_not_a_pass():
    empty = results(metrics={})
    empty["metrics"] = {}
    checks = run_checks(empty, BASELINE, BASE_CONFIG)
    assert verdict(checks) == "FAIL"
    assert any(c.detail == "metric missing from results" for c in checks)


def test_no_baseline_degrades_to_absolute_only():
    checks = run_checks(results(), None, BASE_CONFIG)
    assert verdict(checks) == "PASS_WITH_WARNINGS"
    assert any(c.name == "regression:baseline" for c in checks)


@pytest.mark.parametrize("scope,expected", [("absolute", "WAIVED"), ("regression", "FAIL")])
def test_waiver_applies_only_to_its_own_scope(scope, expected):
    config = copy.deepcopy(BASE_CONFIG)
    config["waivers"] = [{
        "metric": "faithfulness", "scope": scope,
        "until": date.today() + timedelta(days=7),
        "owner": "@mjouhari", "reason": "chunker rewrite in flight",
    }]
    bad = results(metrics={"faithfulness": {"mean": 0.80, "ci95_low": 0.74,
                                            "ci95_high": 0.86, "n": 24}})
    checks = {c.name: c.status for c in run_checks(bad, BASELINE, config)}
    assert checks["absolute:faithfulness"] == ("WAIVED" if scope == "absolute" else "FAIL")
    assert expected in checks.values()


def test_expired_waiver_is_itself_a_failure():
    config = copy.deepcopy(BASE_CONFIG)
    config["waivers"] = [{
        "metric": "faithfulness", "scope": "absolute",
        "until": date.today() - timedelta(days=1),
        "owner": "@mjouhari", "reason": "stale",
    }]
    checks = run_checks(results(), BASELINE, config)
    assert verdict(checks) == "FAIL"
    assert any("expired" in c.detail for c in checks)


def test_unowned_waiver_is_rejected():
    config = copy.deepcopy(BASE_CONFIG)
    config["waivers"] = [{"metric": "faithfulness", "scope": "absolute",
                          "until": date.today() + timedelta(days=7), "reason": "no owner"}]
    assert verdict(run_checks(results(), BASELINE, config)) == "FAIL"
