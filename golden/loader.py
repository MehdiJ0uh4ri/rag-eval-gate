"""Load + validate the golden dataset.

Runs as a standalone check (`make validate-golden`) so a malformed dataset
fails the PR in seconds instead of after a full paid eval run.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DATASET = Path(__file__).parent / "qa.yaml"
CORPUS = Path(__file__).parent.parent / "app" / "corpus"

REQUIRED = {"id", "question", "ground_truth", "reference_ids", "tags"}
KNOWN_TAGS = {
    "factual",
    "multi-hop",
    "unanswerable",
    "error-codes",
    "limits",
    "timing",
    "security",
    "reliability",
    "scope",
    "states",
    "fees",
    "recovery",
}
REFUSAL_SENTINEL = "i do not have that in the documentation"

# Slice floors. A dataset that drifts away from these stops being a gate and
# starts being a vibe check -- unanswerables are the only items that can catch
# a model that answers everything confidently.
MIN_ITEMS = 20
MIN_UNANSWERABLE_RATIO = 0.15


@dataclass(frozen=True)
class GoldenItem:
    id: str
    question: str
    ground_truth: str
    reference_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    must_not_say: list[str] = field(default_factory=list)

    @property
    def unanswerable(self) -> bool:
        return "unanswerable" in self.tags


def load(path: Path = DATASET) -> list[GoldenItem]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a top-level list of items")
    return [
        GoldenItem(
            id=item["id"],
            question=item["question"].strip(),
            ground_truth=item["ground_truth"].strip(),
            reference_ids=list(item.get("reference_ids") or []),
            tags=list(item.get("tags") or []),
            must_not_say=[s.lower() for s in (item.get("must_not_say") or [])],
        )
        for item in raw
    ]


def validate(path: Path = DATASET) -> list[str]:
    errors: list[str] = []
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return [f"{path}: expected a top-level list of items"]

    corpus_files = {p.name for p in CORPUS.glob("*.md")}
    seen: set[str] = set()

    for i, item in enumerate(raw):
        where = f"item[{i}] id={item.get('id', '?')!r}"
        missing = REQUIRED - set(item)
        if missing:
            errors.append(f"{where}: missing keys {sorted(missing)}")
            continue
        if item["id"] in seen:
            errors.append(f"{where}: duplicate id")
        seen.add(item["id"])

        unknown = set(item["tags"]) - KNOWN_TAGS
        if unknown:
            errors.append(f"{where}: unknown tags {sorted(unknown)}")

        for ref in item["reference_ids"]:
            if ref not in corpus_files:
                errors.append(f"{where}: reference_ids points at missing doc {ref!r}")

        unanswerable = "unanswerable" in item["tags"]
        gt_is_refusal = REFUSAL_SENTINEL in item["ground_truth"].lower()
        if unanswerable and not gt_is_refusal:
            errors.append(f"{where}: tagged unanswerable but ground_truth is not the refusal sentinel")
        if gt_is_refusal and not unanswerable:
            errors.append(f"{where}: ground_truth is a refusal but the item is not tagged unanswerable")
        if unanswerable and item["reference_ids"]:
            errors.append(f"{where}: unanswerable items must have empty reference_ids")
        if not unanswerable and not item["reference_ids"]:
            errors.append(f"{where}: answerable items need at least one reference doc")

        # Guard against the most common authoring mistake: a must_not_say
        # substring that also appears in the ground truth, which would make the
        # hallucination check fire on a perfect answer.
        for banned in item.get("must_not_say") or []:
            if banned.lower() in item["ground_truth"].lower():
                errors.append(f"{where}: must_not_say {banned!r} appears in ground_truth")

    if len(raw) < MIN_ITEMS:
        errors.append(f"dataset has {len(raw)} items, minimum is {MIN_ITEMS}")
    ratio = sum("unanswerable" in i.get("tags", []) for i in raw) / max(len(raw), 1)
    if ratio < MIN_UNANSWERABLE_RATIO:
        errors.append(
            f"unanswerable slice is {ratio:.0%}, minimum is {MIN_UNANSWERABLE_RATIO:.0%}"
        )
    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("golden dataset INVALID:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    items = load()
    tags = sorted({t for i in items for t in i.tags})
    print(f"golden dataset OK: {len(items)} items, tags: {', '.join(tags)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
