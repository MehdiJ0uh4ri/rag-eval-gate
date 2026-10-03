# Why these numbers

A threshold nobody can justify gets lowered the first time it goes red. This is
the justification for each one, so the argument happens here rather than in a
PR at 6pm on a Friday.

## The metrics

| Metric | What it actually measures | Why it is gated |
|---|---|---|
| `faithfulness` | share of claims in the answer that are entailed by the retrieved context | the hallucination measure; the one number that maps to "the assistant made something up" |
| `hallucination_rate` | share of *items* that hallucinated, by any of three signals | faithfulness is a mean and hides shape: 0.90 could be every answer slightly loose, or nine perfect answers and one fabrication. Those are very different products |
| `context_recall` | share of the ground-truth answer that the retrieved context supports | separates a retrieval bug from a generation bug. A faithfulness drop with recall intact is a prompt problem; both dropping is a retriever problem |
| `answer_relevancy` | whether the answer addresses the question asked | catches the failure mode where a model refuses politely and scores perfectly on faithfulness |
| `context_precision` | share of retrieved chunks that were useful | advisory. Moves a lot with `top_k` tuning and is a cost signal more than a quality one |
| `answer_correctness` | agreement with the reference answer | advisory. Overlaps the others and is the noisiest of the set |
| `retrieval_hit_rate` | did the known-correct document appear in the retrieved set | deterministic, no LLM involved. When this drops, nothing downstream is interpretable |
| `refusal_accuracy` | refused when it should, answered when it should | the unanswerable slice's own metric |

## `faithfulness >= 0.85`

Observed on main is ~0.94 with a CI of roughly [0.89, 0.98]. The floor sits a
little below the CI's lower bound: high enough that a real regression trips it,
low enough that ordinary run-to-run variation does not.

It is *not* set to "current minus epsilon". A floor that tracks the last run is
a ratchet in whichever direction the model happened to drift.

## `hallucination_rate <= 0.10`

An item counts as hallucinating if **any** of three independent signals fires:

1. judged faithfulness below 0.50 — at least half the claims unsupported;
2. a `must_not_say` phrase from the golden dataset appears — hand-curated, no
   LLM in the loop, catches the specific wrong answers this corpus invites;
3. it answered an `unanswerable` question at all.

Three signals rather than one because each has a distinct blind spot. The judge
misses a fabricated number that happens to look plausible against the context;
the banned-phrase list only catches failures someone already thought of; the
unanswerable check only covers 4 of 24 items. The union is strictly stronger
than the best of them.

0.10 on 24 items means "at most two". That is coarse — it is the honest limit of
a 24-item set, and the reason the dataset should grow. See `compare: mean` in
`eval/thresholds.yaml` and the CI note in `ci-quirks.md`.

## Regression rules exist because floors are not enough

`faithfulness` sliding 0.94 → 0.86 passes every absolute floor and is obviously
a regression. `max_drop: 0.03` catches it while staying above the noise the
repeat-median leaves behind (observed run-to-run spread on an unchanged main is
~0.01–0.02, which is what the nightly rescan is for — it measures that spread so
this number can be set from data rather than taste).

Regression rules compare **means**, not CI bounds: comparing two conservative
bounds double-counts the uncertainty and makes the rule far stricter than it
reads.

## Per-slice floors

The `unanswerable` slice is 4 of 24 items. It can go from perfect to entirely
broken while aggregate faithfulness moves by 0.04 — well inside every absolute
floor and every regression rule. It gets its own floors for exactly that reason,
and it is the slice most likely to break, because "refuse when you do not know"
is the first behaviour a loosened prompt loses.

## Blocking vs advisory

`blocking: false` metrics are reported, coloured, and tracked, but do not fail
the build. `context_precision` and `answer_correctness` are advisory because
they move for legitimate reasons (a `top_k` change, a reworded reference answer)
often enough that blocking on them would train people to ignore the gate.

The test for whether something should block: *if this goes red, would we hold
the deploy?* If the honest answer is "we would look at it tomorrow", it is
advisory.

## Waivers, not threshold edits

When a known regression must ship, add a waiver with an owner, an expiry, and a
reason:

```yaml
waivers:
  - metric: context_precision
    scope: absolute
    until: 2026-09-15
    owner: "@mjouhari"
    reason: "chunker rewrite in flight (PLAT-812); precision recovers after merge"
```

The gate refuses to honour an expired or unowned waiver — it reports it as a
failure. That is deliberate: an expired waiver is a decision that was supposed
to be revisited, and the alternative (silently continuing to skip a check) is
how a gate rots.

Lowering a floor is the option of last resort, and it needs a note in this file
saying what changed about the product.

## Changing the judge model

A different judge scores differently even on identical answers, so the baseline
becomes meaningless. Rotating the model is a single PR that:

1. changes `ci/models.env` only;
2. runs the eval on the *unchanged* main corpus and prompt;
3. commits the new baseline in the same PR;
4. records the before/after in the PR body.

Never bundle a judge rotation with a prompt or retrieval change. You lose the
ability to attribute the delta to either.
