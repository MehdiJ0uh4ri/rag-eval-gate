# Runbook

## The gate blocked my PR

1. Open the PR comment. Start at **Blocking failures**, then **Worst items** —
   the actual question, the actual answer, and the chunks that were retrieved.
2. Read the shape, not just the number:

   | Symptom | Likely cause | Where to look |
   |---|---|---|
   | `retrieval_hit_rate` down | chunker or `top_k` change, corpus file renamed | `app/index.py`, `RAG_CHUNK_*`, `test_pipeline.py` retrieval test |
   | `context_recall` down, `faithfulness` flat | retriever is missing the right passage | same as above |
   | `faithfulness` down, `context_recall` flat | prompt or model change; the context was there and was not used | `app/config.py` prompt variants |
   | `refusal_accuracy` down on `unanswerable` | prompt loosened; the "reply exactly" instruction weakened | `PROMPT_VARIANTS["grounded"]` |
   | `judge_coverage` down | measurement broke, not quality — API errors, rate limits, timeouts | job log for 429/5xx, `JUDGE_MAX_WORKERS` |
   | everything moved a little | judge drift or a model rotation | `ci/models.env`, last nightly rescan |

3. Reproduce locally without re-spending generation tokens:

   ```bash
   gh run download <run-id> -n rag-eval-<run-id> -D artifacts
   make rejudge      # re-scores the cached answers
   make gate         # re-applies thresholds; free, no API calls
   ```

4. Fix it, or — if the regression is known, tracked, and acceptable — add a
   dated waiver (`docs/thresholds.md`). Do not lower the floor to get green.

## The gate is flaky

Before treating it as noise, check the nightly rescan: it runs the identical
eval on an unchanged main and is the measurement of how much this pipeline
wobbles on its own.

- spread within ~0.02 on `faithfulness` → expected; raise `EVAL_REPEATS` to 5
  before touching a threshold;
- spread larger than that → the judge is the problem. Check `judge_coverage`
  first (partial failures show up as movement), then whether the model pin in
  `ci/models.env` actually changed.

Raising `--repeats` costs linearly and only shrinks judge variance. It does
nothing for *generator* variance, which is measured by re-running `make eval`
end to end.

## I need to add golden questions

1. Write the answer **from the corpus**, not from a model. A model-authored
   ground truth bakes today's behaviour into the thing meant to detect
   behaviour change.
2. Keep the slice ratios (~60% factual / ~20% multi-hop / ~20% unanswerable).
   `golden/loader.py` enforces the unanswerable floor.
3. Add `must_not_say` entries for the specific wrong answers the question
   invites. Verify the phrase does not appear in your ground truth — the
   validator checks this, because a phrase in both makes a perfect answer fail.
4. `make validate-golden && make test` — the retrieval test proves the new
   question is answerable by the retriever we actually ship.
5. Adding items changes the aggregate, so **re-baseline in the same PR**. Say
   so in the PR body; a metric delta from a dataset change is not a regression
   and the report cannot tell the difference on its own.

## Rotating the judge or generator model

See `docs/thresholds.md` → "Changing the judge model". One PR, `ci/models.env`
only, baseline committed alongside.

## The eval is too expensive

In order of what to try:

1. `EVAL_REPEATS=1` on draft PRs; keep 3 on ready-for-review.
2. Confirm generator prompt caching is working — `generator_cache_read_tokens`
   in the report should be roughly two-thirds of input tokens. If it is zero, a
   byte changed in the system prompt (see `docs/ci-quirks.md`).
3. Drop the two advisory metrics (`answer_correctness`, `context_precision`)
   from `eval/metrics.py`. `answer_correctness` is the most expensive of the
   set — it makes both an LLM call and an embedding pass.
4. Do **not** shrink the golden set. A cheaper eval that cannot resolve a
   regression costs more than it saves.

## Break glass: ship without the gate

There isn't a bypass flag, and that is deliberate — a bypass flag becomes the
normal path within a quarter. The options are, in order of preference:

1. a dated waiver for the specific metric (leaves the rest of the gate armed);
2. `blocking: false` on the specific rule, in a PR that says when it flips back;
3. an admin merge, which is visible in the audit log and requires someone to
   own it by name.

## Onboarding: prove the gate actually blocks

```bash
make demo-regression
```

Runs the eval with the deliberately ungrounded prompt variant and `top_k=1`,
then gates it. Expected output is **BLOCKED** on the faithfulness floor and the
hallucination-rate rise. Worth running on day one — a gate nobody has seen fail
is a gate nobody believes in.
