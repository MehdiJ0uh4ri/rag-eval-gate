# CI quirks

Things that cost time to discover. Each one is a decision the code depends on.

## RAGAS defaults to OpenAI, silently

`from ragas.metrics import faithfulness` gives you a *module-level singleton*
with no LLM bound. If the global RAGAS config was never set, it constructs an
OpenAI client at evaluation time — so a repo that has never mentioned OpenAI
starts failing on a missing `OPENAI_API_KEY`, or worse, quietly bills a second
vendor because someone had the variable in their shell.

We instantiate the metric **classes** instead (`Faithfulness(llm=judge)`), which
makes the binding explicit and impossible to forget. See `eval/metrics.py`.

## Anthropic has no embeddings endpoint

`answer_relevancy` and `answer_correctness` need embeddings. There is no
Anthropic embeddings API, so a Claude-only eval stack is not possible if you
want those two metrics.

The options were: a second API vendor (rejected — a second key, a second
failure mode, and a second bill for a component that only computes cosine
similarity), or a local sentence-transformers model (chosen). Local also means
the embedding half of every metric is *perfectly reproducible*, which matters
when the gate resolves 0.03 deltas.

The cost is a ~500 MB model download. `ci/bootstrap.sh` pre-fetches it and the
workflow caches `~/.cache/huggingface` keyed on `ci/models.env`, so it is a
one-time cost per model pin.

## `temperature=0` is a 400 on current Claude models

The reflex for a judge LLM is `temperature=0`. Sampling parameters
(`temperature`, `top_p`, `top_k`) were **removed** on Claude Opus 5, Sonnet 5,
and the Opus 4.7/4.8 family — sending one returns
`400 invalid_request_error`. `langchain_anthropic.ChatAnthropic` will happily
forward whatever you pass it, so this surfaces as a wall of 400s halfway
through a paid run.

Determinism therefore comes from elsewhere:

- `--repeats 3` and a **median** per item, so a single flaky verdict is
  outvoted rather than averaged in;
- `output_config.effort: low` on the judge, passed via `model_kwargs`;
- a fixed bootstrap seed in `eval/run_eval.py`, so the CI bounds a given set of
  scores produces are byte-identical across runs.

## A judge failure must not score zero

RAGAS with `raise_exceptions=False` returns `NaN` for an item its judge could
not score. Both obvious handlings are wrong:

- score it `0` → an API blip fails the build and everyone learns to re-run red
  builds, which is how a gate dies;
- score it `1` → a judge that is quietly failing half the set produces a green
  build.

We drop `NaN`s from the aggregate and gate `judge_coverage` (fraction of
`(item, metric)` pairs that produced a number) separately at `>= 0.95`. A broken
measurement blocks with its own message, distinct from a quality regression.

## Rare-event rates cannot be gated on a confidence interval

The gate compares the conservative bound of a bootstrap CI rather than the point
estimate, which stops a two-question wobble from turning red. That is right for
continuous metrics and wrong for `hallucination_rate`: on 24 binary items, one
bad answer is a mean of 0.042 and a CI upper bound of 0.125, so a `<= 0.10` cap
on the bound demands a literal zero.

`hallucination_rate` therefore carries `compare: mean` in
`eval/thresholds.yaml`. Resolving a 10% rate through a CI needs roughly 100
items; when the golden set gets there, flip it back.

This was found by running the gate against its own seed baseline — worth doing
after any threshold change, and it costs nothing (`make gate` re-reads an
existing artifact).

## Prompt caching pays for itself on the generator, not the judge

The generator's system prompt is byte-identical across all 24 questions, so it
sits behind a `cache_control` breakpoint with the per-question context after it.
On a 3-repeat run that is ~28k cached input tokens out of ~41k.

The judge gets no such benefit: RAGAS builds a different prompt per claim, so
there is no stable prefix to cache. Judge cost scales with `repeats × items ×
claims`, which is the real reason `--repeats` defaults to 1 locally and 3 in CI
rather than something larger.

## `dataclass` defaults freeze env vars at import time

`top_k: int = int(os.getenv("RAG_TOP_K", "4"))` evaluates **once**, when the
module is imported. Anything that sets the env var afterwards — a test, a
`make` target, a shell wrapper — is silently ignored, and the eval reports a
config fingerprint that does not match what actually ran.

All of `RagConfig` uses `field(default_factory=...)`. Caught by
`test_unknown_prompt_variant_fails_loudly`, which is a strange-looking test
until you know why it exists.

## `--baseline ''` is not "no baseline"

`Path('')` is `Path('.')`, which exists, which means an empty `--baseline`
resolved to "read the current directory as JSON". The smoke gate passes an
explicit `--no-baseline` flag instead, and the check is `is_file()` rather than
`exists()`.

## Pin RAGAS hard

RAGAS renames result columns between minor versions
(`context_precision` → `llm_context_precision_with_reference` was one). A rename
does not error — the column simply is not there, our normalizer skips it, and
the metric silently disappears from the aggregate. The gate does catch this
(`absolute:<metric>` fails with "metric missing from results"), but only because
a missing metric is treated as a failure rather than a skip.

`requirements-eval.txt` pins exact versions, `eval/metrics.py` maps both the old
and new column names, and bumping the pin is a dedicated PR that re-baselines.
Never a dependabot automerge.
