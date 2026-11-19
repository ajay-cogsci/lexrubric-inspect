# Protocol and provenance

This port implements the original [LexRubric paper v1](https://arxiv.org/abs/2606.09389v1). The authors' released evaluator supplies execution details that the paper does not specify. The model-specific exceptions used by the included configurations are documented below.

## Pinned inputs

| Component | Revision |
|---|---|
| Original 649-question corpus | `13861f99b2b268dbdffad7a95281e4ea27fe172d` |
| Released evaluator, prompt and upstream provenance files | `a65cdbec3f6060906407cf391d2c69446e363d83` |

Within `lexrubric_inspect/`, `data/` holds the original paper corpus, including zero-point criteria. `upstream/data/` retains the later release solely for provenance; evaluation does not use those later weights. Both manifests record source URLs, hashes and byte sizes. Verification checks every pinned file, including the upstream license and README, before evaluation.

## Requests and scoring

| Setting | Behavior |
|---|---|
| Candidate input | Original question as exactly one user message; no added system prompt, reference answer, rubric or tools |
| Candidate output cap | 16,384 tokens |
| Candidate temperature | 0.6, except the explicitly recorded model-specific omissions |
| Judge | Qwen3.6-27B, OpenRouter's pinned Alibaba provider, no provider fallback |
| Judge sampling | Temperature 0; 4,096-token cap; thinking disabled |
| Judging unit | One independent call per criterion, including zero-point criteria |
| Template | Exact released Chinese bytes and substitutions |
| Extraction | Original answer extraction and removal of text through the first `</think>` |
| Retries/parsing | Original HTTP and JSON-parsing retries; no parser repairs or extra retry layer |
| Candidate failure | Original zero-score fallback |
| Judge failure | Original false-verdict fallback |
| Question score | 100 × signed points for met criteria / total positive rubric points |
| Aggregate | Arithmetic mean of question scores; negative penalties retained; no clipping |
| Runner ordering | Generate before grading within each split; five candidate workers and ten rubric workers per graded response |

The released script reports mean raw points; the paper specifies normalized score rates. The adapter retains raw points and reports the paper metric. A dimension with no positive denominator has an undefined rate and is omitted from that dimension's mean, following the authors' clarification in [v2 §4.1](https://arxiv.org/html/2606.09389v2#S4.SS1). This clarification does not change overall question scores.

The paper does not fix a seed, candidate reasoning effort, deployment precision or provider snapshot. Native candidate reasoning defaults are retained except the recorded GLM-5.3 high-effort setting. No `seed`, `top_p` or `top_k` is supplied. OpenAI token-field translation, Anthropic's native request format and OpenRouter's thinking flag are transport adaptations; the request audit records transmitted parameters. Anthropic requests pin API version `2026-10-21`. Model identifiers are API aliases, not dated weight snapshots.

## Recorded configurations

| Configuration | Candidate exception |
|---|---|
| `configs/qwen4-max.json` | None; temperature 0.6 and native default reasoning |
| `configs/gpt-6.1-sol.json` | Omit unsupported temperature; native default reasoning |
| `configs/gpt-astra.json` | Omit unsupported temperature; native default reasoning |
| `configs/claude-fable-5-2.json` | Omit unsupported temperature; native default reasoning |
| `configs/claude-opus-5-5.json` | Omit unsupported temperature; native default reasoning |
| `configs/glm-5.3-high.json` | Approved `reasoning_effort=high`; temperature remains 0.6 |

These files are under `lexrubric_inspect/configs/` and retain the original approval and exception records. The runner requires those records for supported parameter exceptions. The default configuration selects Astra with its recorded exception; select a configuration explicitly for reproducible runs. Judge substitutions are opt-in protocol variants, never automatic fallbacks.

## Direct Inspect usage

For the paper-temperature Qwen4-Max configuration, set `OPENROUTER_API_KEY` securely, then:

```bash
export LEXRUBRIC_CANDIDATE_PROVIDER=openrouter
export LEXRUBRIC_CANDIDATE_CHAT_URL=https://openrouter.ai/api/v1/chat/completions
export LEXRUBRIC_CANDIDATE_API_KEY_ENV=OPENROUTER_API_KEY
export LEXRUBRIC_JUDGE_PROVIDER=openrouter
export LEXRUBRIC_JUDGE_CHAT_URL=https://openrouter.ai/api/v1/chat/completions
export LEXRUBRIC_JUDGE_API_KEY_ENV=OPENROUTER_API_KEY
export LEXRUBRIC_ALLOW_TEMPERATURE_OMISSION=false
unset LEXRUBRIC_CANDIDATE_REASONING_EFFORT

uv run inspect eval lexrubric_inspect/tasks.py@lexrubric \
  --model lexrubric_raw/qwen/qwen4-max -T split=all
```

This invokes the underlying task directly. It does not reproduce the canonical runner's two-phase generation/grading schedule, full-run log checks or configuration validation. Use the README's `lexrubric-run` commands to reproduce the reported execution procedure. Additional sampling settings, prompts, tools or retries would change the protocol.

## Error interpretation

Errors and raw judge responses are retained. A false fallback for a failed negative criterion can avoid a penalty, so failures must not be silently described as conservative scoring. No completed failed grade is automatically replaced on resume. Regrading saved answers requires a fresh run and explicitly selected configuration.

`benchmark_results_valid=false` is an adapter diagnostic when terminal errors occur, not a paper-defined invalidation rule. It does not erase the reported scores or turn an exception-bearing run into an exact reproduction of the paper's published leaderboard. Native Inspect token/cost totals are incomplete because the original evaluator returns text; HTTP usage is retained when supplied by the provider.
