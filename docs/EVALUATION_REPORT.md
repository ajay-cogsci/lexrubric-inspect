# Full-run evaluation report

These results use Qwen3.6-27B as judge through OpenRouter's Alibaba provider. Each candidate completed the 649-question evaluation as two original splits: consultation (473) and exam (176). The evidence consists of four Inspect `.eval` archives, with both splits required for each model.

| Candidate | Overall (%) | Consultation (%) | Exam (%) | Failed answers | Terminal judge failures |
|---|---:|---:|---:|---:|---:|
| Qwen4-Max (`qwen/qwen4-max`) | 75.6882659328 | 71.2422207178 | 87.6370124483 | 13 | 27 |
| GPT-6.1 Sol (`gpt-6.1-sol`) | 65.5122687892 | 61.9906527508 | 74.9766118925 | 0 | 1 |

Overall weights the two split means by 473 and 176, respectively. It is not their unweighted average. Failure counts are terminal outcomes after the original retries. Recovered transient HTTP failures are excluded from the failure columns.

## Settings and exceptions

Both candidates retain the 16,384-token output cap, original question-only input and native default reasoning. Qwen4-Max uses temperature 0.6. Sol omits temperature under the recorded approval because its API does not support the paper's value. Both use the same judge settings: temperature 0, maximum 4,096 tokens, thinking disabled, one independent judgment per criterion and the original retry/parsing behavior.

The source configurations are included under `lexrubric_inspect/configs/`. Actual settings and provider audit records are also retained inside the logs.

## Limitations

Qwen4-Max has 13 failed consultation answers and 27 unresolved criterion judgments (26 consultation, one exam). Sol has no failed answers and one unresolved consultation judgment. The released evaluator assigns zero to a failed answer and false to an unresolved judgment; those outcomes remain in these scores. The diagnostic `benchmark_results_valid=false` remains recorded for both models.

These runs demonstrate full dataset coverage with disclosed exceptions and failures. They do not establish numerical reproduction of the paper's model leaderboard. Failed responses and judgments remain in the archives and reported scores; no samples are removed.

## Evidence verification

The [result manifest](../results/manifest.json) records archive checksums, sample coverage, per-sample scores and failure counts. The unchanged Inspect archives are stored as:

```text
results/logs/
  gpt-6.1-sol/
    consultation.eval
    exam.eval
  qwen4-max/
    consultation.eval
    exam.eval
```

Both split files are required to reconstruct a candidate's overall result. Each consultation log contains 473 samples and each exam log contains 176; together they cover all 649 original question IDs. Use `uv run inspect view --log-dir results/logs` from the repository root to inspect the logs.
