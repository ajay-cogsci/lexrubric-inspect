# LexRubric for Inspect

An Inspect implementation of [LexRubric v1](https://arxiv.org/abs/2606.09389v1), a Chinese open-ended legal benchmark with **473 consultation questions, 176 exam questions, and 12,337 rubric criteria**.

The original corpus, evaluator and Chinese judge template are pinned and verified by SHA-256. The adapter preserves the released evaluator's extraction, retries, parsing and failure behavior, and reports the paper's normalized question scores. See [protocol and provenance](docs/PROTOCOL.md) for the exact settings and interpretation.

## Installation

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --frozen --python 3.12
uv run python -m unittest discover -s tests -v
uv run inspect list tasks lexrubric_inspect/lexrubric_port.py
uv run lexrubric-verify --output-dir verification
```

Runtime versions are pinned to `inspect-ai==0.3.286` and `requests==2.34.2`; `agent-client-protocol==0.12.1` preserves compatibility with this Inspect release. `uv.lock` records the resolved environment. The offline verifier checks all source records and uses mocked HTTP for two synthetic pipeline samples. It makes no paid requests.

## Run the benchmark

Set `OPENROUTER_API_KEY` in the environment for the fixed **Qwen3.6-27B** judge. Sol also requires `OPENAI_API_KEY`; the other configurations name their required key variables. Keep credentials in environment variables.

The canonical runner generates and then grades each split, saves recoverable checkpoints, and writes completed full-run logs under the selected output directory:

```bash
uv run lexrubric-run \
  --config lexrubric_inspect/configs/qwen4-max.json \
  --run-name qwen4-full --output-dir runs

uv run lexrubric-run \
  --config lexrubric_inspect/configs/gpt-6.1-sol.json \
  --run-name sol-full --output-dir runs
```

These commands make paid model requests. Omit `--limit` for all 649 questions; `--limit 1` is a separate pilot with one question per split. Choose a new run name for each independent evaluation.

The discoverable task is `lexrubric_inspect/lexrubric_port.py@lexrubric`, with `split=all|consultation|exam` and `judge_model=qwen3.6-27b`. It uses the custom `lexrubric_raw/<candidate-model>` provider. [Direct Inspect usage](docs/PROTOCOL.md#direct-inspect-usage) requires explicit endpoint/provider settings; the canonical runner supplies the recorded generation-before-grading workflow.

## Results

Two complete Qwen-judged evaluations are included in `results/logs/`, each as a consultation/exam pair covering all 649 questions.

| Candidate | Overall | Consultation | Exam | Failed answers | Failed judge verdicts |
|---|---:|---:|---:|---:|---:|
| Qwen4-Max | 75.69% | 71.24% | 87.64% | 13 | 27 |
| GPT-6.1 Sol | 65.51% | 61.99% | 74.98% | 0 | 1 |

Overall is the mean of all 649 question scores. Judge failures are criterion judgments unresolved after the original retries, excluding recovered request errors. The released failure handling remains in the scores. Qwen4-Max retains temperature 0.6; Sol uses its recorded temperature omission. See [the evaluation report](docs/EVALUATION_REPORT.md) and [result manifest](results/manifest.json) for exact scores, hashes and coverage.

View the included full-run logs locally:

```bash
uv run inspect view --log-dir results/logs
```

## Repository contents

- `lexrubric_inspect/`: task, provider adapter, canonical runner, pinned assets and six recorded candidate configurations.
- `tests/`: offline protocol, provider, preflight, recovery and publication checks.
- `docs/`: protocol, provenance and evaluation report.
- `results/`: complete evaluation logs, scores, coverage and checksums.

## License and citation

The adapter uses the MIT license. Original benchmark authorship and the upstream MIT notice are retained in [NOTICE](NOTICE) and [the upstream license](lexrubric_inspect/upstream/LICENSE). Please cite the [LexRubric paper](https://arxiv.org/abs/2606.09389v1); its original citation is retained in [the upstream README](lexrubric_inspect/upstream/README.md).
