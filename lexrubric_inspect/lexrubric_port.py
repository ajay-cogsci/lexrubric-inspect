"""Inspect adapter for LexRubric v1; upstream inference/judging is unmodified.

The original paper controls the dataset and metrics. The authors' released
Chinese template and evaluator supply implementation details omitted by v1.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType
from urllib.parse import urlsplit

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig, ModelAPI, ModelOutput, modelapi
from inspect_ai.scorer import SampleScore, Score, mean, metric, scorer
from inspect_ai.solver import generate

ROOT = Path(__file__).resolve().parent
PAPER = "https://arxiv.org/html/2606.09389v1"
DATA_COMMIT = "13861f99b2b268dbdffad7a95281e4ea27fe172d"
CODE_COMMIT = "a65cdbec3f6060906407cf391d2c69446e363d83"
JUDGE_MODEL = "qwen3.6-27b"
DATA_HASHES = {
    "falvzixun.json": "b4a729a9a9bb496b8fbd8791e1d3773110334423b013fdea74d191a75eced45a",
    "sifakaoshi.json": "7c173c63647ce1a6051dd6e5ebee18c4394d83b6d0b6767b17a5c9f6b6cc286a",
}
SPLITS = {"consultation": "falvzixun.json", "exam": "sifakaoshi.json"}
DIMENSIONS = {
    "legal_accuracy": "法律准确性",
    "reasoning_and_logic": "逻辑推理与分析",
    "task_compliance": "指令/题意遵从",
    "ethics_and_safety": "伦理安全",
    "completeness": "全面性",
    "clarity_and_structure": "表达与结构",
}


def _verified_bytes(path: Path, expected: str) -> bytes:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Pinned source was changed: {path}")
    return data


def verify_sources() -> dict:
    """Check file bytes against the downloaded manifest and original data hashes."""
    manifest = json.loads((ROOT / "upstream_manifest.json").read_text())
    if manifest["commit"] != CODE_COMMIT:
        raise ValueError("Unexpected upstream implementation revision")
    for name, info in manifest["files"].items():
        _verified_bytes(ROOT / "upstream" / name, info["sha256"])
    for name, sha in DATA_HASHES.items():
        _verified_bytes(ROOT / "data" / name, sha)
    return {"paper": PAPER, "data_commit": DATA_COMMIT,
            "code_commit": CODE_COMMIT, "data_sha256": DATA_HASHES,
            "prompt_sha256": manifest["files"]["code/prompt.txt"]["sha256"],
            "code_sha256": manifest["files"]["code/eval.py"]["sha256"]}


def load_upstream() -> ModuleType:
    """An isolated copy prevents candidate/judge endpoint configuration races."""
    manifest = json.loads((ROOT / "upstream_manifest.json").read_text())
    path = ROOT / "upstream/code/eval.py"
    _verified_bytes(path, manifest["files"]["code/eval.py"]["sha256"])
    _verified_bytes(ROOT / "upstream/code/prompt.txt", manifest["files"]["code/prompt.txt"]["sha256"])
    spec = importlib.util.spec_from_file_location("lexrubric_author_reference", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_dataset(split: str = "all") -> MemoryDataset:
    if split not in ["all", *SPLITS]:
        raise ValueError(f"Unknown split: {split}")
    samples = []
    for group, filename in SPLITS.items():
        if split not in ("all", group):
            continue
        rows = json.loads(_verified_bytes(ROOT / "data" / filename, DATA_HASHES[filename]))
        expected_rows, expected_criteria = (473, 10601) if group == "consultation" else (176, 1736)
        if len(rows) != expected_rows or sum(len(r["rubrics"]) for r in rows) != expected_criteria:
            raise ValueError("Original-paper corpus counts do not match")
        for row in rows:
            if sum(max(r["point"], 0) for r in row["rubrics"]) <= 0:
                raise ValueError(f"Undefined overall score for {group}:{row['ID']}")
            samples.append(Sample(id=f"{group}:{row['ID']}", input=row["question"],
                                  metadata={"original_id": row["ID"], "split": group,
                                            "answer": row["answer"], "rubrics": row["rubrics"]}))
    if len({s.id for s in samples}) != len(samples):
        raise ValueError("Duplicate sample IDs")
    return MemoryDataset(samples=samples, name=f"lexrubric_v1_{split}")


def score_rates(rubrics: list[dict], criterion_results: list[dict]) -> dict:
    """Original v1 Eqs. 1–3; percentages, signed penalties, no clipping.

    A dimension with no positive denominator is undefined (None). The authors
    explicitly clarify exclusion of those instances in v2 section 4.1.
    """
    if len(rubrics) != len(criterion_results):
        raise ValueError("Every criterion, including zero-point criteria, needs a result")
    points = sum(r["point"] for r, result in zip(rubrics, criterion_results, strict=True)
                 if result["criteria_met"])
    positive = sum(max(r["point"], 0) for r in rubrics)
    if positive <= 0:
        raise ValueError("Undefined overall positive denominator")
    dimensions = {}
    for dimension in dict.fromkeys(r["dimension"] for r in rubrics):
        pairs = [(r, res) for r, res in zip(rubrics, criterion_results, strict=True)
                 if r["dimension"] == dimension]
        denominator = sum(max(r["point"], 0) for r, _ in pairs)
        numerator = sum(r["point"] for r, res in pairs if res["criteria_met"])
        dimensions[dimension] = 100 * numerator / denominator if denominator else None
    return {"score_rate": 100 * points / positive, "dimensions": dimensions,
            "raw_points": points, "positive_points": positive}


def _dimension_metric(name, dimension):
    @metric(name=f"lexrubric_{name}")
    def factory():
        def compute(scores: list[SampleScore]) -> float:
            values = [s.score.metadata["dimension_rates"].get(dimension) for s in scores]
            values = [v for v in values if v is not None]
            return sum(values) / len(values) if values else float("nan")
        return compute
    return factory


DIMENSION_METRICS = [_dimension_metric(k, v) for k, v in DIMENSIONS.items()]


def endpoint_config(role: str) -> tuple[str, str]:
    """Use a complete upstream-compatible URL; never print credential values."""
    prefix = f"LEXRUBRIC_{role.upper()}"
    url = os.getenv(prefix + "_CHAT_URL") or os.getenv("LEXRUBRIC_CHAT_URL")
    key_var = os.getenv(prefix + "_API_KEY_ENV")
    key = os.getenv(key_var, "") if key_var else (
        os.getenv(prefix + "_API_KEY") or os.getenv("LEXRUBRIC_API_KEY")
    )
    if not url or not key:
        raise ValueError(f"Missing {role} endpoint or credential; configure {prefix}_CHAT_URL and {prefix}_API_KEY (or {prefix}_API_KEY_ENV)")
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(f"Invalid {role} endpoint: supply a full HTTP chat-completions URL without credentials/query/fragment")
    if key in ("api_key", "YOUR_DASHSCOPE_API_KEY", "replace-me"):
        raise ValueError(f"Placeholder {role} credential")
    return url, key


def configured_upstream(role: str):
    module = load_upstream()
    module.DASHSCOPE_CHAT_URL, module.DASHSCOPE_API_KEY = endpoint_config(role)
    provider = os.getenv(f"LEXRUBRIC_{role.upper()}_PROVIDER", "upstream")
    if provider != "upstream":
        from lexrubric_inspect.provider_transport import RequestsTransport
        module.requests = RequestsTransport(
            module.requests, provider=provider, role=role,
            allow_temperature_omission=os.getenv("LEXRUBRIC_ALLOW_TEMPERATURE_OMISSION") == "true",
            candidate_reasoning_effort=(os.getenv("LEXRUBRIC_CANDIDATE_REASONING_EFFORT")
                                        if role == "candidate" else None))
    return module


@modelapi(name="lexrubric_raw")
class LexRubricAPI(ModelAPI):
    """Use the authors' function with explicit, audited provider translations.

    The internal request retains paper settings. Only the configured transport
    may apply an approved temperature exception or a provider's token-field name.
    """

    def __init__(self, model_name, base_url=None, api_key=None, config=GenerateConfig(), **kwargs):
        if base_url or api_key or kwargs:
            raise ValueError("Configure endpoints with LEXRUBRIC_* environment variables; model/provider overrides are not supported")
        super().__init__(model_name=model_name, config=config)

    async def generate(self, input, tools, tool_choice, config):
        if tools or len(input) != 1 or input[0].role != "user" or not isinstance(input[0].content, str):
            raise ValueError("LexRubric requires exactly the original question as one user message, with no tools")
        if config.temperature != 0.6 or config.max_tokens != 16384:
            raise ValueError("Original paper requires candidate temperature=0.6, max_tokens=16384")
        permitted = {"temperature", "max_tokens", "max_connections", "max_retries", "max_tool_output"}
        extra = set(config.model_dump(exclude_none=True)) - permitted
        if extra or config.max_retries not in (None, 0):
            raise ValueError(f"Extra generation settings would change the protocol: {sorted(extra)}")
        ref = configured_upstream("candidate")
        model_cfg = {"name": self.model_name, "provider": "dashscope_chat",
                     "enable_thinking": None, "res_max_tokens": 16384, "temperature": 0.6}
        result = await asyncio.to_thread(ref.generate_model_response, model_cfg, input[0].content, None)
        output = ModelOutput.from_content(model=self.model_name, content=result["model_response"], stop_reason="unknown")
        output.metadata = {"upstream_candidate_error": result["error"],
                           "adapter": "unchanged upstream generate_model_response",
                           "request_parameters": model_cfg,
                           "usage_accounting": "upstream returns text only; tokens/cost unavailable"}
        if hasattr(ref.requests, "audit"):
            output.metadata["http_audit"] = ref.requests.audit
            output.metadata["transport_provider"] = ref.requests.provider
            output.metadata["temperature_omission_requested"] = ref.requests.allow_temperature_omission
            output.metadata["usage_accounting"] = "HTTP usage, when supplied, is retained in http_audit"
        return output


@scorer(metrics=[mean(), *(factory() for factory in DIMENSION_METRICS)])
def lexrubric_scorer(judge_model: str = JUDGE_MODEL):
    # Upstream scores one response at a time, with ten rubric workers per response.
    lock = asyncio.Lock()

    async def score(state, target):
        async with lock:
            ref = configured_upstream("judge")
            ref.JUDGE_MODEL = judge_model
            rubrics = state.metadata["rubrics"]
            response = state.output.completion
            checkpoint = None
            cache_dir = os.getenv("LEXRUBRIC_SCORE_CACHE")
            if cache_dir:
                cache_identity = {"sample_id": state.sample_id, "response": response,
                                  "rubrics": rubrics, "prompt": ref.load_prompt_template(),
                                  "judge_model": judge_model,
                                  "judge_endpoint": ref.DASHSCOPE_CHAT_URL,
                                  "transport": os.getenv("LEXRUBRIC_JUDGE_PROVIDER", "upstream"),
                                  "adapter": "v1"}
                digest = hashlib.sha256(json.dumps(cache_identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                checkpoint = Path(cache_dir) / f"{digest}.json"
                if checkpoint.exists():
                    return Score.model_validate_json(checkpoint.read_text())
            candidate_error = ref.is_api_error(response)
            if candidate_error:
                results = []
                filtered = ""
                # The author's runner assigns zero, not unscored, on candidate errors.
                rates = score_rates(rubrics, [{"criteria_met": False} for _ in rubrics])
            else:
                raw, results, filtered = await asyncio.to_thread(
                    ref.evaluate_response_with_rubrics, state.input_text, response,
                    rubrics, ref.load_prompt_template())
                rates = score_rates(rubrics, results)
                if raw != rates["raw_points"]:
                    raise AssertionError("Adapter raw score disagrees with upstream")
            metadata = {"raw_points": rates["raw_points"], "positive_points": rates["positive_points"],
                        "dimension_rates": rates["dimensions"], "criterion_evaluations": results,
                        "filtered_response": filtered, "candidate_error": candidate_error,
                        "grader_errors": sum(r["parse_error"] for r in results),
                        "judge_model": ref.JUDGE_MODEL,
                        "judge_config": {"temperature": ref.JUDGE_TEMPERATURE,
                                         "max_tokens": ref.JUDGE_MAX_TOKENS,
                                         "enable_thinking": ref.JUDGE_ENABLE_THINKING},
                        "split": state.metadata["split"]}
            if hasattr(ref.requests, "audit"):
                metadata["http_audit"] = ref.requests.audit
            scored = Score(value=rates["score_rate"], answer=filtered,
                         explanation="Original-paper score rate (%); original evaluator verdicts retained.",
                         metadata=metadata)
            if checkpoint:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                temporary = checkpoint.with_suffix(".tmp")
                temporary.write_text(scored.model_dump_json(indent=2))
                temporary.replace(checkpoint)
            return scored
    return score


@task
def lexrubric(split: str = "all", judge_model: str = JUDGE_MODEL) -> Task:
    provenance = verify_sources()
    return Task(dataset=load_dataset(split),
                solver=generate(tool_calls="none"), scorer=lexrubric_scorer(judge_model),
                config=GenerateConfig(temperature=0.6, max_tokens=16384, max_retries=0, max_connections=5),
                metadata={**provenance, "protocol": "original-paper-v1",
                          "judge_model": judge_model, "split": split,
                          "judge_changed_from_paper": judge_model != JUDGE_MODEL,
                          "dimension_zero_denominator": "undefined in v1; exclusion clarified by authors in v2",
                          "usage_accounting": "upstream text-only calls do not expose tokens or cost"},
                version=1)
