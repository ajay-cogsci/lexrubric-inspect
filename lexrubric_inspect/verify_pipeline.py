"""Offline integration verification; fixture outputs are NOT benchmark results.

Run with lexrubric-verify or python -m lexrubric_inspect.verify_pipeline.
The real Inspect task, provider, and unchanged upstream generation/grading
functions run. Only requests.post is replaced by a deterministic transport;
socket connections and other requests transports are blocked throughout eval.
"""
from __future__ import annotations

import copy
import argparse
import hashlib
import json
import math
import os
import socket
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from inspect_ai import eval as inspect_eval, task_with

from . import lexrubric_port as port


ROOT = Path(__file__).resolve().parent
LABEL = "SYNTHETIC OFFLINE VERIFICATION — NOT BENCHMARK PERFORMANCE"
MODEL = "fixture-candidate"
SAMPLE_IDS = ["consultation:42926", "exam:142"]
CANDIDATE_URL = "http://127.0.0.1:1/mock-candidate/chat/completions"
JUDGE_URL = "http://127.0.0.1:1/mock-judge/chat/completions"
SYNTHETIC_KEY = "offline-verification-synthetic-no-real-credential"


def digest(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def check_corpus(dataset):
    """Compare every field against original-paper files, not adapter helpers."""
    expected = {}
    counts = {}
    for split, filename in port.SPLITS.items():
        rows = json.loads((ROOT / "data" / filename).read_text(encoding="utf-8"))
        counts[split] = {"samples": len(rows),
                         "criteria": sum(len(row["rubrics"]) for row in rows)}
        for row in rows:
            expected[f"{split}:{row['ID']}"] = {
                "input": row["question"],
                "metadata": {"original_id": row["ID"], "split": split,
                             "answer": row["answer"], "rubrics": row["rubrics"]},
            }
    assert len(dataset) == len(expected) == 649
    assert counts == {"consultation": {"samples": 473, "criteria": 10601},
                      "exam": {"samples": 176, "criteria": 1736}}
    observed = {}
    for sample in dataset:
        assert sample.id not in observed, f"Duplicate ID: {sample.id}"
        observed[sample.id] = {"input": sample.input, "metadata": sample.metadata}
        assert observed[sample.id] == expected[sample.id], sample.id
        assert sample.target == "", f"Reference answer leaked to target: {sample.id}"
    assert observed == expected
    return expected, counts


def fixture_verdict(point, index):
    # Exercise a real negative penalty and both zero-point verdicts.
    if point < 0:
        return True
    if point == 0:
        return index % 2 == 0
    return index % 3 != 1


def expected_rates(rubrics):
    """Independently calculate fixture arithmetic without the port scorer."""
    verdicts = [fixture_verdict(r["point"], i) for i, r in enumerate(rubrics)]
    numerator = sum(r["point"] for r, met in zip(rubrics, verdicts) if met)
    denominator = sum(r["point"] for r in rubrics if r["point"] > 0)
    dimensions = {}
    for name in {r["dimension"] for r in rubrics}:
        indices = [i for i, r in enumerate(rubrics) if r["dimension"] == name]
        dim_positive = sum(max(rubrics[i]["point"], 0) for i in indices)
        dim_points = sum(rubrics[i]["point"] for i in indices if verdicts[i])
        dimensions[name] = 100 * dim_points / dim_positive if dim_positive else None
    return numerator, denominator, 100 * numerator / denominator, dimensions


class FixtureResponse:
    status_code = 200

    def __init__(self, content):
        self.body = {"choices": [{"message": {"role": "assistant", "content": content}}]}
        self.text = json.dumps(self.body, ensure_ascii=False)

    def json(self):
        return copy.deepcopy(self.body)


class MockTransport:
    def __init__(self, records):
        self.lock = threading.Lock()
        self.captured = []
        self.network_attempts = []
        self.expected = {}
        self.fixture = {}
        template = (ROOT / "upstream/code/prompt.txt").read_text(encoding="utf-8")
        for sample_id in SAMPLE_IDS:
            row = records[sample_id]
            question = row["input"]
            final = f"[SYNTHETIC OFFLINE FIXTURE: {sample_id}; not a legal answer.]"
            candidate = f"  <think>Synthetic verification reasoning only.</think>\n{final}  "
            self.fixture[sample_id] = {"candidate_raw": candidate,
                                       "candidate_extracted": candidate.strip(),
                                       "filtered": final}
            candidate_payload = {"model": MODEL,
                                 "messages": [{"role": "user", "content": question}],
                                 "max_tokens": 16384, "temperature": 0.6}
            self.add(CANDIDATE_URL, candidate_payload, candidate,
                     {"role": "candidate", "sample_id": sample_id})
            conversation = f"user: {question}\nassistant: {final}"
            for i, criterion in enumerate(row["metadata"]["rubrics"]):
                point = criterion["point"]
                direction = "不是" if point >= 0 else "是"
                prompt = (template.replace("{conversation}", conversation)
                          .replace("{rubric_item}", criterion["criterion"])
                          .replace("{direction}", direction))
                met = fixture_verdict(point, i)
                # String booleans also exercise the author's to_bool coercion.
                verdict = {"explanation": f"Synthetic transport fixture {sample_id}/{i}; no assessment performed.",
                           "criteria_met": str(met).upper() if i % 2 else met}
                text = json.dumps(verdict, ensure_ascii=False)
                if i % 2:
                    text = f"```json\n{text}\n```"
                payload = {"model": "qwen3.6-27b",
                           "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": 4096, "temperature": 0.0,
                           "enable_thinking": False}
                info = {"role": "judge", "sample_id": sample_id,
                        "criterion_index": i,
                        "weight_sign": "positive" if point > 0 else "negative" if point < 0 else "zero",
                        "direction": direction}
                self.add(JUDGE_URL, payload, text, info)

    def add(self, url, payload, response, info):
        fingerprint = digest({"url": url, "payload": payload})
        assert fingerprint not in self.expected, "Ambiguous fixture payload"
        self.expected[fingerprint] = {"url": url, "payload": payload,
                                      "response": response, "info": info}

    def post(self, url, **kwargs):
        assert set(kwargs) == {"json", "headers", "timeout"}, "Unexpected HTTP arguments"
        assert kwargs["timeout"] == 3600
        assert kwargs["headers"] == {"Content-Type": "application/json",
                                    "Authorization": f"Bearer {SYNTHETIC_KEY}"}
        payload = kwargs["json"]
        fingerprint = digest({"url": url, "payload": payload})
        assert fingerprint in self.expected, "HTTP payload differs from the original protocol fixture"
        expected = self.expected[fingerprint]
        assert url == expected["url"] and payload == expected["payload"]
        # Store compact hashes only, never authorization headers or credentials.
        with self.lock:
            self.captured.append({**expected["info"], "payload_sha256": digest(payload),
                                  "messages_sha256": digest(payload["messages"]),
                                  "request_sha256": fingerprint,
                                  "model": payload["model"], "payload_keys": sorted(payload)})
        return FixtureResponse(expected["response"])

    def blocked_network(self, *args, **kwargs):
        with self.lock:
            self.network_attempts.append("Blocked unmocked network call")
        raise RuntimeError("Offline verification prohibits all real network connections")


def verify_logs(logs, records, transport):
    assert len(logs) == 1, "Expected exactly one Inspect eval log"
    assert logs[0].status == "success", (
        f"Inspect eval failed: {logs[0].error.message if logs[0].error else logs[0].status}"
    )
    log = logs[0]
    assert log.eval.metadata["verification_only"] is True
    assert set(s.id for s in log.samples) == set(SAMPLE_IDS)
    checked_criteria = 0
    independent_rates = []
    for sample in log.samples:
        assert sample.error is None
        row = records[sample.id]
        fixture = transport.fixture[sample.id]
        assert sample.input == row["input"] and sample.metadata == row["metadata"]
        assert [(m.role, m.content) for m in sample.messages] == [
            ("user", row["input"]), ("assistant", fixture["candidate_extracted"])]
        assert sample.output.completion == fixture["candidate_extracted"]
        assert len(sample.scores) == 1
        score = next(iter(sample.scores.values()))
        info = score.metadata
        rubrics = row["metadata"]["rubrics"]
        judgments = info["criterion_evaluations"]
        assert len(judgments) == len(rubrics)
        assert info["candidate_error"] is False and info["grader_errors"] == 0
        assert info["judge_model"] == "qwen3.6-27b"
        assert info["judge_config"] == {"temperature": 0.0, "max_tokens": 4096,
                                         "enable_thinking": False}
        assert info["filtered_response"] == score.answer == fixture["filtered"]
        for i, (criterion, judgment) in enumerate(zip(rubrics, judgments, strict=True)):
            assert judgment["criterion_idx"] == i
            assert all(judgment[k] == criterion[k] for k in ("criterion", "point", "dimension"))
            assert judgment["criteria_met"] is fixture_verdict(criterion["point"], i)
            assert judgment["direction"] == ("是" if criterion["point"] < 0 else "不是")
            assert judgment["parse_error"] is False and judgment["raw_response"]
        raw, positive, rate, dimensions = expected_rates(rubrics)
        assert info["raw_points"] == raw and info["positive_points"] == positive
        assert math.isclose(score.value, rate, rel_tol=1e-12)
        assert info["dimension_rates"] == dimensions
        independent_rates.append((rate, dimensions))
        checked_criteria += len(judgments)
    assert log.results and len(log.results.scores) == 1
    expected_metrics = {"mean": sum(rate for rate, _ in independent_rates) / len(independent_rates)}
    for name, dimension in port.DIMENSIONS.items():
        values = [dimensions.get(dimension) for _, dimensions in independent_rates]
        values = [value for value in values if value is not None]
        expected_metrics[f"lexrubric_{name}"] = sum(values) / len(values) if values else float("nan")
    actual_metrics = {name.rsplit("/", 1)[-1]: value.value
                      for name, value in log.results.scores[0].metrics.items()}
    assert set(actual_metrics) == set(expected_metrics)
    assert all((math.isnan(actual_metrics[name]) if math.isnan(value)
                else math.isclose(actual_metrics[name], value, rel_tol=1e-12))
               for name, value in expected_metrics.items())
    actual = Counter(c["request_sha256"] for c in transport.captured)
    assert actual == Counter({key: 1 for key in transport.expected})
    assert not transport.network_attempts
    signs = Counter(c["weight_sign"] for c in transport.captured if c["role"] == "judge")
    assert all(signs[sign] > 0 for sign in ("positive", "negative", "zero"))
    return log, checked_criteria, dict(signs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("."),
                        help="Writable directory for synthetic verification artifacts")
    output_root = parser.parse_args().output_dir
    base_task = port.lexrubric()
    records, counts = check_corpus(base_task.dataset)
    transport = MockTransport(records)
    task = task_with(base_task, name="lexrubric_offline_synthetic_verification",
                     description=LABEL,
                     metadata={**base_task.metadata, "verification_only": True,
                               "verification_label": LABEL,
                               "transport": "mocked requests.post; all real sockets blocked"},
                     tags=["synthetic", "offline-verification", "not-benchmark-performance"])
    env = {"LEXRUBRIC_CANDIDATE_CHAT_URL": CANDIDATE_URL,
           "LEXRUBRIC_JUDGE_CHAT_URL": JUDGE_URL,
           "LEXRUBRIC_CANDIDATE_API_KEY": SYNTHETIC_KEY,
           "LEXRUBRIC_JUDGE_API_KEY": SYNTHETIC_KEY,
           "LEXRUBRIC_CANDIDATE_API_KEY_ENV": "",
           "LEXRUBRIC_JUDGE_API_KEY_ENV": "",
           "LEXRUBRIC_CANDIDATE_PROVIDER": "upstream",
           "LEXRUBRIC_JUDGE_PROVIDER": "upstream",
           "LEXRUBRIC_ALLOW_TEMPERATURE_OMISSION": "false",
           "LEXRUBRIC_SCORE_CACHE": ""}
    print(LABEL)
    with (patch.dict(os.environ, env),
          patch("inspect_ai._util.appdirs.user_data_path",
                return_value=output_root / "logs/offline-verification/.inspect-state/data"),
          patch("inspect_ai._util.appdirs.user_cache_path",
                return_value=output_root / "logs/offline-verification/.inspect-state/cache"),
          patch("requests.post", side_effect=transport.post),
          patch("requests.sessions.Session.request", side_effect=transport.blocked_network),
          patch.object(socket.socket, "connect", side_effect=transport.blocked_network),
          patch.object(socket.socket, "connect_ex", side_effect=transport.blocked_network)):
        logs = inspect_eval(task, model=f"lexrubric_raw/{MODEL}",
                            sample_id=SAMPLE_IDS, display="none", ctl_server=False,
                            log_dir=str(output_root / "logs/offline-verification"),
                            log_samples=True, log_model_api=True,
                            metadata={"verification_only": True, "verification_label": LABEL})
    log, checked_criteria, signs = verify_logs(logs, records, transport)
    report = {"status": "passed", "purpose": LABEL,
              "benchmark_performance_claim": False,
              "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "transport": "mocked requests.post; original upstream HTTP functions executed",
              "real_network_connections": 0, "paid_api_calls": 0,
              "source_verification": port.verify_sources(),
              "corpus": {"all_samples_checked": 649, "all_criteria_checked": 12337,
                         "splits": counts, "exact_input_and_metadata_match": True},
              "inspect_pipeline": {"sample_ids": SAMPLE_IDS,
                                   "candidate_requests": 2, "judge_requests": checked_criteria,
                                   "criterion_weight_sign_counts": signs,
                                   "exact_payloads_verified": True,
                                   "upstream_parsing_and_think_removal_verified": True,
                                   "signed_score_and_dimension_arithmetic_verified": True,
                                   "overall_and_six_dimension_macro_averages_verified": True,
                                   "criterion_order_and_raw_judgments_verified": True,
                                   "eval_log": str(Path(log.location).resolve().relative_to(output_root.resolve()))},
              "captured_requests": sorted(transport.captured,
                                          key=lambda item: (item["sample_id"], item["role"],
                                                            item.get("criterion_index", -1)))}
    destination = output_root / "results/offline_verification.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(f"Passed: all 649 records; 2 synthetic pipeline samples; {checked_criteria} mocked judge calls.")
    print(f"Verification report (no benchmark scores): {destination}")


if __name__ == "__main__":
    main()
