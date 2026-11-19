"""Offline checks for the paper equations and the released runner's behavior.

Run from the project root with:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1] / "lexrubric_inspect"

from lexrubric_inspect import lexrubric_port as port
from inspect_ai._util.registry import registry_info
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import SampleScore, Score, Target


def rubric(point: int, dimension: str = "法律准确性", criterion: str = "标准") -> dict:
    return {"criterion": criterion, "point": point, "dimension": dimension}


def judgments(rubrics: list[dict], met: list[bool]) -> list[dict]:
    return [
        {**item, "criterion_idx": index, "criteria_met": flag, "parse_error": False}
        for index, (item, flag) in enumerate(zip(rubrics, met, strict=True))
    ]


class OriginalCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = list(port.load_dataset())
        cls.rows = {
            split: json.loads((ROOT / "data" / filename).read_text(encoding="utf-8"))
            for split, filename in (
                ("consultation", "falvzixun.json"),
                ("exam", "sifakaoshi.json"),
            )
        }

    def test_original_june_release_and_checksums(self):
        manifest = json.loads((ROOT / "data_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["commit"], "13861f99b2b268dbdffad7a95281e4ea27fe172d")
        for filename, entry in manifest["files"].items():
            data = (ROOT / "data" / filename).read_bytes()
            with self.subTest(file=filename):
                self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"])
                self.assertEqual(len(data), entry["bytes"])

    def test_full_paper_corpus_counts(self):
        self.assertEqual(len(self.samples), 649)
        self.assertEqual(len(port.load_dataset("consultation")), 473)
        self.assertEqual(len(port.load_dataset("exam")), 176)
        self.assertEqual(sum(len(s.metadata["rubrics"]) for s in self.samples), 12337)
        for split, expected in (("consultation", 10601), ("exam", 1736)):
            with self.subTest(split=split):
                self.assertEqual(
                    sum(len(s.metadata["rubrics"]) for s in self.samples if s.metadata["split"] == split),
                    expected,
                )

    def test_every_original_question_reference_and_rubric_is_preserved(self):
        originals = {
            (split, row["ID"]): row
            for split, rows in self.rows.items()
            for row in rows
        }
        self.assertEqual(len({s.id for s in self.samples}), 649)
        for sample in self.samples:
            metadata = sample.metadata
            original = originals[(metadata["split"], metadata["original_id"])]
            with self.subTest(sample=sample.id):
                self.assertEqual(sample.id, f'{metadata["split"]}:{original["ID"]}')
                self.assertEqual(sample.input, original["question"])
                self.assertEqual(metadata["answer"], original["answer"])
                self.assertEqual(metadata["rubrics"], original["rubrics"])

    def test_both_original_zero_weight_criteria_remain_in_dataset(self):
        zeros = [
            (sample.metadata["original_id"], item)
            for sample in self.samples
            for item in sample.metadata["rubrics"]
            if item["point"] == 0
        ]
        self.assertEqual(len(zeros), 2)
        self.assertEqual({sample_id for sample_id, _ in zeros}, {42926})

    def test_invalid_split_is_rejected(self):
        with self.assertRaises(ValueError):
            port.load_dataset("unrecognized-split")


class PaperEquationTests(unittest.TestCase):
    def test_signed_weight_sum_uses_only_positive_denominator(self):
        items = [rubric(8), rubric(2), rubric(-4)]
        result = port.score_rates(items, judgments(items, [True, False, True]))
        self.assertEqual(result["raw_points"], 4)
        self.assertEqual(result["positive_points"], 10)
        self.assertEqual(result["score_rate"], 40.0)
        self.assertEqual(result["dimensions"]["法律准确性"], 40.0)

    def test_negative_score_is_not_clamped(self):
        items = [rubric(2), rubric(-6)]
        result = port.score_rates(items, judgments(items, [True, True]))
        self.assertEqual(result["score_rate"], -200.0)
        self.assertEqual(result["dimensions"]["法律准确性"], -200.0)

    def test_all_positive_met_and_negative_untriggered_is_full_credit(self):
        items = [rubric(3), rubric(7), rubric(-10)]
        result = port.score_rates(items, judgments(items, [True, True, False]))
        self.assertEqual(result["score_rate"], 100.0)

    def test_unmet_positive_and_untriggered_negative_both_contribute_zero(self):
        items = [rubric(4), rubric(-10)]
        self.assertEqual(port.score_rates(items, judgments(items, [False, False]))["score_rate"], 0.0)

    def test_zero_weight_criterion_is_scored_but_does_not_change_rate(self):
        items = [rubric(5), rubric(0), rubric(-2)]
        first = port.score_rates(items, judgments(items, [True, False, True]))
        second = port.score_rates(items, judgments(items, [True, True, True]))
        self.assertEqual(first, second)
        self.assertEqual(first["score_rate"], 60.0)

    def test_dimension_denominators_are_computed_per_question(self):
        items = [rubric(8), rubric(-2), rubric(2, "全面性"), rubric(-7, "伦理安全")]
        result = port.score_rates(items, judgments(items, [True, True, False, True]))
        self.assertEqual(result["score_rate"], -10.0)
        self.assertEqual(result["dimensions"]["法律准确性"], 75.0)
        self.assertEqual(result["dimensions"]["全面性"], 0.0)
        self.assertIsNone(result["dimensions"]["伦理安全"])

    def test_missing_criterion_result_is_rejected_instead_of_silently_dropped(self):
        with self.assertRaises(ValueError):
            port.score_rates([rubric(5), rubric(0)], [{"criteria_met": True}])

    def test_undefined_overall_denominator_is_rejected(self):
        items = [rubric(0), rubric(-3)]
        with self.assertRaises(ValueError):
            port.score_rates(items, judgments(items, [True, True]))

    def test_registered_overall_metric_is_mean_of_question_rates_not_pooled_points(self):
        items_one = [rubric(2)]
        items_two = [rubric(100)]
        rates = [
            port.score_rates(items_one, judgments(items_one, [True])),
            port.score_rates(items_two, judgments(items_two, [False])),
        ]
        scores = [SampleScore(score=Score(value=r["score_rate"])) for r in rates]
        metrics = registry_info(port.lexrubric_scorer()).metadata["metrics"]
        overall = next(m for m in metrics if registry_info(m).name.rsplit("/", 1)[-1] == "mean")
        self.assertEqual(overall(scores), 50.0)
        self.assertNotAlmostEqual(overall(scores), 100 * 2 / 102)

    def test_registered_dimension_metric_excludes_undefined_but_includes_zero(self):
        metrics = registry_info(port.lexrubric_scorer()).metadata["metrics"]
        accuracy = next(m for m in metrics if registry_info(m).name.rsplit("/", 1)[-1] == "lexrubric_legal_accuracy")
        scores = [
            SampleScore(score=Score(value=0, metadata={"dimension_rates": {"法律准确性": value}}))
            for value in (100.0, None, 0.0)
        ]
        self.assertEqual(accuracy(scores), 50.0)
        self.assertTrue(math.isnan(accuracy([scores[1]])))


class ReleasedRunnerTests(unittest.TestCase):
    def setUp(self):
        self.upstream = port.load_upstream()
        self.network = patch.object(
            self.upstream.requests, "post", side_effect=AssertionError("Offline test attempted HTTP")
        )
        self.network.start()
        self.addCleanup(self.network.stop)

    def test_loaded_module_is_isolated_between_runs(self):
        other = port.load_upstream()
        self.assertIsNot(self.upstream, other)
        self.upstream.JUDGE_MODEL = "changed-locally"
        self.assertEqual(other.JUDGE_MODEL, "qwen3.6-27b")

    def test_candidate_gets_only_original_question_and_paper_generation_settings(self):
        with patch.object(self.upstream, "call_dashscope_chat", return_value="答复") as call:
            result = self.upstream.generate_model_response(self.upstream.MODELS[0], "原始法律问题", 1)
        self.assertFalse(result["error"])
        self.assertEqual(call.call_args.kwargs["messages"], [{"role": "user", "content": "原始法律问题"}])
        self.assertEqual(call.call_args.kwargs["temperature"], 0.6)
        self.assertEqual(call.call_args.kwargs["max_tokens"], 16384)
        self.assertIsNone(call.call_args.kwargs["enable_thinking"])

    def test_judge_uses_unchanged_chinese_prompt_and_original_settings(self):
        template = (ROOT / "upstream" / "code" / "prompt.txt").read_text(encoding="utf-8")
        self.assertEqual(self.upstream.load_prompt_template(), template)
        conversation = "user: 原始问题\nassistant: 模型回答"
        for point, direction in ((5, "不是"), (-5, "是"), (0, "不是")):
            with self.subTest(point=point):
                item = rubric(point, criterion="单条中文标准")
                with patch.object(self.upstream, "call_dashscope_chat", return_value='{"criteria_met": true, "explanation": "说明"}') as call:
                    result = self.upstream.evaluate_single_criterion(template, conversation, item, 7)
                expected = template.replace("{conversation}", conversation).replace("{rubric_item}", item["criterion"]).replace("{direction}", direction)
                kwargs = call.call_args.kwargs
                self.assertEqual(kwargs["messages"], [{"role": "user", "content": expected}])
                self.assertIn("{{", expected)
                self.assertEqual(kwargs["model"], "qwen3.6-27b")
                self.assertEqual(kwargs["temperature"], 0.0)
                self.assertEqual(kwargs["max_tokens"], 4096)
                self.assertIs(kwargs["enable_thinking"], False)
                self.assertTrue(result["criteria_met"])
                self.assertEqual(result["direction"], direction)

    def test_response_filter_removes_only_first_think_prefix(self):
        extract = self.upstream.extract_response_after_think
        self.assertEqual(extract("<think>hidden</think>  答案  "), "答案")
        self.assertEqual(extract("<think>a</think>b</think>c"), "b</think>c")
        self.assertEqual(extract("  untagged answer  "), "  untagged answer  ")

    def test_raw_and_markdown_json_are_accepted_without_schema_rewrite(self):
        expected = {"criteria_met": True, "explanation": "说明"}
        body = json.dumps(expected, ensure_ascii=False)
        for response in (body, f"```json\n{body}\n```", f"```\n{body}\n```"):
            with self.subTest(response=response):
                self.assertEqual(self.upstream.parse_json_response(response), expected)
        self.assertIsNone(self.upstream.parse_json_response("unfenced prose " + body))

    def test_original_boolean_coercion_is_retained(self):
        for value, expected in ((True, True), (False, False), ("TRUE", True), (" true ", False), ("yes", False), (1, True), (0, False), (None, False)):
            with self.subTest(value=value):
                self.assertIs(self.upstream.to_bool(value), expected)

    def test_parse_retry_reuses_same_prompt_exactly_three_times(self):
        with patch.object(self.upstream, "call_dashscope_chat", side_effect=["invalid", "{}", '{"criteria_met": "TRUE"}']) as call:
            result = self.upstream.evaluate_single_criterion("{conversation}\n{rubric_item}\n{direction}", "conversation", rubric(5), 0)
        self.assertEqual(call.call_count, 3)
        self.assertEqual(call.call_args_list[0], call.call_args_list[1])
        self.assertEqual(call.call_args_list[1], call.call_args_list[2])
        self.assertTrue(result["criteria_met"])
        self.assertFalse(result["parse_error"])
        self.assertEqual(result["explanation"], "")

    def test_exhausted_parse_retries_return_original_false_fallback(self):
        with patch.object(self.upstream, "call_dashscope_chat", return_value="invalid") as call:
            result = self.upstream.evaluate_single_criterion("{rubric_item}", "conversation", rubric(-5), 0)
        self.assertEqual(call.call_count, 3)
        self.assertFalse(result["criteria_met"])
        self.assertTrue(result["parse_error"])

    def test_api_error_does_not_start_extra_parse_retries(self):
        with patch.object(self.upstream, "call_dashscope_chat", return_value="[API timeout]") as call:
            result = self.upstream.evaluate_single_criterion("{rubric_item}", "conversation", rubric(-5), 0)
        self.assertEqual(call.call_count, 1)
        self.assertFalse(result["criteria_met"])
        self.assertTrue(result["parse_error"])

    def test_full_response_judging_uses_final_answer_one_call_per_criterion(self):
        items = [rubric(8), rubric(-3), rubric(0)]
        with patch.object(self.upstream, "call_dashscope_chat", return_value='{"criteria_met":true,"explanation":"yes"}') as call:
            raw, results, answer = self.upstream.evaluate_response_with_rubrics(
                "原始问题", "<think>secret</think>最终回答", items, "{conversation}\n{rubric_item}\n{direction}"
            )
        self.assertEqual(call.call_count, 3)
        self.assertEqual(raw, 5)
        self.assertEqual(answer, "最终回答")
        self.assertEqual([r["criterion_idx"] for r in results], [0, 1, 2])
        for call_record in call.call_args_list:
            prompt = call_record.kwargs["messages"][0]["content"]
            self.assertTrue(prompt.startswith("user: 原始问题\nassistant: 最终回答\n"))
            self.assertNotIn("secret", prompt)


class InspectScorerIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def state(self, items, response):
        return SimpleNamespace(
            metadata={"rubrics": items, "split": "consultation"},
            input_text="原始问题",
            output=ModelOutput.from_content(model="test", content=response),
        )

    async def test_actual_inspect_scorer_preserves_author_verdicts_and_metadata(self):
        ref = port.load_upstream()
        items = [rubric(8), rubric(-3), rubric(0)]
        with patch.object(port, "configured_upstream", return_value=ref), patch.object(
            ref, "call_dashscope_chat", return_value='{"criteria_met":true,"explanation":"说明"}'
        ) as call:
            score = await port.lexrubric_scorer()(self.state(items, "<think>hidden</think>最终回答"), Target(""))
        self.assertEqual(call.call_count, 3)
        self.assertEqual(score.value, 62.5)
        self.assertEqual(score.answer, "最终回答")
        self.assertEqual(score.metadata["raw_points"], 5)
        self.assertEqual(score.metadata["positive_points"], 8)
        self.assertEqual(len(score.metadata["criterion_evaluations"]), 3)
        self.assertEqual(score.metadata["grader_errors"], 0)
        self.assertFalse(score.metadata["candidate_error"])
        self.assertEqual(score.metadata["judge_model"], "qwen3.6-27b")

    async def test_actual_inspect_scorer_retains_judge_parse_failure_as_false(self):
        ref = port.load_upstream()
        items = [rubric(8), rubric(-3)]
        with patch.object(port, "configured_upstream", return_value=ref), patch.object(
            ref, "call_dashscope_chat", return_value="invalid"
        ) as call:
            score = await port.lexrubric_scorer()(self.state(items, "回答"), Target(""))
        self.assertEqual(call.call_count, 6)
        self.assertEqual(score.value, 0)
        self.assertEqual(score.metadata["grader_errors"], 2)
        self.assertTrue(all(not r["criteria_met"] for r in score.metadata["criterion_evaluations"]))

    async def test_candidate_failure_is_zero_without_judging(self):
        ref = port.load_upstream()
        items = [rubric(8), rubric(-3)]
        with patch.object(port, "configured_upstream", return_value=ref), patch.object(
            ref, "evaluate_response_with_rubrics", side_effect=AssertionError("Candidate errors must skip judging")
        ) as call:
            score = await port.lexrubric_scorer()(self.state(items, "[API timeout]"), Target(""))
        call.assert_not_called()
        self.assertEqual(score.value, 0)
        self.assertTrue(score.metadata["candidate_error"])
        self.assertEqual(score.metadata["criterion_evaluations"], [])


if __name__ == "__main__":
    unittest.main()
