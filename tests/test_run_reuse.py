"""Offline generation-import and failure-status tests; no model/API calls."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lexrubric_inspect import run


class GenerationReuseTests(unittest.TestCase):
    def setUp(self):
        self.config = {"candidate_model": "gpt-6-astra", "candidate_provider": "openai",
                       "candidate_chat_url": "https://candidate.invalid/v1/chat/completions",
                       "candidate_api_key_env": "SYNTHETIC_CANDIDATE_KEY",
                       "allow_temperature_omission": True,
                       "authorized_exception": "Synthetic fixture authorization record",
                       "judge_model": "fixture-new-judge", "judge_provider": "upstream",
                       "judge_chat_url": "https://judge.invalid/v1/chat/completions",
                       "judge_api_key_env": "SYNTHETIC_JUDGE_KEY"}
        self.provenance = {"data_commit": "pinned-data", "code_commit": "pinned-code",
                           "prompt_sha256": "pinned-prompt"}
        self.datasets = {}
        for split in ("consultation", "exam"):
            self.datasets[split] = [NS(id=f"{split}:{i}", input=f"Original question {split}/{i}", target="",
                metadata={"original_id": i, "split": split, "answer": "Reference only",
                          "rubrics": [{"criterion": "desired", "point": 5, "dimension": "法律准确性"},
                                      {"criterion": "undesired", "point": -2, "dimension": "法律准确性"}]})
                for i in (1, 2)]
        self.load_patch = patch.object(run, "load_dataset", side_effect=self.dataset)
        self.load_patch.start()
        self.addCleanup(self.load_patch.stop)

    def dataset(self, split="all"):
        return deepcopy(sum(self.datasets.values(), []) if split == "all" else self.datasets[split])

    def generation(self, split):
        original = self.dataset(split)[0]
        old_config = {**self.config, "judge_model": "fixture-old-judge"}
        output = NS(completion="Preserved synthetic answer", metadata={
            "upstream_candidate_error": False,
            "request_parameters": {"name": "gpt-6-astra", "provider": "dashscope_chat",
                                   "enable_thinking": None, "res_max_tokens": 16384, "temperature": .6}})
        sample = NS(**vars(original), error=None, scores=None, output=output,
                    messages=[NS(role="user", content=original.input),
                              NS(role="assistant", content=output.completion)])
        return NS(status="success", eval=NS(model="lexrubric_raw/gpt-6-astra",
            task_args={"split": split, "judge_model": "fixture-old-judge"},
            task_args_passed={"split": split, "judge_model": "fixture-old-judge"},
            metadata={**self.provenance, "protocol": "original-paper-v1", "split": split,
                      "run_configuration": old_config, "judge_model": "fixture-old-judge"},
            config=NS(epochs=1), dataset=NS(name=f"lexrubric_v1_{split}", samples=2,
                shuffled=False, sample_ids=[original.id])), samples=[sample])

    def test_accepts_new_judge_but_rejects_generation_changes(self):
        good = self.generation("consultation")
        info = run.validate_generation(good, "consultation", self.config, self.provenance, 1)
        self.assertEqual(info["source_judge_model"], "fixture-old-judge")
        for field, value in (("candidate_model", "different"), ("candidate_provider", "different"),
                             ("candidate_chat_url", "https://different.invalid"),
                             ("allow_temperature_omission", False), ("authorized_exception", "changed")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                run.validate_generation(good, "consultation", {**self.config, field: value}, self.provenance, 1)
        with self.assertRaisesRegex(ValueError, "selected IDs"):
            run.validate_generation(good, "consultation", self.config, self.provenance, None)
        mutations = [lambda log: log.eval.metadata.update(code_commit="changed"),
                     lambda log: log.eval.metadata.update(verification_only=True),
                     lambda log: log.samples[0].metadata["rubrics"][0].update(point=100),
                     lambda log: log.samples[0].messages.insert(0, NS(role="system", content="changed")),
                     lambda log: log.samples[0].output.metadata.update(upstream_candidate_error=True),
                     lambda log: setattr(log.samples[0], "scores", {"old": 1}),
                     lambda log: log.eval.dataset.sample_ids.append("consultation:other")]
        for mutation in mutations:
            log = deepcopy(good)
            mutation(log)
            with self.assertRaises(ValueError):
                run.validate_generation(log, "consultation", self.config, self.provenance, 1)

    def test_import_copy_preserves_bytes_and_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            source.mkdir()
            for split in ("consultation", "exam"):
                (source / f"{split}-generation.eval").write_bytes(f"fixture-{split}".encode())
            with patch.object(run, "read_eval_log", side_effect=lambda path: self.generation(Path(path).name.split("-")[0])):
                imported = run.prepare_generation_import(source, self.config, self.provenance, 1)
            info = imported["files"]["exam"]
            destination = Path(temp) / "new-generation.eval"
            run.copy_generation_import(info, destination)
            self.assertEqual(destination.read_bytes(), b"fixture-exam")
            Path(info["source_path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed after"):
                run.copy_generation_import(info, destination)

    def exercise_main(self, *, grader_error=False, denied=False):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            source = root / "logs/old-run"
            source.mkdir(parents=True)
            stored = {}
            for split in ("consultation", "exam"):
                path = source / f"{split}-generation.eval"
                path.write_bytes(f"immutable-{split}".encode())
                stored[path.name] = self.generation(split)
            source_cache = root / "results/old-run/score-cache"
            source_cache.mkdir(parents=True)
            (source_cache / "do-not-import.json").write_text("old judge cache", encoding="utf-8")
            config_path = root / "config.json"
            config_path.write_text(json.dumps(self.config), encoding="utf-8")
            original_config = config_path.read_bytes()

            def score(log, **kwargs):
                expected_args = {"split": log.eval.metadata["split"], "judge_model": "fixture-new-judge"}
                self.assertEqual(log.eval.task_args, expected_args)
                self.assertEqual(log.eval.task_args_passed, expected_args)
                self.assertEqual(log.eval.metadata["judge_model"], "fixture-new-judge")
                self.assertEqual(log.eval.metadata["run_configuration"], self.config)
                self.assertEqual(log.eval.metadata["generation_import"]["source_judge_model"], "fixture-old-judge")
                for sample in log.samples:
                    sample.scores = {"fixture": NS(value=0 if grader_error else 60,
                        metadata={"raw_points": 0 if grader_error else 3,
                                  "dimension_rates": {"法律准确性": 0 if grader_error else 60},
                                  "candidate_error": False, "grader_errors": int(grader_error),
                                  "criterion_evaluations": [{}, {}]})}
                return log

            def write(log, path):
                stored[Path(path).name] = deepcopy(log)
                Path(path).write_bytes(b"synthetic-scored-log")

            stack.enter_context(patch.dict("os.environ", {}, clear=False))
            stack.enter_context(patch.object(run, "ROOT", root))
            stack.enter_context(patch.object(run, "configure"))
            stack.enter_context(patch.object(run, "verify_sources", return_value=self.provenance))
            stack.enter_context(patch.object(run, "read_eval_log", side_effect=lambda path: deepcopy(stored[Path(path).name])))
            stack.enter_context(patch.object(run, "write_eval_log", side_effect=write))
            generate = stack.enter_context(patch.object(run, "inspect_eval", side_effect=AssertionError("Must not regenerate")))
            grading = stack.enter_context(patch.object(run, "inspect_score", side_effect=score))
            access = stack.enter_context(patch("lexrubric_inspect.preflight.check_provider_access",
                side_effect=RuntimeError("fixture auth denied") if denied else None,
                return_value=[{"status": "authorized fixture"}]))
            stack.enter_context(patch.object(sys, "argv", ["run.py", "--config", str(config_path),
                "--run-name", "new-run", "--limit", "1", "--generation-from", str(source)]))
            stack.enter_context(redirect_stdout(io.StringIO()))
            if denied:
                with self.assertRaisesRegex(RuntimeError, "auth denied"):
                    run.main()
                self.assertEqual(grading.call_count, 0)
            else:
                run.main()
                self.assertEqual(grading.call_count, 2)
                for split in ("consultation", "exam"):
                    self.assertEqual((root / f"results/new-run/working/{split}-generation.eval").read_bytes(),
                                     (source / f"{split}-generation.eval").read_bytes())
                    source_eval = stored[f"{split}-generation.eval"].eval
                    self.assertEqual(source_eval.task_args["judge_model"], "fixture-old-judge")
                    self.assertEqual(source_eval.task_args_passed["judge_model"], "fixture-old-judge")
                self.assertFalse((root / "logs/new-run").exists())
                self.assertFalse((root / "results/new-run/score-cache/do-not-import.json").exists())
                summary = json.loads((root / "results/new-run/summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["benchmark_results_valid"], not grader_error)
                self.assertEqual(summary["overall_score_rate_percent"], 0 if grader_error else 60)
                manifest = json.loads((root / "results/new-run/run_manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["generation_import"]["source_log_dir"], str(source.resolve()))
                access.reset_mock()
                grading.reset_mock()
                run.main()  # Matching completed resume needs no API access or regrading.
                access.assert_not_called()
                grading.assert_not_called()
            generate.assert_not_called()
            self.assertEqual(config_path.read_bytes(), original_config)
            self.assertEqual((source_cache / "do-not-import.json").read_text(encoding="utf-8"), "old judge cache")
            status = json.loads((root / "results/new-run/status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["phase"], "failed" if denied else "complete_with_errors" if grader_error else "complete")

    def test_new_judge_reuses_generation_and_completed_resume_is_offline(self):
        self.exercise_main()

    def test_api_errors_preserve_upstream_scores_but_mark_results_invalid(self):
        self.exercise_main(grader_error=True)

    def test_auth_failure_stops_before_any_inference(self):
        self.exercise_main(denied=True)


if __name__ == "__main__":
    unittest.main()
