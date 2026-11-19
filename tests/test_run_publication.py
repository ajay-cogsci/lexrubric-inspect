"""Offline checks for private working logs and full-run-only publication."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lexrubric_inspect import run


class FinalLogPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datasets = {split: list(run.load_dataset(split)) for split in ("consultation", "exam")}

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.root_patch = patch.object(run, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.config = {"candidate_model": "provider/fixture-candidate", "judge_model": "qwen/qwen3.6-27b"}
        self.run_name = "fixture-full"
        self.run_dir = self.root / "results" / self.run_name
        self.working = self.run_dir / "working"
        self.working.mkdir(parents=True)
        self.logs = []
        for split, dataset in self.datasets.items():
            samples = [NS(id=s.id, input=s.input, metadata=s.metadata, error=None,
                          scores={"fixture": NS(value=42)}) for s in dataset]
            self.logs.append(NS(status="success", samples=samples,
                                eval=NS(metadata={"split": split, "run_configuration": self.config})))
            (self.working / f"{split}-scored.eval").write_bytes(f"Complete fixture {split} bytes".encode())
            (self.working / f"{split}-generation.eval").write_bytes(b"Private generated answers")
        (self.working / "native-auto-generated.eval").write_bytes(b"Private automatic log")
        self.summary = {"questions": 649, "criteria_expected": 12337,
                        "has_api_errors": False, "benchmark_results_valid": True}

    def publish(self, *, logs=None, limit=None, summary=None):
        return run.publish_full_run(self.logs if logs is None else logs, self.config, self.run_name,
                                    self.run_dir, self.working, limit, summary or self.summary)

    def test_only_two_final_scored_files_are_atomically_published(self):
        real_link = run.os.link

        def check_stage(source, destination):
            self.assertTrue(Path(source).is_relative_to(self.run_dir))
            self.assertFalse(Path(source).is_relative_to(self.root / "logs"))
            self.assertFalse(Path(destination).exists())
            self.assertIn(b"Complete fixture", Path(source).read_bytes())
            real_link(source, destination)

        with patch.object(run.os, "link", side_effect=check_stage) as link:
            record = self.publish()
        self.assertEqual(link.call_count, 2)
        visible = set((self.root / "logs").rglob("*.eval"))
        self.assertEqual(visible, set(run.publication_paths(self.config, self.run_name).values()))
        self.assertTrue(all(p.parent.name == "qwen3.6-27b" for p in visible))
        self.assertTrue(all(p.name.startswith("fixture-candidate-fixture-full-") for p in visible))
        for split, info in record["files"].items():
            data = (self.working / f"{split}-scored.eval").read_bytes()
            self.assertEqual(Path(info["path"]).read_bytes(), data)
            self.assertEqual(info["sha256"], hashlib.sha256(data).hexdigest())
        self.assertFalse(list(self.run_dir.glob("publish-*.tmp")))
        self.assertEqual(self.publish(), record)  # Idempotent completed resume.

    def test_pilots_and_incomplete_runs_never_publish(self):
        self.assertIsNone(self.publish(limit=1))
        with self.assertRaisesRegex(ValueError, "Both completed splits"):
            self.publish(logs=self.logs[:1])
        self.logs[1].samples.pop()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.publish()
        self.assertFalse((self.root / "logs").exists())
        self.assertFalse((self.run_dir / "publication.json").exists())

    def test_verification_and_conflicting_logs_never_replace_final_files(self):
        self.logs[0].eval.metadata["verification_only"] = True
        with self.assertRaisesRegex(ValueError, "Synthetic"):
            self.publish()
        self.logs[0].eval.metadata.pop("verification_only")
        paths = run.publication_paths(self.config, self.run_name)
        paths["exam"].parent.mkdir(parents=True)
        paths["exam"].write_bytes(b"Existing different final result")
        with self.assertRaisesRegex(ValueError, "Refusing to replace"):
            self.publish()
        self.assertEqual(paths["exam"].read_bytes(), b"Existing different final result")
        self.assertFalse(paths["consultation"].exists())

    def test_complete_api_error_results_retain_the_invalidity_flag(self):
        summary = {**self.summary, "has_api_errors": True, "benchmark_results_valid": False}
        record = self.publish(summary=summary)
        self.assertTrue(record["has_api_errors"])
        self.assertFalse(record["benchmark_results_valid"])

    def test_pending_and_archived_runs_stop_before_provider_setup(self):
        config = self.root / "pending.json"
        config.write_text(json.dumps({"approval_status": "pending"}), encoding="utf-8")
        with (patch.object(sys, "argv", ["run.py", "--config", str(config), "--run-name", "pending-run"]),
              patch.object(run, "configure", side_effect=AssertionError("Must not configure")) as configure,
              patch.object(run, "inspect_eval", side_effect=AssertionError("Must not infer")) as infer):
            with self.assertRaisesRegex(ValueError, "awaiting user approval"):
                run.main()
            configure.assert_not_called()
            infer.assert_not_called()
        self.assertFalse((self.root / "results/pending-run").exists())
        (self.run_dir / "storage.json").write_text("{}", encoding="utf-8")
        with (patch.object(sys, "argv", ["run.py", "--config", str(self.root / "absent.json"),
                                        "--run-name", self.run_name]),
              patch.object(run, "configure", side_effect=AssertionError("Must not configure")) as configure):
            with self.assertRaisesRegex(ValueError, "completed run has been reorganized"):
                run.main()
            configure.assert_not_called()


if __name__ == "__main__":
    unittest.main()
