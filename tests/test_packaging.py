"""Offline regression checks for installed assets and writable output paths."""
from contextlib import redirect_stdout
import io
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from inspect_ai._util.module import load_module
from inspect_ai._util.registry import registry_lookup
from lexrubric_inspect import lexrubric_port, run


class PackagingTests(unittest.TestCase):
    def test_public_task_name_survives_source_tree_build_metadata(self):
        importlib.import_module("lexrubric_inspect._registry")
        factory = registry_lookup("task", "lexrubric_inspect/lexrubric")
        self.assertIsNotNone(factory)
        self.assertEqual(len(factory().dataset), 649)

    def test_public_task_and_legacy_entrypoint_have_identical_settings_and_samples(self):
        from lexrubric_inspect import lexrubric

        original = lexrubric_port.lexrubric(split="exam")
        public = lexrubric(split="exam")
        self.assertEqual(public.config, original.config)
        self.assertEqual(public.metadata, original.metadata)
        self.assertEqual(list(public.dataset), list(original.dataset))
        task_module = load_module(lexrubric_port.ROOT / "tasks.py")
        self.assertEqual(task_module.lexrubric(split="exam").config, original.config)

    def test_preflight_writes_only_to_explicit_output_directory(self):
        asset_root = lexrubric_port.ROOT
        assets = {p.relative_to(asset_root): p.read_bytes()
                  for p in asset_root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run-output"
            args = ["lexrubric-run", "--preflight-only", "--run-name", "offline-package-check",
                    "--output-dir", str(output)]
            with (patch.object(sys, "argv", args), patch.object(run, "ROOT", Path(".")),
                  patch.dict(os.environ, {}, clear=True),
                  patch.object(run, "configure"), patch.object(run, "inspect_eval") as generate,
                  patch("lexrubric_inspect.preflight.check_provider_access") as access,
                  redirect_stdout(io.StringIO())):
                run.main()
            manifest_path = output / "results/offline-package-check/run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["original_questions"], 649)
            self.assertEqual(manifest["original_criteria"], 12337)
            self.assertEqual(manifest["configuration"], json.loads((asset_root / "run_config.json").read_text(encoding="utf-8")))
            self.assertEqual(Path(manifest["log_dir"]), output / "results/offline-package-check/working")
            generate.assert_not_called()
            access.assert_not_called()
        self.assertEqual(assets, {p.relative_to(asset_root): p.read_bytes()
                                 for p in asset_root.rglob("*")
                                 if p.is_file() and "__pycache__" not in p.parts})

    def test_direct_inspect_taskfile_can_load_its_provider_from_the_package(self):
        task_module = load_module(lexrubric_port.ROOT / "lexrubric_port.py")
        with (patch.dict(os.environ, {"LEXRUBRIC_CANDIDATE_PROVIDER": "zai"}, clear=True),
              patch.object(task_module, "endpoint_config",
                           return_value=("https://example.invalid", "synthetic-test-key"))):
            reference = task_module.configured_upstream("candidate")
        self.assertEqual(reference.requests.provider, "zai")
        self.assertIsNone(reference.requests.candidate_reasoning_effort)
        self.assertEqual(len(task_module.load_dataset()), 649)


if __name__ == "__main__":
    unittest.main()
