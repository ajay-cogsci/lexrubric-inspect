"""Run original LexRubric in Inspect, with explicit, recorded provider choices.

Default config selects the user-authorized Astra temperature exception. The task
itself remains strict by default. No credential values are written to disk.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from unittest.mock import patch

from inspect_ai import eval as inspect_eval
from inspect_ai import score as inspect_score
from inspect_ai.log import read_eval_log, write_eval_log

from .lexrubric_port import (DIMENSIONS, JUDGE_MODEL, ROOT as ASSET_ROOT, endpoint_config,
                           lexrubric, lexrubric_scorer, load_dataset, verify_sources)

ROOT = Path(".")


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def configure(config):
    from .provider_transport import validate_reasoning_policy, validate_temperature_policy
    omission = validate_temperature_policy(config)
    reasoning_effort = validate_reasoning_policy(config)
    config["allow_temperature_omission"] = omission
    for role in ("candidate", "judge"):
        for key in ("provider", "chat_url", "api_key_env"):
            os.environ[f"LEXRUBRIC_{role.upper()}_{key.upper()}"] = config[f"{role}_{key}"]
        endpoint_config(role)
    os.environ["LEXRUBRIC_ALLOW_TEMPERATURE_OMISSION"] = "true" if omission else "false"
    if reasoning_effort is None:
        os.environ.pop("LEXRUBRIC_CANDIDATE_REASONING_EFFORT", None)
    else:
        os.environ["LEXRUBRIC_CANDIDATE_REASONING_EFFORT"] = reasoning_effort


def candidate_configuration(config):
    """Generation identity, excluding independently changeable judge settings."""
    fields = {key: value for key, value in config.items() if key.startswith("candidate_")}
    fields["allow_temperature_omission"] = bool(config.get("allow_temperature_omission", False))
    if fields["allow_temperature_omission"]:
        fields["authorized_exception"] = config.get("authorized_exception")
    if config.get("candidate_reasoning_effort") is not None:
        fields["authorized_reasoning_exception"] = config.get("authorized_reasoning_exception")
    return fields


def validate_generation(log, split, config, provenance, limit):
    """Reject incompatible or incomplete generations before any regrading calls."""
    if log.status != "success":
        raise ValueError(f"Source {split} generation did not finish successfully")
    if log.eval.model != f"lexrubric_raw/{config['candidate_model']}":
        raise ValueError(f"Source {split} candidate model differs")
    meta = log.eval.metadata or {}
    source_config = meta.get("run_configuration")
    if not isinstance(source_config, dict) or candidate_configuration(source_config) != candidate_configuration(config):
        raise ValueError(f"Source {split} candidate configuration differs or is absent")
    if any(meta.get(key) != value for key, value in provenance.items()):
        raise ValueError(f"Source {split} pinned data/code/prompt provenance differs")
    if meta.get("split") != split or meta.get("protocol") != "original-paper-v1":
        raise ValueError(f"Source {split} protocol or split differs")
    if meta.get("verification_only"):
        raise ValueError("Synthetic verification logs cannot be reused as benchmark answers")
    expected = list(load_dataset(split))
    selected = expected[:limit] if limit is not None else expected
    selected_ids = [sample.id for sample in selected]
    actual = log.samples or []
    actual_ids = [sample.id for sample in actual]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(selected_ids):
        raise ValueError(f"Source {split} selected IDs do not match this run's --limit/dataset")
    if (log.eval.dataset.name != f"lexrubric_v1_{split}"
            or log.eval.dataset.samples != len(expected)
            or log.eval.dataset.shuffled
            or log.eval.dataset.sample_ids != selected_ids):
        raise ValueError(f"Source {split} dataset declaration/selection differs")
    if log.eval.config.epochs not in (None, 1):
        raise ValueError(f"Source {split} must have exactly one generation per sample")
    by_id = {sample.id: sample for sample in selected}
    expected_request = {"name": config["candidate_model"], "provider": "dashscope_chat",
                        "enable_thinking": None, "res_max_tokens": 16384, "temperature": 0.6}
    for sample in actual:
        original = by_id[sample.id]
        if sample.error or sample.scores:
            raise ValueError(f"Source {sample.id} is failed or scored; use preserved generation.eval files")
        if sample.input != original.input or sample.metadata != original.metadata or sample.target != original.target:
            raise ValueError(f"Source {sample.id} input/reference/rubric metadata differs")
        output = sample.output
        output_meta = output.metadata or {}
        if (output_meta.get("upstream_candidate_error") is not False
                or output_meta.get("request_parameters") != expected_request
                or not output.completion):
            raise ValueError(f"Source {sample.id} lacks a successful matching candidate response")
        if [(message.role, message.content) for message in sample.messages] != [
                ("user", original.input), ("assistant", output.completion)]:
            raise ValueError(f"Source {sample.id} contains a modified candidate conversation")
    return {"source_candidate_configuration": candidate_configuration(source_config),
            "source_judge_model": source_config.get("judge_model"),
            "selected_sample_ids": selected_ids, "samples": len(actual)}


def prepare_generation_import(source_dir, config, provenance, limit):
    """Validate both splits up front and record immutable source identities."""
    source_dir = source_dir.resolve(strict=True)
    if not source_dir.is_dir():
        raise ValueError("--generation-from must be a prior log directory")
    imported = {"source_log_dir": str(source_dir), "files": {}}
    for split in ("consultation", "exam"):
        path = source_dir / f"{split}-generation.eval"
        if not path.is_file():
            raise ValueError(f"Missing preserved generation file: {path}")
        content = path.read_bytes()
        log = read_eval_log(path)
        info = validate_generation(log, split, config, provenance, limit)
        if path.read_bytes() != content:
            raise ValueError(f"Source generation changed during validation: {path}")
        imported["files"][split] = {"source_path": str(path),
                                     "sha256": hashlib.sha256(content).hexdigest(),
                                     "bytes": len(content), **info}
    return imported


def copy_generation_import(info, destination):
    """Copy source bytes without modifying the old run or copying its score cache."""
    content = Path(info["source_path"]).read_bytes()
    if hashlib.sha256(content).hexdigest() != info["sha256"]:
        raise ValueError("Source generation changed after import validation")
    if destination.exists():
        if destination.read_bytes() != content:
            raise ValueError(f"Imported generation file was modified: {destination}")
        return
    temporary = destination.with_suffix(".import.tmp")
    temporary.write_bytes(content)
    temporary.replace(destination)


def regrading_metadata(generated, config, import_info=None):
    """Current judge configuration belongs to the new scored log, not its source."""
    generated.eval.task_args = {**(generated.eval.task_args or {}),
                                "judge_model": config["judge_model"]}
    generated.eval.task_args_passed = {**(generated.eval.task_args_passed or {}),
                                       "judge_model": config["judge_model"]}
    meta = dict(generated.eval.metadata or {})
    if import_info is not None:
        meta["generation_import"] = import_info
        meta["original_candidate_configuration"] = import_info["source_candidate_configuration"]
    meta["run_configuration"] = config
    meta["judge_model"] = config["judge_model"]
    meta["judge_changed_from_paper"] = config["judge_model"] != JUDGE_MODEL
    generated.eval.metadata = meta


def log_component(value):
    """Turn a provider-qualified model or run name into a safe path component."""
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", value.rsplit("/", 1)[-1]).strip(".-_")
    if not component:
        raise ValueError("Model and run names must contain a usable log filename")
    return component


def publication_paths(config, run_name):
    folder = ROOT / "logs" / log_component(config["judge_model"])
    stem = f"{log_component(config['candidate_model'])}-{log_component(run_name)}"
    return {split: folder / f"{stem}-{split}.eval" for split in ("consultation", "exam")}


def publish_full_run(logs, config, run_name, run_dir, working_dir, limit, summary):
    """Publish only finished full-run scored bytes; never expose working logs."""
    if limit is not None:
        return None
    if summary["questions"] != 649 or summary["criteria_expected"] != 12337:
        raise ValueError("Only the complete 649-question corpus can be published")
    if len(logs) != 2:
        raise ValueError("Both completed splits are required before publishing")
    by_split = {}
    for log in logs:
        split = (log.eval.metadata or {}).get("split")
        if split not in ("consultation", "exam") or split in by_split or log.status != "success":
            raise ValueError("Expected one successful scored log for each split")
        if (log.eval.metadata or {}).get("verification_only"):
            raise ValueError("Synthetic verification logs cannot be published")
        if (log.eval.metadata or {}).get("run_configuration") != config:
            raise ValueError("Scored log configuration differs from this run")
        expected = {sample.id: sample for sample in load_dataset(split)}
        samples = log.samples or []
        if len(samples) != len(expected) or {sample.id for sample in samples} != set(expected):
            raise ValueError(f"Incomplete or duplicated {split} sample coverage")
        for sample in samples:
            original = expected[sample.id]
            if (sample.error or not sample.scores or len(sample.scores) != 1
                    or sample.input != original.input or sample.metadata != original.metadata):
                raise ValueError(f"Invalid final scored sample: {sample.id}")
        by_split[split] = log
    destinations = publication_paths(config, run_name)
    payloads, files = {}, {}
    # Check every source and destination before publishing either split.
    for split, destination in destinations.items():
        source = working_dir / f"{split}-scored.eval"
        content = source.read_bytes()
        if destination.exists() and destination.read_bytes() != content:
            raise ValueError(f"Refusing to replace a different published log: {destination}")
        payloads[split] = content
        files[split] = {"path": str(destination), "working_path": str(source),
                        "sha256": hashlib.sha256(content).hexdigest()}
    record = {"run_name": run_name, "candidate": config["candidate_model"],
              "judge": config["judge_model"], "questions": 649, "criteria_expected": 12337,
              "has_api_errors": summary["has_api_errors"],
              "benchmark_results_valid": summary["benchmark_results_valid"], "files": files}
    record_path = run_dir / "publication.json"
    previous = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else None
    if previous is not None and {k: v for k, v in previous.items() if k != "published_utc"} != record:
        raise ValueError("Existing publication record differs from this completed run")
    for split, destination in destinations.items():
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Stage outside the viewer tree, then atomically expose a complete file.
        with tempfile.NamedTemporaryFile(dir=run_dir, prefix="publish-", suffix=".tmp", delete=False) as handle:
            staged = Path(handle.name)
            handle.write(payloads[split])
        try:
            try:
                os.link(staged, destination)
            except FileExistsError:
                if destination.read_bytes() != payloads[split]:
                    raise ValueError(f"Published log appeared with different contents: {destination}")
        finally:
            staged.unlink(missing_ok=True)
    if previous is not None:
        return previous
    record["published_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(record_path, record)
    return record


def summarize(logs, config, limit, complete_dataset=True):
    records = []
    audits = []
    for log in logs:
        for sample in log.samples or []:
            if sample.error:
                raise RuntimeError(f"Inspect sample failed: {sample.id}")
            if not sample.scores or len(sample.scores) != 1:
                raise RuntimeError(f"Missing or ambiguous score: {sample.id}")
            scored = next(iter(sample.scores.values()))
            meta = scored.metadata
            records.append({"id": sample.id, "split": sample.metadata["split"],
                            "score_rate": scored.value, "raw_points": meta["raw_points"],
                            "dimension_rates": meta["dimension_rates"],
                            "candidate_error": meta["candidate_error"],
                            "grader_errors": meta["grader_errors"],
                            "criteria_expected": len(sample.metadata["rubrics"]),
                            "criteria_judged": len(meta["criterion_evaluations"])})
            audits.extend((sample.output.metadata or {}).get("http_audit", []))
            audits.extend(meta.get("http_audit", []))
    if len({r["id"] for r in records}) != len(records):
        raise RuntimeError("Duplicate scored question")
    summary = {"candidate": config["candidate_model"], "judge": config["judge_model"],
               "scope": "pilot" if limit else "full", "questions": len(records),
               "criteria_expected": sum(r["criteria_expected"] for r in records),
               "criteria_judged": sum(r["criteria_judged"] for r in records),
               "candidate_errors": sum(r["candidate_error"] for r in records),
               "grader_errors": sum(r["grader_errors"] for r in records),
               "authorized_exception": config.get("authorized_exception") if config.get("allow_temperature_omission") else None,
               "authorized_judge_exception": config.get("judge_authorized_exception"),
               "judge_changed_from_paper": config["judge_model"] != JUDGE_MODEL,
               "data_and_code": verify_sources(), "splits": {}, "dimensions": {},
               "http_attempts": len(audits), "records": records}
    summary["overall_score_rate_percent"] = sum(r["score_rate"] for r in records) / len(records)
    for split in ("consultation", "exam"):
        group = [r for r in records if r["split"] == split]
        summary["splits"][split] = {"questions": len(group),
                                    "score_rate_percent": sum(r["score_rate"] for r in group) / len(group) if group else None}
    for english, chinese in DIMENSIONS.items():
        values = [r["dimension_rates"].get(chinese) for r in records]
        values = [v for v in values if v is not None]
        summary["dimensions"][english] = {"eligible_questions": len(values),
                                          "score_rate_percent": sum(values) / len(values) if values else None}
    if not limit and complete_dataset and (len(records) != 649 or summary["criteria_expected"] != 12337):
        raise RuntimeError("Full run does not cover the original-paper corpus")
    summary["has_api_errors"] = bool(summary["candidate_errors"] or summary["grader_errors"])
    summary["benchmark_results_valid"] = not summary["has_api_errors"]
    summary["score_interpretation"] = (
        "Raw upstream scores retained for diagnostics; API errors invalidate benchmark results."
        if summary["has_api_errors"] else
        "Scores for the recorded configuration and scope; protocol exceptions remain recorded."
    )
    return summary


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ASSET_ROOT / "run_config.json")
    parser.add_argument("--output-dir", type=Path,
                        help="Writable directory for results and logs (default: current directory)")
    parser.add_argument("--run-name", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--limit", type=int, help="Pilot questions per split; omitted runs all 649")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--judge-model", help="Explicit future judge change; recorded as a protocol variant")
    parser.add_argument("--judge-provider", choices=["openai", "openrouter", "upstream"])
    parser.add_argument("--generation-from", type=Path,
                        help="Reuse both preserved generation.eval files from a prior log directory in a NEW run")
    args = parser.parse_args()
    if args.output_dir is not None:
        ROOT = args.output_dir
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        parser.error("run-name must be a directory name")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    archived_run = ROOT / "results" / args.run_name
    if (archived_run / "storage.json").is_file():
        raise ValueError(
            "This completed run has been reorganized and cannot be re-executed. "
            "Choose a fresh --run-name and reuse its saved answers with "
            f"--generation-from {archived_run / 'generation'}."
        )
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("approval_status") == "pending":
        raise ValueError("This planned run is awaiting user approval; no provider setup or inference was started")
    if args.judge_model:
        config["judge_model"] = args.judge_model
    if args.judge_provider:
        config["judge_provider"] = args.judge_provider
        if args.judge_provider == "openai":
            config["judge_chat_url"] = "https://api.openai.com/v1/chat/completions"
            config["judge_api_key_env"] = "OPENAI_API_KEY"
    configure(config)
    provenance = verify_sources()
    dataset = load_dataset()
    run_dir = ROOT / "results" / args.run_name
    log_dir = run_dir / "working"
    if not (run_dir / "run_manifest.json").exists() and any(
            path.exists() for path in publication_paths(config, args.run_name).values()):
        raise ValueError("Published logs already use this model/run name; choose a fresh --run-name")
    generation_import = None
    if args.generation_from is not None:
        if args.generation_from.resolve() == log_dir.resolve():
            raise ValueError("--generation-from requires a different, new destination run")
        generation_import = prepare_generation_import(args.generation_from, config, provenance, args.limit)
        if ((run_dir.exists() and any(run_dir.iterdir())) or (log_dir.exists() and any(log_dir.iterdir()))):
            prior_manifest = run_dir / "run_manifest.json"
            if not prior_manifest.is_file() or json.loads(prior_manifest.read_text(encoding="utf-8")).get("generation_import") != generation_import:
                raise ValueError("Generation reuse requires a new run directory/cache, or a matching import resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LEXRUBRIC_SCORE_CACHE"] = str(run_dir / "score-cache")
    manifest_path = run_dir / "run_manifest.json"
    manifest = {"configuration": config, "provenance": provenance,
                "inspect_ai_version": version("inspect-ai"), "requests_version": version("requests"),
                "original_questions": len(dataset), "original_criteria": sum(len(s.metadata["rubrics"]) for s in dataset),
                "limit_per_split": args.limit, "log_dir": str(log_dir),
                "log_layout": "private-working-final-only-v1"}
    if generation_import is not None:
        manifest["generation_import"] = generation_import
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError("Cannot resume this run with a different configuration")
    save_json(manifest_path, manifest)
    if args.preflight_only:
        print(json.dumps({"preflight": "passed", "questions": len(dataset), "criteria": 12337,
                          "candidate": config["candidate_model"], "judge": config["judge_model"],
                          "temperature_exception": config.get("allow_temperature_omission", False)}, indent=2))
        return
    status_path = run_dir / "status.json"

    def status(phase, **extra):
        state = {"phase": phase, "updated_utc": datetime.now(timezone.utc).isoformat(), **extra}
        save_json(status_path, state)
        print(json.dumps(state), flush=True)

    def local_path(base):
        def directory(package, *unused, **kwargs):
            destination = log_dir / ".inspect-state" / base / str(package)
            destination.mkdir(parents=True, exist_ok=True)
            return destination
        return directory

    logs = []
    try:
        # Keep Inspect's incidental trace/cache files inside this project too.
        with ExitStack() as stack:
            stack.enter_context(patch("inspect_ai._util.appdirs.user_data_path", local_path("data")))
            stack.enter_context(patch("inspect_ai._util.appdirs.user_cache_path", local_path("cache")))
            if not all((log_dir / f"{split}-scored.eval").exists() for split in ("consultation", "exam")):
                from .preflight import check_provider_access
                status("checking_access")
                checks = check_provider_access(config)
                save_json(run_dir / "access_check.json", checks)
            for split in ("consultation", "exam"):
                scored_path = log_dir / f"{split}-scored.eval"
                generation_path = log_dir / f"{split}-generation.eval"
                if scored_path.exists():
                    logs.append(read_eval_log(scored_path))
                    continue
                import_info = generation_import["files"][split] if generation_import is not None else None
                if import_info is not None:
                    status("importing_generation", split=split, questions=import_info["samples"])
                    copy_generation_import(import_info, generation_path)
                if generation_path.exists():
                    generated = read_eval_log(generation_path)
                else:
                    status("generating", split=split)
                    task = lexrubric(split, config["judge_model"])
                    task.metadata["run_configuration"] = config
                    generated = inspect_eval(task, model=f"lexrubric_raw/{config['candidate_model']}",
                                             score=False, limit=args.limit, max_samples=5,
                                             log_dir=str(log_dir), display="plain")[0]
                    if generated.status != "success":
                        raise RuntimeError(f"Inspect generation status: {generated.status}")
                    write_eval_log(generated, generation_path)
                regrading_metadata(generated, config, import_info)
                status("grading", split=split, generated_questions=len(generated.samples or []))
                scored = inspect_score(generated, scorers=lexrubric_scorer(config["judge_model"]),
                                       copy=False, display="plain", action="overwrite")
                write_eval_log(scored, scored_path)
                logs.append(scored)
                save_json(run_dir / f"{split}_summary.json", summarize([scored], config, args.limit, complete_dataset=False))
            summary = summarize(logs, config, args.limit)
            save_json(run_dir / "summary.json", summary)
            publication = publish_full_run(logs, config, args.run_name, run_dir, log_dir, args.limit, summary)
            status("complete_with_errors" if summary["has_api_errors"] else "complete",
                   benchmark_results_valid=summary["benchmark_results_valid"],
                   score_interpretation=summary["score_interpretation"],
                   questions=summary["questions"], criteria_judged=summary["criteria_judged"],
                   candidate_errors=summary["candidate_errors"], grader_errors=summary["grader_errors"],
                   overall_score_rate_percent=summary["overall_score_rate_percent"],
                   published_logs={split: info["path"] for split, info in publication["files"].items()}
                   if publication is not None else {})
    except Exception as e:
        status("failed", error=f"{type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
