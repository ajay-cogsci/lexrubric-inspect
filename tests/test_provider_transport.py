"""Offline transport-contract checks: no credentials and no API calls."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lexrubric_inspect.provider_transport import (ANTHROPIC_API_VERSION, RequestsTransport,
                                transform_headers, transform_payload,
                                validate_reasoning_policy, validate_temperature_policy)


def judge_payload():
    return {"model": "qwen3.6-27b", "messages": [{"role": "user", "content": "完整中文评分提示"}],
            "temperature": 0, "max_tokens": 4096, "enable_thinking": False}


class PayloadTests(unittest.TestCase):
    def test_qwen_disables_thinking_and_pins_alibaba_without_prompt_changes(self):
        source = judge_payload()
        saved = deepcopy(source)
        transformed = transform_payload(source, "openrouter", "judge")
        self.assertEqual(transformed, {
            "model": "qwen/qwen3.6-27b", "messages": source["messages"],
            "temperature": 0, "max_tokens": 4096, "reasoning": {"enabled": False},
            "provider": {"only": ["alibaba"], "allow_fallbacks": False, "require_parameters": True}})
        transformed["messages"][0]["content"] = "changed"
        self.assertEqual(source, saved)

    def test_astra_temperature_exception_is_opt_in_and_candidate_only(self):
        source = {"model": "gpt-6-astra", "messages": [], "temperature": .6, "max_tokens": 16384}
        strict = transform_payload(source, "openai", "candidate")
        authorized = transform_payload(source, "openai", "candidate", True)
        self.assertEqual(strict["temperature"], .6)
        self.assertNotIn("temperature", authorized)
        self.assertEqual(authorized["max_completion_tokens"], 16384)
        self.assertNotIn("max_tokens", authorized)
        self.assertEqual(source["max_tokens"], 16384)
        self.assertEqual(transform_payload(source, "openai", "judge", True), source)

    def test_no_model_fallback_and_no_unrequested_setting_changes(self):
        source = judge_payload()
        self.assertEqual(transform_payload(source, "raw", "judge"), source)
        self.assertEqual(transform_payload(source, "openrouter", "candidate"), source)
        source["model"] = "other-model"
        self.assertEqual(transform_payload(source, "openrouter", "judge"), source)

    def test_gpt41_judge_removes_only_explicit_false_thinking(self):
        source = {**judge_payload(), "model": "gpt-4.1"}
        self.assertEqual(transform_payload(source, "openai", "judge"),
                         {key: value for key, value in source.items() if key != "enable_thinking"})
        source["enable_thinking"] = True
        self.assertEqual(transform_payload(source, "openai", "judge"), source)

    def test_conflicting_token_limits_fail_before_network(self):
        with self.assertRaises(ValueError):
            transform_payload({"model": "gpt-6-astra", "max_tokens": 12,
                               "max_completion_tokens": 15}, "openai", "candidate")

    def test_sol_token_field_changes_but_temperature_requires_explicit_flag(self):
        source = {"model": "gpt-6.1-sol", "messages": [{"role": "user", "content": "原始问题"}],
                  "temperature": .6, "max_tokens": 16384}
        strict = transform_payload(source, "openai", "candidate")
        self.assertEqual(strict, {"model": source["model"], "messages": source["messages"],
                                  "temperature": .6, "max_completion_tokens": 16384})
        approved = transform_payload(source, "openai", "candidate", True)
        self.assertEqual(approved, {key: value for key, value in strict.items() if key != "temperature"})
        self.assertEqual(source["temperature"], .6)
        self.assertNotIn("reasoning_effort", approved)

    def test_fable_body_remains_native_and_temperature_omission_is_opt_in(self):
        source = {"model": "claude-fable-5-2", "messages": [{"role": "user", "content": "原始问题"}],
                  "temperature": .6, "max_tokens": 16384}
        self.assertEqual(transform_payload(source, "anthropic", "candidate"), source)
        approved = transform_payload(source, "anthropic", "candidate", True)
        self.assertEqual(approved, {key: value for key, value in source.items() if key != "temperature"})
        self.assertNotIn("thinking", approved)
        self.assertNotIn("system", approved)
        self.assertNotIn("tools", approved)

    def test_qwen_candidate_pin_preserves_paper_temperature_and_default_reasoning(self):
        source = {"model": "qwen/qwen4-max", "messages": [{"role": "user", "content": "原始问题"}],
                  "temperature": .6, "max_tokens": 16384}
        self.assertEqual(transform_payload(source, "openrouter", "candidate"),
                         {**source, "provider": {"only": ["alibaba"], "allow_fallbacks": False,
                                                  "require_parameters": True}})

    def test_opus_replacement_omits_only_temperature_when_explicitly_requested(self):
        source = {"model": "claude-opus-5-5", "messages": [{"role": "user", "content": "原始问题"}],
                  "temperature": .6, "max_tokens": 16384}
        self.assertEqual(transform_payload(source, "anthropic", "candidate"), source)
        self.assertEqual(transform_payload(source, "anthropic", "candidate", True),
                         {key: value for key, value in source.items() if key != "temperature"})
        self.assertEqual(transform_payload(source, "anthropic", "judge", True), source)
        for provider, model in (("anthropic", "claude-opus-5-5-later"),
                                ("openrouter", "anthropic/claude-opus-5-5")):
            with self.subTest(provider=provider, model=model), self.assertRaises(ValueError):
                transform_payload({**source, "model": model}, provider, "candidate", True)

    def test_deepseek_snapshot_pin_and_explicit_temperature_omission(self):
        source = {"model": "deepseek/deepseek-v4.1-flash-1017", "messages": [],
                  "temperature": .6, "max_tokens": 16384}
        strict = transform_payload(source, "openrouter", "candidate")
        self.assertEqual(strict, {**source, "provider": {"only": ["deepseek"],
                                                        "allow_fallbacks": False, "require_parameters": True}})
        approved = transform_payload(source, "openrouter", "candidate", True)
        self.assertEqual(approved, {key: value for key, value in strict.items() if key != "temperature"})
        self.assertNotIn("reasoning", approved)
        self.assertNotIn("reasoning_effort", approved)

    def test_zai_payload_is_unchanged(self):
        source = {"model": "glm-5.3", "messages": [{"role": "user", "content": "原始问题"}],
                  "temperature": .6, "max_tokens": 16384}
        result = transform_payload(source, "zai", "candidate")
        self.assertEqual(result, source)
        result["messages"][0]["content"] = "mutation"
        self.assertEqual(source["messages"][0]["content"], "原始问题")

    def test_temperature_flag_never_broadens_to_other_models_or_providers(self):
        for provider, model in (("openai", "gpt-6.1-sol-later"), ("openrouter", "openai/gpt-6.1-sol"),
                                ("anthropic", "claude-fable-5-3"), ("zai", "glm-5.3"),
                                ("openrouter", "qwen/qwen4-max")):
            with self.subTest(provider=provider, model=model), self.assertRaises(ValueError):
                transform_payload({"model": model, "temperature": .6}, provider, "candidate", True)

    def test_candidate_exception_flag_never_changes_fixed_judge(self):
        self.assertEqual(transform_payload(judge_payload(), "openrouter", "judge", True),
                         transform_payload(judge_payload(), "openrouter", "judge", False))


class TemperaturePolicyTests(unittest.TestCase):
    def config(self, provider="openai", model="gpt-6.1-sol", mode="omit"):
        return {"candidate_provider": provider, "candidate_model": model,
                "candidate_temperature_policy": {
                    "mode": mode, "provider": provider, "model": model,
                    "reason": "Documented model sampling behavior",
                    "documentation": "https://example.invalid/official-documentation"},
                "approval_status": "approved", "authorized_exception": "Explicit approval fixture"}

    def test_each_frozen_omission_candidate_requires_explicit_policy(self):
        for provider, model in (("openai", "gpt-6.1-sol"), ("anthropic", "claude-fable-5-2"),
                                ("openrouter", "deepseek/deepseek-v4.1-flash-1017")):
            with self.subTest(model=model):
                self.assertTrue(validate_temperature_policy(self.config(provider, model)))

    def test_opus_replacement_requires_its_exact_approved_policy(self):
        config = self.config("anthropic", "claude-opus-5-5")
        config["candidate_api_version"] = ANTHROPIC_API_VERSION
        self.assertTrue(validate_temperature_policy(config))
        config["approval_status"] = "pending"
        with self.assertRaises(ValueError):
            validate_temperature_policy(config)
        config["approval_status"] = "approved"
        config["candidate_temperature_policy"]["model"] = "claude-fable-5-2"
        with self.assertRaises(ValueError):
            validate_temperature_policy(config)
        config.pop("candidate_temperature_policy")
        config["allow_temperature_omission"] = True
        with self.assertRaises(ValueError):
            validate_temperature_policy(config)

    def test_pending_or_missing_approval_never_authorizes_omission(self):
        for value in (None, "pending", "rejected", "Approved"):
            config = self.config()
            config["approval_status"] = value
            with self.subTest(status=value), self.assertRaises(ValueError):
                validate_temperature_policy(config)
        config = self.config()
        config["authorized_exception"] = " "
        with self.assertRaises(ValueError):
            validate_temperature_policy(config)

    def test_policy_must_match_exact_model_and_provider(self):
        for field, value in (("model", "gpt-6-astra"), ("provider", "openrouter")):
            config = self.config()
            config["candidate_temperature_policy"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_temperature_policy(config)

    def test_unknown_model_and_incomplete_policy_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_temperature_policy(self.config(model="gpt-future"))
        for field in ("reason", "documentation"):
            config = self.config()
            config["candidate_temperature_policy"][field] = ""
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_temperature_policy(config)

    def test_paper_policy_preserves_point_six_without_requesting_exception(self):
        for provider, model in (("zai", "glm-5.3"), ("openrouter", "qwen/qwen4-max")):
            config = self.config(provider, model, "paper")
            config["approval_status"] = "pending"
            config.pop("authorized_exception")
            self.assertFalse(validate_temperature_policy(config))

    def test_conflicting_legacy_and_modern_flags_are_rejected(self):
        config = self.config()
        config["allow_temperature_omission"] = False
        with self.assertRaises(ValueError):
            validate_temperature_policy(config)

    def test_existing_authorized_astra_configuration_remains_compatible(self):
        config = {"candidate_provider": "openai", "candidate_model": "gpt-6-astra",
                  "allow_temperature_omission": True, "authorized_exception": "Existing approval"}
        self.assertTrue(validate_temperature_policy(config))
        config["candidate_model"] = "gpt-6.1-sol"
        with self.assertRaises(ValueError):
            validate_temperature_policy(config)

    def test_anthropic_api_version_is_pinned(self):
        config = self.config("anthropic", "claude-fable-5-2")
        config["candidate_api_version"] = ANTHROPIC_API_VERSION
        self.assertTrue(validate_temperature_policy(config))
        config["candidate_api_version"] = "2023-06-01"
        with self.assertRaises(ValueError):
            validate_temperature_policy(config)


class ReasoningPolicyTests(unittest.TestCase):
    def config(self):
        return {"candidate_provider": "zai", "candidate_model": "glm-5.3",
                "candidate_reasoning_effort": "high", "approval_status": "approved",
                "authorized_reasoning_exception": "User explicitly requested high reasoning effort",
                "candidate_chat_url": "https://example.invalid/candidate",
                "candidate_api_key_env": "DUMMY_CANDIDATE_KEY",
                "judge_provider": "openrouter", "judge_model": "qwen3.6-27b",
                "judge_chat_url": "https://example.invalid/judge",
                "judge_api_key_env": "DUMMY_JUDGE_KEY"}

    def test_glm_high_requires_exact_model_provider_and_explicit_approval(self):
        self.assertEqual(validate_reasoning_policy(self.config()), "high")
        self.assertIsNone(validate_reasoning_policy({}))
        for field, value in (("candidate_model", "glm-5.3-later"),
                             ("candidate_provider", "openrouter"),
                             ("candidate_reasoning_effort", "max"),
                             ("candidate_reasoning_effort", ""),
                             ("approval_status", "pending"),
                             ("authorized_reasoning_exception", " ")):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_reasoning_policy({**self.config(), field: value})

    def test_glm_payload_adds_only_high_effort_and_never_changes_judge(self):
        source = {"model": "glm-5.3", "messages": [{"role": "user", "content": "原始问题"}],
                  "temperature": .6, "max_tokens": 16384}
        saved = deepcopy(source)
        self.assertEqual(transform_payload(source, "zai", "candidate"), source)
        result = transform_payload(source, "zai", "candidate", candidate_reasoning_effort="high")
        self.assertEqual(result, {**source, "reasoning_effort": "high"})
        result["messages"][0]["content"] = "mutation"
        self.assertEqual(source, saved)
        for payload, provider, role, effort in (
                (judge_payload(), "openrouter", "judge", "high"),
                ({**source, "model": "qwen/qwen4-max"}, "openrouter", "candidate", "high"),
                (source, "zai", "candidate", "max"),
                ({**source, "reasoning_effort": "max"}, "zai", "candidate", "high")):
            with self.subTest(provider=provider, role=role, effort=effort), self.assertRaises(ValueError):
                transform_payload(payload, provider, role, candidate_reasoning_effort=effort)

    def test_configuration_rejects_unapproved_effort_before_endpoint_access(self):
        from lexrubric_inspect import run
        config = {**self.config(), "approval_status": "pending"}
        with patch.dict(os.environ, {}, clear=True), patch.object(run, "endpoint_config") as endpoint:
            with self.assertRaises(ValueError):
                run.configure(config)
            endpoint.assert_not_called()
            self.assertNotIn("LEXRUBRIC_CANDIDATE_REASONING_EFFORT", os.environ)

    def test_configuration_wires_candidate_only_and_clears_stale_override(self):
        from lexrubric_inspect import lexrubric_port as port
        from lexrubric_inspect import run
        with (patch.dict(os.environ, {}, clear=True),
              patch.object(run, "endpoint_config", return_value=("https://example.invalid", "dummy")),
              patch.object(port, "endpoint_config", return_value=("https://example.invalid", "dummy"))):
            config = self.config()
            run.configure(config)
            self.assertEqual(os.environ["LEXRUBRIC_CANDIDATE_REASONING_EFFORT"], "high")
            candidate = port.configured_upstream("candidate")
            judge = port.configured_upstream("judge")
            self.assertEqual(candidate.requests.candidate_reasoning_effort, "high")
            self.assertIsNone(judge.requests.candidate_reasoning_effort)
            self.assertEqual(run.candidate_configuration(config)["authorized_reasoning_exception"],
                             config["authorized_reasoning_exception"])
            config.pop("candidate_reasoning_effort")
            config.pop("authorized_reasoning_exception")
            run.configure(config)
            self.assertNotIn("LEXRUBRIC_CANDIDATE_REASONING_EFFORT", os.environ)
            self.assertIsNone(port.configured_upstream("candidate").requests.candidate_reasoning_effort)

    def test_original_runner_transmits_only_approved_extra_field_and_audits_it(self):
        from lexrubric_inspect.lexrubric_port import load_upstream
        ref = load_upstream()
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({"choices": [{"message": {"content": "final answer"}}]}).encode()
        backend = SimpleNamespace(post=Mock(return_value=response), exceptions=requests.exceptions)
        transport = RequestsTransport(backend, "zai", "candidate", candidate_reasoning_effort="high")
        ref.requests = transport
        ref.DASHSCOPE_API_KEY = "dummy-glm-test-secret"
        question = "保持原问题\n{原样}"
        result = ref.generate_model_response({"name": "glm-5.3", "provider": "dashscope_chat",
                                              "enable_thinking": None, "res_max_tokens": 16384,
                                              "temperature": .6}, question, "fixture")
        self.assertFalse(result["error"])
        self.assertEqual(result["model_response"], "final answer")
        self.assertEqual(backend.post.call_count, 1)
        self.assertIs(backend.post.return_value, response)
        self.assertEqual(backend.post.call_args.kwargs["json"], {
            "model": "glm-5.3", "messages": [{"role": "user", "content": question}],
            "temperature": .6, "max_tokens": 16384, "reasoning_effort": "high"})
        self.assertEqual(transport.audit[0]["parameters"], {
            "temperature": .6, "max_tokens": 16384, "reasoning_effort": "high"})
        self.assertNotIn("dummy-glm-test-secret", str(transport.audit))
        self.assertNotIn(question, str(transport.audit))


class AnthropicTransportTests(unittest.TestCase):
    def test_native_headers_retain_bearer_auth_and_add_only_pinned_version(self):
        headers = {"Authorization": "Bearer dummy-test-secret", "Content-Type": "application/json"}
        result = transform_headers(headers, "anthropic", "candidate")
        self.assertEqual(result, {**headers, "anthropic-version": "2026-10-21"})
        self.assertNotIn("anthropic-version", headers)
        self.assertEqual(transform_headers(headers, "anthropic", "judge"), headers)
        with self.assertRaises(ValueError):
            transform_headers({**headers, "anthropic-version": "2023-06-01"}, "anthropic", "candidate")

    def native_reference(self, status, body):
        from lexrubric_inspect.lexrubric_port import load_upstream
        ref = load_upstream()
        response = requests.Response()
        response.status_code = status
        response._content = json.dumps(body, ensure_ascii=False).encode("utf-8")
        backend = SimpleNamespace(post=Mock(return_value=response), exceptions=requests.exceptions)
        transport = RequestsTransport(backend, "anthropic", "candidate", True)
        ref.requests = transport
        ref.DASHSCOPE_CHAT_URL = "https://api.anthropic.com/v1/messages"
        ref.DASHSCOPE_API_KEY = "dummy-native-test-secret"
        return ref, backend, response, transport

    def test_original_runner_extracts_native_text_and_preserves_exact_question(self):
        ref, backend, response, transport = self.native_reference(200, {
            "model": "claude-fable-5-2", "type": "message",
            "content": [{"type": "thinking", "thinking": "not final text"},
                        {"type": "text", "text": "第一段"}, {"type": "text", "text": "第二段"}],
            "usage": {"input_tokens": 5, "output_tokens": 7}})
        question = "中文原始问题\n{不改写}"
        config = {"name": "claude-fable-5-2", "provider": "dashscope_chat",
                  "enable_thinking": None, "res_max_tokens": 16384, "temperature": .6}
        result = ref.generate_model_response(config, question, "fixture")
        self.assertFalse(result["error"])
        self.assertEqual(result["model_response"], "第一段\n第二段")
        self.assertEqual(backend.post.call_count, 1)
        self.assertEqual(backend.post.call_args.args, ("https://api.anthropic.com/v1/messages",))
        sent = backend.post.call_args.kwargs
        self.assertEqual(sent["json"], {"model": "claude-fable-5-2", "max_tokens": 16384,
                                        "messages": [{"role": "user", "content": question}]})
        self.assertEqual(sent["headers"], {"Content-Type": "application/json",
                                           "Authorization": "Bearer dummy-native-test-secret",
                                           "anthropic-version": "2026-10-21"})
        self.assertEqual(sent["timeout"], ref.API_TIMEOUT)
        self.assertIs(backend.post.return_value, response)
        self.assertEqual(transport.audit[0]["api_version"], "2026-10-21")
        self.assertTrue(transport.audit[0]["temperature_omitted"])
        self.assertEqual(transport.audit[0]["parameters"], {"max_tokens": 16384})
        self.assertNotIn("dummy-native-test-secret", str(transport.audit))
        self.assertNotIn(question, str(transport.audit))

    def test_native_529_retains_original_nonretry_behavior(self):
        ref, backend, response, transport = self.native_reference(529, {
            "type": "error", "error": {"type": "overloaded_error", "message": "overloaded"}})
        with patch.object(ref.time, "sleep") as sleep:
            answer = ref.call_dashscope_chat("claude-fable-5-2", [{"role": "user", "content": "q"}])
        self.assertTrue(answer.startswith("[API HTTP 529:"))
        self.assertEqual(backend.post.call_count, 1)
        sleep.assert_not_called()
        self.assertEqual(len(transport.audit), 1)

    def test_opus_original_runner_preserves_native_question_cap_and_secret_redaction(self):
        ref, backend, response, transport = self.native_reference(200, {
            "model": "claude-opus-5-5", "type": "message",
            "content": [{"type": "thinking", "thinking": "private reasoning"},
                        {"type": "text", "text": "最终答案"}]})
        question = "原始问题\n{保持原样}"
        result = ref.generate_model_response({"name": "claude-opus-5-5", "provider": "dashscope_chat",
                                              "enable_thinking": None, "res_max_tokens": 16384,
                                              "temperature": .6}, question, "fixture")
        self.assertEqual(result["model_response"], "最终答案")
        self.assertFalse(result["error"])
        self.assertEqual(backend.post.call_count, 1)
        self.assertEqual(backend.post.call_args.kwargs["json"], {
            "model": "claude-opus-5-5", "max_tokens": 16384,
            "messages": [{"role": "user", "content": question}]})
        self.assertEqual(backend.post.call_args.kwargs["headers"]["anthropic-version"], ANTHROPIC_API_VERSION)
        self.assertEqual(transport.audit[0]["parameters"], {"max_tokens": 16384})
        self.assertNotIn("dummy-native-test-secret", str(transport.audit))
        self.assertNotIn(question, str(transport.audit))


class TransportTests(unittest.TestCase):
    def make_transport(self, status=200, body=None, fail=None):
        response = SimpleNamespace(status_code=status,
                                   content=json.dumps(body or {"model": "qwen/qwen3.6-27b", "usage": {"total_tokens": 17}, "provider": "Alibaba"}).encode())
        calls = []
        def post(url, **kwargs):
            calls.append((url, kwargs))
            if fail is not None:
                raise fail
            return response
        fake = SimpleNamespace(post=post, exceptions=SimpleNamespace(Timeout=TimeoutError))
        return RequestsTransport(fake, "openrouter", "judge"), calls, response, fake

    def call(self, transport):
        return transport.post("https://example.invalid/chat", json=judge_payload(),
                              headers={"Authorization": "Bearer secret_test_credential"}, timeout=180)

    def test_original_response_and_arguments_pass_through_with_compact_audit(self):
        transport, calls, response, fake = self.make_transport()
        self.assertIs(self.call(transport), response)
        self.assertIs(transport.exceptions, fake.exceptions)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["timeout"], 180)
        self.assertEqual(calls[0][1]["headers"]["Authorization"], "Bearer secret_test_credential")
        row = transport.audit[0]
        self.assertEqual(row["http_status"], 200)
        self.assertEqual(row["response_provider"], "Alibaba")
        self.assertEqual(row["usage"], {"total_tokens": 17})
        raw = json.dumps(judge_payload()["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(row["prompt_sha256"], hashlib.sha256(raw).hexdigest())
        serialized = json.dumps(row, ensure_ascii=False)
        self.assertNotIn("secret_test_credential", serialized)
        self.assertNotIn("完整中文评分提示", serialized)
        self.assertNotIn("messages", row)

    def test_http_errors_remain_responses_and_credentials_are_redacted(self):
        transport, calls, response, _ = self.make_transport(401, {"error": {
            "code": "unauthorized", "message": "Invalid secret_test_credential or sk-anotherkey"}})
        self.assertIs(self.call(transport), response)
        self.assertEqual(len(calls), 1)
        row = transport.audit[0]
        self.assertEqual(row["error"]["code"], "unauthorized")
        self.assertNotIn("secret_test_credential", str(row))
        self.assertNotIn("sk-anotherkey", str(row))

    def test_transport_exception_identity_is_preserved_without_retries(self):
        error = TimeoutError("timed out Bearer secret_test_credential")
        transport, calls, _, _ = self.make_transport(fail=error)
        with self.assertRaises(TimeoutError) as caught:
            self.call(transport)
        self.assertIs(caught.exception, error)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("secret_test_credential", str(transport.audit))

    def test_invalid_json_does_not_change_response_or_raise(self):
        transport, _, response, _ = self.make_transport()
        response.content = b"not JSON"
        self.assertIs(self.call(transport), response)
        self.assertEqual(transport.audit[0]["response_parse_error"], "invalid_json")

    def test_concurrent_audit_is_complete_and_snapshots_are_independent(self):
        transport, calls, _, _ = self.make_transport()
        with ThreadPoolExecutor(max_workers=10) as pool:
            list(pool.map(lambda _: self.call(transport), range(40)))
        self.assertEqual(len(calls), 40)
        self.assertEqual(len(transport.audit), 40)
        snapshot = transport.audit
        snapshot[0]["usage"]["total_tokens"] = 999
        snapshot.clear()
        self.assertEqual(len(transport.audit), 40)
        self.assertEqual(transport.audit[0]["usage"]["total_tokens"], 17)


if __name__ == "__main__":
    unittest.main()
