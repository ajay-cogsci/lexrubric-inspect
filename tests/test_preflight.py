"""Authentication failures must stop before any paid inference."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lexrubric_inspect.preflight import ProviderAccessError, check_provider_access
from lexrubric_inspect.provider_transport import ANTHROPIC_API_VERSION


class AccessTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "candidate_provider": "openai", "candidate_model": "gpt-6-astra",
            "candidate_chat_url": "https://api.openai.com/v1/chat/completions",
            "candidate_api_key_env": "TEST_OPENAI_KEY",
            "judge_provider": "openrouter", "judge_model": "qwen3.6-27b",
            "judge_chat_url": "https://openrouter.ai/api/v1/chat/completions",
            "judge_api_key_env": "TEST_OPENROUTER_KEY",
        }
        self.env = patch.dict(os.environ, {"TEST_OPENAI_KEY": "secret-a", "TEST_OPENROUTER_KEY": "secret-b",
                                          "TEST_ANTHROPIC_KEY": "secret-c", "TEST_ZAI_KEY": "secret-z"})
        self.env.start()
        self.addCleanup(self.env.stop)

    @staticmethod
    def response(status, body):
        return Mock(status_code=status, json=Mock(return_value=body))

    def test_success_uses_only_read_requests_and_no_saved_credentials(self):
        responses = [self.response(200, {"id": "gpt-6-astra"}),
                     self.response(200, {"data": {"balance": "private"}}),
                     self.response(200, {"data": [{"id": "qwen/qwen3.6-27b"}]})]
        with patch("lexrubric_inspect.preflight.requests.get", side_effect=responses) as get, patch("lexrubric_inspect.preflight.requests.post") as post:
            report = check_provider_access(self.config)
        self.assertEqual(get.call_count, 3)
        post.assert_not_called()
        self.assertTrue(all(item["authenticated"] for item in report))
        self.assertNotIn("secret", str(report))
        self.assertNotIn("balance", str(report))

    def test_unauthorized_judge_stops_and_does_not_echo_body(self):
        responses = [self.response(200, {"id": "gpt-6-astra"}),
                     self.response(401, {"error": "secret-b"})]
        with patch("lexrubric_inspect.preflight.requests.get", side_effect=responses) as get, patch("lexrubric_inspect.preflight.requests.post") as post:
            with self.assertRaisesRegex(ProviderAccessError, "judge openrouter.*401") as raised:
                check_provider_access(self.config)
        self.assertEqual(get.call_count, 2)
        post.assert_not_called()
        self.assertNotIn("secret", str(raised.exception))

    def test_absent_model_does_not_silently_fallback(self):
        responses = [self.response(200, {"id": "gpt-6-astra"}), self.response(200, {}),
                     self.response(200, {"data": [{"id": "openai/gpt-4.1"}]})]
        with patch("lexrubric_inspect.preflight.requests.get", side_effect=responses):
            with self.assertRaisesRegex(ProviderAccessError, "does not list"):
                check_provider_access(self.config)

    def test_configured_endpoint_mismatch_never_transmits_key(self):
        self.config["candidate_chat_url"] = "https://example.invalid/v1/chat/completions"
        with patch("lexrubric_inspect.preflight.requests.get") as get:
            with self.assertRaises(ProviderAccessError):
                check_provider_access(self.config)
        get.assert_not_called()

    def anthropic_candidate(self):
        self.config.update({"candidate_provider": "anthropic", "candidate_model": "claude-fable-5-2",
                            "candidate_chat_url": "https://api.anthropic.com/v1/messages",
                            "candidate_api_key_env": "TEST_ANTHROPIC_KEY",
                            "candidate_api_version": ANTHROPIC_API_VERSION})

    def zai_candidate(self):
        self.config.update({"candidate_provider": "zai", "candidate_model": "glm-5.3",
                            "candidate_chat_url": "https://api.z.ai/api/paas/v4/chat/completions",
                            "candidate_api_key_env": "TEST_ZAI_KEY"})

    def judge_responses(self):
        return [self.response(200, {"data": {"balance": "private"}}),
                self.response(200, {"data": [{"id": "qwen/qwen3.6-27b"}]})]

    def test_anthropic_exact_model_uses_versioned_native_get_and_bearer_auth(self):
        self.anthropic_candidate()
        response = self.response(200, {"id": "claude-fable-5-2", "private": "secret-c"})
        with patch("lexrubric_inspect.preflight.requests.get", side_effect=[response, *self.judge_responses()]) as get, \
                patch("lexrubric_inspect.preflight.requests.post") as post:
            report = check_provider_access(self.config)
        self.assertEqual(get.call_count, 3)
        self.assertEqual(get.call_args_list[0].args, ("https://api.anthropic.com/v1/models/claude-fable-5-2",))
        self.assertEqual(get.call_args_list[0].kwargs, {
            "headers": {"Authorization": "Bearer secret-c", "anthropic-version": "2026-10-21"},
            "timeout": 30})
        self.assertEqual(report[0], {"role": "candidate", "provider": "anthropic",
                                     "model": "claude-fable-5-2", "resolved_model": "claude-fable-5-2",
                                     "api_version": ANTHROPIC_API_VERSION,
                                     "authenticated": True, "model_available": True})
        self.assertNotIn("secret", str(report))
        self.assertNotIn("private", str(report))
        post.assert_not_called()

    def test_anthropic_dated_alias_resolution_is_recorded(self):
        self.anthropic_candidate()
        resolved = "claude-fable-5-2-20261110"
        with patch("lexrubric_inspect.preflight.requests.get", side_effect=[self.response(200, {"id": resolved}),
                                                         *self.judge_responses()]), \
                patch("lexrubric_inspect.preflight.requests.post") as post:
            report = check_provider_access(self.config)
        self.assertEqual(report[0]["model"], "claude-fable-5-2")
        self.assertEqual(report[0]["resolved_model"], resolved)
        post.assert_not_called()

    def test_anthropic_different_model_or_arbitrary_suffix_never_confirms_access(self):
        self.anthropic_candidate()
        for resolved in ("claude-fable-5-3", "claude-fable-5-2-other", "claude-fable-5-2-20261110-extra", None, 52):
            with self.subTest(resolved=resolved), \
                    patch("lexrubric_inspect.preflight.requests.get", return_value=self.response(200, {"id": resolved})) as get, \
                    patch("lexrubric_inspect.preflight.requests.post") as post:
                with self.assertRaisesRegex(ProviderAccessError, "did not confirm"):
                    check_provider_access(self.config)
                self.assertEqual(get.call_count, 1)
                post.assert_not_called()

    def test_anthropic_unpinned_version_stops_before_any_request(self):
        self.anthropic_candidate()
        self.config["candidate_api_version"] = "2023-06-01"
        with patch("lexrubric_inspect.preflight.requests.get") as get, patch("lexrubric_inspect.preflight.requests.post") as post:
            with self.assertRaisesRegex(ProviderAccessError, "API version differs"):
                check_provider_access(self.config)
        get.assert_not_called()
        post.assert_not_called()

    def test_anthropic_endpoint_mismatch_stops_before_transmitting_credentials(self):
        self.anthropic_candidate()
        self.config["candidate_chat_url"] = "https://example.invalid/v1/messages"
        with patch("lexrubric_inspect.preflight.requests.get") as get, patch("lexrubric_inspect.preflight.requests.post") as post:
            with self.assertRaisesRegex(ProviderAccessError, "public Messages endpoint"):
                check_provider_access(self.config)
        get.assert_not_called()
        post.assert_not_called()

    def test_anthropic_authentication_failure_has_no_inference_or_body_echo(self):
        self.anthropic_candidate()
        with patch("lexrubric_inspect.preflight.requests.get", return_value=self.response(401, {"error": "secret-c"})) as get, \
                patch("lexrubric_inspect.preflight.requests.post") as post:
            with self.assertRaisesRegex(ProviderAccessError, "candidate anthropic.*401") as raised:
                check_provider_access(self.config)
        self.assertEqual(get.call_count, 1)
        self.assertNotIn("secret-c", str(raised.exception))
        post.assert_not_called()

    def test_anthropic_invalid_json_or_object_is_reported_without_contents(self):
        self.anthropic_candidate()
        bad_json = Mock(status_code=200, json=Mock(side_effect=ValueError("secret-c")))
        for response in (bad_json, self.response(200, ["secret-c"]), self.response(200, None)):
            with self.subTest(response=response), patch("lexrubric_inspect.preflight.requests.get", return_value=response), \
                    patch("lexrubric_inspect.preflight.requests.post") as post:
                with self.assertRaises(ProviderAccessError) as raised:
                    check_provider_access(self.config)
                self.assertNotIn("secret-c", str(raised.exception))
                post.assert_not_called()

    def test_anthropic_transport_failure_does_not_echo_request_secrets(self):
        self.anthropic_candidate()
        with patch("lexrubric_inspect.preflight.requests.get", side_effect=requests.Timeout("Bearer secret-c")) as get, \
                patch("lexrubric_inspect.preflight.requests.post") as post:
            with self.assertRaisesRegex(ProviderAccessError, "candidate anthropic.*Timeout") as raised:
                check_provider_access(self.config)
        self.assertEqual(get.call_count, 1)
        self.assertNotIn("secret-c", str(raised.exception))
        post.assert_not_called()

    def test_zai_presence_does_not_claim_authentication_or_model_access(self):
        self.zai_candidate()
        with patch("lexrubric_inspect.preflight.requests.get", side_effect=self.judge_responses()) as get, \
                patch("lexrubric_inspect.preflight.requests.post") as post:
            report = check_provider_access(self.config)
        self.assertEqual(get.call_count, 2)
        self.assertEqual([call.args[0] for call in get.call_args_list],
                         ["https://openrouter.ai/api/v1/key", "https://openrouter.ai/api/v1/models"])
        zai = report[0]
        self.assertEqual(zai["provider"], "zai")
        self.assertEqual(zai["model"], "glm-5.3")
        self.assertTrue(zai["credential_present"])
        self.assertIsNone(zai["authenticated"])
        self.assertIsNone(zai["model_available"])
        self.assertIn("approved pilot", zai["reason"])
        self.assertNotIn("secret", str(report))
        post.assert_not_called()

    def test_zai_missing_credential_stops_without_a_probe(self):
        self.zai_candidate()
        with patch.dict(os.environ, {"TEST_ZAI_KEY": ""}), patch("lexrubric_inspect.preflight.requests.get") as get, \
                patch("lexrubric_inspect.preflight.requests.post") as post:
            with self.assertRaisesRegex(ProviderAccessError, "Missing configured candidate credential"):
                check_provider_access(self.config)
        get.assert_not_called()
        post.assert_not_called()

    def test_zai_coding_plan_or_unrecognized_endpoint_is_not_silently_used(self):
        self.zai_candidate()
        self.config["candidate_chat_url"] = "https://api.z.ai/api/coding/paas/v4/chat/completions"
        with patch("lexrubric_inspect.preflight.requests.get") as get, patch("lexrubric_inspect.preflight.requests.post") as post:
            with self.assertRaisesRegex(ProviderAccessError, "standard API endpoint"):
                check_provider_access(self.config)
        get.assert_not_called()
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
