"""Exercise the public task with Inspect's mock model and the original judge."""

import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, ModelUsage, get_model

from lexrubric_inspect import lexrubric, verify_pipeline


class EndToEndTests(unittest.TestCase):
    def test_mock_model_reaches_original_judge_and_paper_scores(self):
        task = lexrubric()
        records, _ = verify_pipeline.check_corpus(task.dataset)
        transport = verify_pipeline.MockTransport(records)
        by_question = {records[sample_id]["input"]: sample_id
                       for sample_id in verify_pipeline.SAMPLE_IDS}
        generated = []

        def candidate_response(messages, tools, tool_choice, config):
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0].role, "user")
            self.assertEqual(tools, [])
            self.assertEqual(config.temperature, 0.6)
            self.assertEqual(config.max_tokens, 16384)
            sample_id = by_question[messages[0].content]
            generated.append(sample_id)
            output = ModelOutput.from_content(
                model="mockllm/model", content=transport.fixture[sample_id]["candidate_raw"])
            output.usage = ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)
            return output

        model = get_model("mockllm/model", custom_outputs=candidate_response)
        env = {
            "LEXRUBRIC_JUDGE_CHAT_URL": verify_pipeline.JUDGE_URL,
            "LEXRUBRIC_JUDGE_API_KEY": verify_pipeline.SYNTHETIC_KEY,
            "LEXRUBRIC_JUDGE_API_KEY_ENV": "",
            "LEXRUBRIC_JUDGE_PROVIDER": "upstream",
            "LEXRUBRIC_SCORE_CACHE": "",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (patch.dict(os.environ, env),
                  patch("inspect_ai._util.appdirs.user_data_path", return_value=root / "data"),
                  patch("inspect_ai._util.appdirs.user_cache_path", return_value=root / "cache"),
                  patch("requests.post", side_effect=transport.post),
                  patch("requests.sessions.Session.request", side_effect=transport.blocked_network),
                  patch.object(socket.socket, "connect", side_effect=transport.blocked_network),
                  patch.object(socket.socket, "connect_ex", side_effect=transport.blocked_network)):
                [log] = inspect_eval(
                    task, model=model, sample_id=verify_pipeline.SAMPLE_IDS,
                    display="none", ctl_server=False, log_dir=str(root / "logs"))
            samples = list(log.samples or [])

        self.assertEqual(log.status, "success", log.error.message if log.error else None)
        self.assertCountEqual(generated, verify_pipeline.SAMPLE_IDS)
        self.assertEqual(len(samples), 2)
        self.assertEqual(transport.network_attempts, [])
        self.assertEqual(len(transport.captured), 36)
        self.assertTrue(all(row["role"] == "judge" for row in transport.captured))
        for sample in samples:
            result = next(iter(sample.scores.values()))
            rubrics = records[sample.id]["metadata"]["rubrics"]
            points, denominator, rate, dimensions = verify_pipeline.expected_rates(rubrics)
            self.assertAlmostEqual(result.value, rate)
            self.assertEqual(result.metadata["raw_points"], points)
            self.assertEqual(result.metadata["positive_points"], denominator)
            self.assertEqual(result.metadata["dimension_rates"], dimensions)
            self.assertEqual(result.metadata["grader_errors"], 0)
            self.assertEqual(result.answer, transport.fixture[sample.id]["filtered"])


if __name__ == "__main__":
    unittest.main()
