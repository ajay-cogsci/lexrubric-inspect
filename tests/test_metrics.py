"""Exercise every generated dimension metric through Inspect's epoch reducer."""

from functools import partial
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelAPI
from inspect_ai.scorer import Score
from pytest_inspect_evals.metrics import assert_agreeing_epochs_change_nothing, run_metrics

from lexrubric_inspect.lexrubric_port import DIMENSIONS, DIMENSION_METRICS


def dimension_score(value):
    return Score(value=0.0 if value is None else value,
                 metadata={"dimension_rates": {name: value for name in DIMENSIONS.values()}})


class DimensionMetricTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        for target, value in (
            ("inspect_ai._util.appdirs.user_data_path", root / "data"),
            ("inspect_ai._util.appdirs.user_cache_path", root / "cache"),
        ):
            guard = patch(target, return_value=value)
            guard.start()
            self.addCleanup(guard.stop)
        for guard in (
            patch("pytest_inspect_evals.metrics.inspect_eval",
                  new=partial(inspect_eval, ctl_server=False)),
            # MockLLM's default usage estimate otherwise downloads a tokenizer.
            # Token accounting is unrelated to these score-reduction checks.
            patch.object(ModelAPI, "count_tokens", new=AsyncMock(return_value=0)),
            patch.object(ModelAPI, "count_text_tokens", new=AsyncMock(return_value=0)),
            patch("requests.sessions.Session.request", side_effect=AssertionError("Unexpected HTTP")),
            patch.object(socket.socket, "connect", side_effect=AssertionError("Unexpected network")),
            patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Unexpected network")),
        ):
            guard.start()
            self.addCleanup(guard.stop)
        self.metrics = [factory() for factory in DIMENSION_METRICS]

    def test_equal_epochs_preserve_each_dimension_rate(self):
        assert_agreeing_epochs_change_nothing(
            self.metrics, [dimension_score(value) for value in (-20.0, 0.0, 100.0, None)])

    def test_one_epoch_preserves_penalties_zero_and_undefined_denominators(self):
        # The benchmark uses one epoch. Undefined dimensions are excluded while
        # zero and negative dimension rates contribute to the question mean.
        results = run_metrics(
            self.metrics, [[dimension_score(value)] for value in (-20.0, 0.0, 100.0, None)])
        self.assertEqual(len(results), len(DIMENSIONS))
        self.assertEqual({name.rsplit("/", 1)[-1] for name in results},
                         {f"lexrubric_{name}" for name in DIMENSIONS})
        for value in results.values():
            self.assertAlmostEqual(value, 80.0 / 3)


if __name__ == "__main__":
    unittest.main()
