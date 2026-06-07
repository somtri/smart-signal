import json
import tempfile
import unittest
from pathlib import Path

from smartsignal.config import ModelConfig
from smartsignal.data import generate_demo_data
from smartsignal.pipeline import ForecastPipeline


class PipelineTests(unittest.TestCase):
    def test_pipeline_produces_reproducible_artifacts(self) -> None:
        config = ModelConfig(
            n_estimators=40,
            max_depth=6,
            min_samples_leaf=5,
            validation_folds=3,
            min_train_size=120,
            n_jobs=1,
        )
        data = generate_demo_data(periods=550, seed=42)
        with tempfile.TemporaryDirectory() as directory:
            result = ForecastPipeline(config).run(
                data,
                ticker="TEST",
                data_source="unit test",
                artifact_dir=directory,
            )
            output = Path(directory)
            self.assertTrue((output / "model.joblib").exists())
            self.assertTrue((output / "predictions.csv").exists())
            self.assertTrue((output / "latest_signal.json").exists())
            self.assertGreater(len(result.predictions), 50)
            with (output / "metrics.json").open(encoding="utf-8") as handle:
                metrics = json.load(handle)
            self.assertEqual(metrics["ticker"], "TEST")
            self.assertIn(metrics["latest_signal"]["direction"], {"UP", "DOWN"})
            self.assertIsNotNone(metrics["sentiment_ablation"])


if __name__ == "__main__":
    unittest.main()

