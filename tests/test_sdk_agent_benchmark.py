"""Concurrency evidence must distinguish nested prediction from dataset parallelism."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.benchmark_sdk_agent import max_overlap, docker_overlaps, compare_results


class SdkAgentBenchmarkTest(unittest.TestCase):
    def test_overlap_excludes_touching_intervals_and_detects_three_workers(self):
        self.assertEqual(max_overlap([{"start": 0, "end": 1}, {"start": 1, "end": 2}]), 1)
        self.assertEqual(max_overlap([{"start": 0, "end": 3}, {"start": 1, "end": 3}, {"start": 2, "end": 4}]), 3)

    def test_nested_prediction_is_not_counted_as_another_training_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = root / "parallel/A/run/host-1/container.jsonl"
            record.parent.mkdir(parents=True)
            launches = [{"command": kind, "argv": ["docker", "run", "--name", name]}
                        for kind, name in (("train", "gas-real-train-a"), ("predict", "gas-real-predict-a"))]
            record.write_text("\n".join(json.dumps(row) for row in launches), encoding="utf-8")
            events = []
            for identifier, name, begin, end in (("train-id", "gas-real-train-a", 1, 10), ("predict-id", "gas-real-predict-a", 3, 5)):
                for action, timestamp in (("start", begin), ("die", end)):
                    events.append({"Action": action, "Actor": {"ID": identifier, "Attributes": {"name": name}}, "time": timestamp})
            path = root / "events.jsonl"
            path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
            result = docker_overlaps(path, ["parallel"], root)["parallel"]
            self.assertEqual(result["max_overlapping_all_containers"], 2)
            self.assertEqual(result["max_overlapping_train_containers"], 1)
            self.assertEqual(result["container_start_die_intervals_captured"], 2)

    def test_comparison_rejects_missing_or_changed_predictor_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for mode in ("serial", "parallel"):
                for dataset in "ABC":
                    folder = root / mode / dataset / "run/measurement-1"
                    folder.mkdir(parents=True)
                    (folder / "predictions.jsonl").write_text('{"outputs":{"value":1}}\n', encoding="utf-8")
            self.assertTrue(all(r["model_and_predictions_byte_equal"] for r in compare_results(root / "serial", root / "parallel")))
            (root / "parallel/C/run/measurement-1/predictions.jsonl").write_text("different", encoding="utf-8")
            self.assertFalse(compare_results(root / "serial", root / "parallel")[-1]["model_and_predictions_byte_equal"])


if __name__ == "__main__":
    unittest.main()
