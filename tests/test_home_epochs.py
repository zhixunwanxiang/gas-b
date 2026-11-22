"""Home epoch configuration changes actual training while preserving parent state."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import contest_submission  # Configure the same source path and CPU threads as the entry.
import numpy as np
from gas_a.home_training import train_home
from gas_a.model import Model as HomeModel
from gas_a.decisions import training_parts
from gas_a.sensor_training.data import load_pair
from gas_a.continuation import continue_candidate


class HomeEpochsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "submission_config.json").read_text(encoding="utf-8"))
        cls.spec = cls.config["experiments"][0]["routing"]["home"]
        cls.train, cls.validation = load_pair(ROOT / "data/public/train.jsonl", ROOT / "data/public/validation.jsonl")

    def test_multiple_epochs_update_head_and_counters_but_preserve_hidden_adam_and_scaler(self):
        parent = HomeModel.load(ROOT / "models/reference.npz")
        before = deepcopy(parent)
        with tempfile.TemporaryDirectory(prefix="home-epochs-") as temporary:
            dest = Path(temporary)
            report = train_home(parent, self.train, {**self.spec, "epochs": 3}, dest)
            fit, _ = training_parts(self.train)
            batches = (len(fit["y"]) + parent.meta["batch_size"] - 1) // parent.meta["batch_size"]
            for name in ("home_trunk.npz", "home_router.npz"):
                child = HomeModel.load(dest / name)
                self.assertEqual(child.meta["epochs"], parent.meta["epochs"] + 3)
                self.assertEqual(child.meta["steps"], parent.meta["steps"] + 3 * batches)
                for group in ("params", "first", "second"):
                    for key in ("w1", "b1"):
                        np.testing.assert_array_equal(getattr(child, group)[key], getattr(parent, group)[key])
                    self.assertFalse(np.array_equal(getattr(child, group)["w2"], getattr(parent, group)["w2"]))
                np.testing.assert_array_equal(child.mean, parent.mean)
                np.testing.assert_array_equal(child.scale, parent.scale)
            self.assertEqual(report["epochs_requested"], 3)
            self.assertEqual(report["epochs_executed"], 3)
            self.assertEqual(len(report["epoch_history"]), 3)
        self.assertEqual(parent.meta, before.meta)
        for key in parent.params:
            np.testing.assert_array_equal(parent.params[key], before.params[key])

    def test_invalid_epoch_count_and_expired_budget_do_not_write_artifacts(self):
        parent = HomeModel.load(ROOT / "models/reference.npz")
        with tempfile.TemporaryDirectory(prefix="home-budget-") as temporary:
            dest = Path(temporary)
            for value in (0, -1, True, 1.5, 4):
                with self.subTest(epochs=value), self.assertRaises(ValueError):
                    train_home(parent, self.train, {**self.spec, "epochs": value}, dest, max_epochs=3)
            with self.assertRaises(TimeoutError):
                train_home(parent, self.train, self.spec, dest, deadline=time.monotonic() - 1)
            self.assertEqual(list(dest.iterdir()), [])

    def test_routed_candidate_passes_home_epoch_setting_and_records_execution(self):
        spec = deepcopy(self.config["experiments"][0])
        spec["epochs"], spec["minimum_epochs"] = 1, 1
        spec["routing"]["five_sensor"].update(epochs=1, minimum_epochs=1)
        spec["routing"]["home"]["epochs"] = 3
        with tempfile.TemporaryDirectory(prefix="home-route-") as temporary:
            dest = Path(temporary) / "candidate"
            continue_candidate(ROOT / "models/reference.npz", self.train, self.validation, dest,
                               spec, {"weakest_class": "Acetaldehyde"}, self.config["limits"])
            home = json.loads((dest / "routing_training.json").read_text(encoding="utf-8"))["home"]
            self.assertEqual(home["training"]["epochs_executed"], 3)


if __name__ == "__main__":
    unittest.main()
