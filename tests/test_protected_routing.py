"""A high-F1 route must not displace a route that protects Ethanol recall."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from tests.test_submission import ROOT
from gas_a.frozen_router import calibrate_policy
from gas_a.sensor_training.data import score_predictions
from gas_a.evolution import promotion_gate


class ProtectedRoutingTest(unittest.TestCase):
    def test_feasible_lower_weight_beats_higher_f1_with_excess_ethanol_loss(self):
        y = np.array([0]*100 + [3]*200 + [1, 2, 4, 5]*20)
        four = np.eye(6)[y].astype(float)
        four[:100] = 0
        four[:100, 0], four[:100, 3] = .6, .4
        four[100:200] = 0
        four[100:200, 0], four[100:200, 3] = .52, .48
        four[200:300] = 0
        four[200:300, 0], four[200:300, 3] = .9, .1
        challenger = np.eye(6)[y].astype(float)
        challenger[:30] = 0
        challenger[:30, 0], challenger[:30, 3] = .4, .6
        challenger[100:200] = 0
        challenger[100:200, 0], challenger[100:200, 3] = .1, .9
        challenger[200:300] = 0
        challenger[200:300, 0], challenger[200:300, 3] = .3, .7
        models = {"four_sensor": SimpleNamespace(predict=lambda x: four)}
        validation = {"x": np.zeros((len(y), 128)), "y": y, "sha256": "fixture"}
        search = {"inner_weights": [.75], "outer_weights": [.25, .8], "correction_choices": [False],
                  "minimum_support": 20, "minimum_macro_f1_gain": .0005}
        controller = {"minimum_macro_f1_gain": .0005, "minimum_class_support": 20, "maximum_class_recall_drop": .2}
        before = score_predictions(four, y)
        scores = (four, challenger, challenger, np.zeros(len(y), dtype=bool))
        with patch("gas_a.frozen_router._expert_scores", return_value=scores), \
                patch("gas_a.frozen_router._new_stream", return_value={}):
            raw, raw_report = calibrate_policy(models, validation, {}, search)
            protected, report = calibrate_policy(models, validation, {}, search,
                                                  controller=controller, parent_metrics=before)
        self.assertEqual(raw["outer_weight"], .8)
        self.assertEqual(protected["outer_weight"], .25)
        self.assertGreater(raw_report["selected_metrics"]["macro_f1"], report["selected_metrics"]["macro_f1"])
        self.assertFalse(promotion_gate(before, raw_report["selected_metrics"], controller)["eligible"])
        self.assertTrue(report["selected_gate"]["eligible"])
        self.assertEqual(report["selected_metrics"]["per_class"]["Ethanol"]["recall"], 1.)
        self.assertTrue(any("Ethanol" in t["gate"]["class_recall_losses"] for t in report["trials"]))
