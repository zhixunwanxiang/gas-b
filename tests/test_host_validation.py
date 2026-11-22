"""Host admission must not be bypassed by local scores or model prose."""
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gas_a.host_validation import parent_reference, candidate_gate


class HostValidationTest(unittest.TestCase):
    def setUp(self):
        self.sha = "sha256:" + "a"*64
        self.before, self.after = {"macro_f1":.7}, {"macro_f1":.8}
        self.config = {"minimum_macro_f1_gain":.0005}

    def receipt(self, score, **changes):
        return {"metrics":{"macro_f1":score}, "scope":"authorized_public_validation",
                "validation_sha256":self.sha, **changes}

    def gate(self, candidate, parent=None):
        parent = parent_reference(parent or {},self.before,self.sha)
        return candidate_gate(candidate,parent,self.after,self.sha,self.config)

    def test_verified_same_scope_candidate_is_allowed(self):
        self.assertTrue(self.gate({"artifact_id":"new", "summary":self.receipt(.8)},self.receipt(.7))["eligible"])

    def test_host_regression_cannot_be_overridden_by_local_improvement(self):
        self.assertFalse(self.gate(self.receipt(0.),self.receipt(.7))["eligible"])

    def test_missing_or_different_digest_and_scope_are_rejected(self):
        for changes in ({"validation_sha256":None},{"validation_sha256":"other"},{"scope":"hidden"},{"scope":None},{"scope":[]}):
            with self.subTest(changes=changes):
                self.assertFalse(self.gate(self.receipt(.8,**changes))["eligible"])

    def test_invalid_scores_and_unrecognized_schema_are_rejected(self):
        for value in (True,float("nan"),float("inf"),-1,101,10**1000,None):
            with self.subTest(score=value):
                self.assertFalse(self.gate(self.receipt(value))["eligible"])
        self.assertFalse(self.gate({"score":.8})["eligible"])

    def test_local_disagreement_is_not_silently_accepted(self):
        self.assertFalse(self.gate(self.receipt(.9))["eligible"])
        self.assertFalse(self.gate(self.receipt(.8),self.receipt(.6))["eligible"])

    def test_rounding_is_allowed_and_missing_parent_source_is_explicit(self):
        result = self.gate(self.receipt(.8000001))
        self.assertTrue(result["eligible"])
        self.assertEqual(result["parent"]["source"],"local_parent_host_measurement_absent")

    def test_present_but_unverifiable_parent_is_not_replaced_by_local_score(self):
        self.assertFalse(self.gate(self.receipt(.8),self.receipt(.7,validation_sha256=None))["eligible"])


if __name__ == "__main__":
    unittest.main()
