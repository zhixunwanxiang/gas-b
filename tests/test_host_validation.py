"""Host admission must not be bypassed by local scores or model prose."""
import unittest
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gas_a.host_validation import (parent_reference, candidate_gate, CandidateValidationBinding,
                                  bind_candidate_validation, measurement)
from gas_a.sensor_training.io import digest


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

    def binding(self):
        return CandidateValidationBinding("new",self.sha,"A",1,"macro_f1")

    def bound(self, summary, **fields):
        receipt={"type":"candidate_response","artifact_id":"new","summary":summary,**fields}
        parent=parent_reference({},self.before,self.sha)
        return candidate_gate(receipt,parent,self.after,self.sha,self.config,binding=self.binding())

    def test_compact_metric_layouts_require_live_submission_binding(self):
        for summary in ({"metrics":{"macro_f1":.8}}, {"macro_f1":.8},
                        {"metric":"macro_f1","value":.8}, {"metric":"macro_f1","score":.8}):
            with self.subTest(summary=summary):
                self.assertFalse(self.gate({"summary":summary})["eligible"])
                result=self.bound(summary)
                self.assertTrue(result["eligible"])
                self.assertEqual(result["candidate"]["macro_f1"],.8)
                self.assertEqual(result["candidate"]["validation_sha256_source"],"locally_hashed_sdk_submission_projection")

    def test_bound_submission_never_overrides_explicit_conflicting_metadata(self):
        for changes in ({"scope":"hidden"},{"scope":None},{"validation_sha256":None},
                        {"validation_sha256":"wrong"},{"artifact_id":"other"},{"dataset_key":"B"}):
            with self.subTest(changes=changes):
                self.assertFalse(self.bound({"macro_f1":.8,**changes})["eligible"])
        self.assertFalse(self.bound({"macro_f1":.8},scope="hidden")["eligible"])
        self.assertFalse(self.bound({"macro_f1":.8},artifact_id="other")["eligible"])

    def test_binding_never_invents_or_replaces_host_measurement(self):
        for summary in ({"score":.8}, {"metrics":{}}, {"macro_f1":0.}, {"macro_f1":.9},
                        {"macro_f1":True}, {"macro_f1":float("nan")}, {"macro_f1":10**1000},
                        {"metrics":{"macro_f1":.8},"macro_f1":.9}, {"metric":"rmse","score":.8}):
            with self.subTest(summary=summary):
                self.assertFalse(self.bound(summary)["eligible"])
        value={"type":"candidate_response","artifact_id":"new","summary":{"macro_f1":.8}}
        self.assertFalse(measurement(value,self.sha,binding={"artifact_id":"new"})["comparable"])
        self.assertFalse(measurement(value,"sha256:other",binding=self.binding())["comparable"])

    def test_submission_binding_checks_current_file_and_sdk_response_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"validation.jsonl"
            path.write_text("authorized public projection\n",encoding="utf-8")
            sha=digest(path)
            rnd=SimpleNamespace(dataset_key="A",round_index=1,
                frame={"protocol":"contest.harness.workspace.v2"},task={"task":"classification"},
                data=SimpleNamespace(validation_path=path))
            receipt={"type":"candidate_response","artifact_id":"new"}
            self.assertEqual(bind_candidate_validation(rnd,receipt,sha).validation_sha256,sha)
            path.write_text("changed projection\n",encoding="utf-8")
            with self.assertRaises(ValueError): bind_candidate_validation(rnd,receipt,sha)
            with self.assertRaises(ValueError): bind_candidate_validation(rnd,{"artifact_id":"new"},digest(path))


if __name__ == "__main__":
    unittest.main()
