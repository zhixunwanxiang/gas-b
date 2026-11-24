"""Protocol mutation tests for the read-only SDK timing observer."""
import base64
from copy import deepcopy
import hashlib
import unittest

from scripts.sdk_protocol_audit import audit_events


def conversation():
    start = {"protocol": "contest.harness.workspace.v2", "prompt": "Offline test", "prompt_version": "test",
             "dataset_key": "C", "round_index": 1, "task_contract": {"task": "regression"},
             "model_name": "offline", "max_output_tokens": 32, "parent_checkpoint_sha256": "sha256:" + "0" * 64,
             "parent_artifact": {"artifact_id": "initial", "artifact": {"checkpoint_paths": ["checkpoint.npz"]}},
             "artifacts": []}
    return [
        {"direction": "host", "elapsed_seconds": .1, "frame": start},
        {"direction": "harness", "elapsed_seconds": .2, "frame": {"type": "history_request", "id": 0}},
        {"direction": "host", "elapsed_seconds": .5, "frame": {"type": "history_response", "id": 0, "records": []}},
        {"direction": "harness", "elapsed_seconds": .6, "frame": {"type": "model_request", "id": 1,
            "messages": [{"role": "user", "content": "Retain parent"}], "max_tokens": 32}},
        {"direction": "host", "elapsed_seconds": 1.6, "frame": {"type": "model_response", "id": 1,
            "call_id": "offline-1", "response": {"choices": [{"message": {"content": "retain parent"}}]}}},
        {"direction": "harness", "elapsed_seconds": 1.7, "frame": {"type": "round_complete", "record": {
            "selected_artifact_id": "initial", "model_call_ids": ["offline-1"], "history_ids": []}}},
        {"direction": "host", "elapsed_seconds": 1.8, "frame": {"type": "round_ack", "accepted": True}},
    ]


class SdkProtocolAuditTest(unittest.TestCase):
    def test_minimum_sdk_round_and_host_vs_participant_time(self):
        result = audit_events(conversation())
        self.assertTrue(result["order_valid"], result["errors"])
        self.assertEqual(result["round_trips"], 3)
        self.assertAlmostEqual(result["host_wait_seconds_by_request"]["model_request"], 1)
        self.assertAlmostEqual(result["participant_gap_seconds_total"], .3)

    def test_wrong_id_type_or_two_outstanding_requests_are_rejected(self):
        for change in ("id", "type", "overlap"):
            events = conversation()
            if change == "id":
                events[2]["frame"]["id"] = 2
            elif change == "type":
                events[2]["frame"]["type"] = "model_response"
            else:
                events.insert(2, deepcopy(events[3]))
            self.assertFalse(audit_events(events)["order_valid"], change)

    def test_missing_ack_frames_after_ack_and_unverified_calls_are_rejected(self):
        for events in (conversation()[:-1], conversation() + [deepcopy(conversation()[1])]):
            self.assertFalse(audit_events(events)["order_valid"])
        events = conversation()
        events[-2]["frame"]["record"]["model_call_ids"] = ["invented-call"]
        self.assertFalse(audit_events(events)["order_valid"])

    def test_upload_offsets_digest_and_finish_before_transaction(self):
        raw = b"hello"
        requests = [
            ({"type": "upload_begin_request", "kind": "candidate", "files": [
                {"path": "checkpoint.npz", "size_bytes": len(raw), "sha256": "sha256:" + hashlib.sha256(raw).hexdigest()}]},
             {"type": "upload_begin_response", "upload_id": "upload-1"}),
            ({"type": "upload_chunk_request", "upload_id": "upload-1", "path": "checkpoint.npz", "offset": 0,
              "data_base64": base64.b64encode(raw).decode()}, {"type": "upload_chunk_response", "offset": 5}),
            ({"type": "upload_finish_request", "upload_id": "upload-1"},
             {"type": "upload_finish_response", "upload_id": "upload-1"}),
            ({"type": "candidate_request", "upload_id": "upload-1"},
             {"type": "candidate_response", "artifact_id": "candidate-1"}),
        ]
        events = conversation()[:5]
        for identifier, (request, reply) in enumerate(requests, 2):
            for direction, frame in (("harness", request), ("host", reply)):
                frame["id"] = identifier
                events.append({"direction": direction, "elapsed_seconds": 1.7 + .1 * (len(events) - 5), "frame": frame})
        finish = deepcopy(conversation()[-2:])
        finish[0]["elapsed_seconds"], finish[1]["elapsed_seconds"] = 3.0, 3.1
        finish[0]["frame"]["record"]["selected_artifact_id"] = "candidate-1"
        events += finish
        self.assertTrue(audit_events(events)["order_valid"], audit_events(events)["errors"])
        for index, field, bad in ((7, "offset", 1), (8, "offset", 4), (5, "files", [])):
            changed = deepcopy(events)
            changed[index]["frame"][field] = bad
            self.assertFalse(audit_events(changed)["order_valid"])
        changed = deepcopy(events)
        changed[9]["frame"] = {"type": "candidate_request", "id": 4, "upload_id": "upload-1"}
        self.assertFalse(audit_events(changed)["order_valid"])

    def test_optional_history_is_not_invented_as_mandatory(self):
        events = conversation()
        del events[1:3]
        events[1]["frame"]["id"] = events[2]["frame"]["id"] = 0
        self.assertTrue(audit_events(events)["order_valid"])


if __name__ == "__main__":
    unittest.main()
