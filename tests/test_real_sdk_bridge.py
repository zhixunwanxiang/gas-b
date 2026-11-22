"""Offline bridge checks: never send HTTP requests or bill a model."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import test_real_sdk as live


class RealSdkBridgeTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="real-sdk-bridge-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = {"model":live.MODEL, "base_url":"https://pytrio.com"}
        self.gateway = live.RealGateway(self.config, "private-test-key", self.root, max_calls=2)
        self.start = {"dataset_key":"A", "round_index":1, "task_contract":{"task":"classification"},
                      "prompt_version":"local-test-v1", "prompt":"FIXED_HOST_PROMPT"}
        self.frame = {"type":"model_request", "id":0,
                      "messages":[{"role":"user", "content":json.dumps({"phase":"plan"})}],
                      "max_tokens":512, "temperature":0, "enable_thinking":False}
        self.response = {"id":"provider-id-1", "model":live.MODEL,
                         "choices":[{"message":{"content":"{\"action\":\"observe\"}"}}],
                         "usage":{"total_tokens":20}}
        self.http = self.enterContext(patch.object(live, "request_json", return_value=self.response))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def test_preserves_provider_response_id_options_and_adds_prompt_once(self):
        identifier, response, bound = self.gateway.invoke(self.frame,self.start)
        self.assertEqual(identifier,"provider-id-1")
        self.assertIs(response,self.response)
        self.assertEqual(sum(m["content"].count("FIXED_HOST_PROMPT") for m in bound),1)
        payload = self.http.call_args.args[2]
        self.assertEqual(payload["max_tokens"],512)
        self.assertEqual(payload["temperature"],0)
        self.assertIs(payload["enable_thinking"],False)
        self.assertEqual(payload["model"],live.MODEL)
        self.assertEqual(self.frame["messages"][0]["role"],"user")
        for p in self.root.rglob("*.json"):
            self.assertNotIn("private-test-key",p.read_text(encoding="utf-8"))

    def test_existing_system_message_receives_one_host_prefix(self):
        self.frame["messages"].insert(0,{"role":"system","content":"Participant instructions"})
        _,_,bound = self.gateway.invoke(self.frame,self.start)
        self.assertEqual([m["role"] for m in bound],["system","user"])
        self.assertIn("Participant instructions",bound[0]["content"])
        self.assertEqual(self.gateway.verified[0]["phase"],"plan")

    def test_unknown_request_outcome_is_not_retried(self):
        self.http.side_effect = TimeoutError("private-test-key")
        with self.assertRaises(TimeoutError):
            self.gateway.invoke(self.frame,self.start)
        self.http.assert_called_once()
        self.assertEqual(self.gateway.attempts[0]["status"],"outcome_unknown")
        self.assertEqual(self.gateway.verified,[])
        self.assertNotIn("private-test-key",(self.root/"provider-calls/call-001/receipt.json").read_text())

    def test_wrong_model_is_not_accepted_or_substituted(self):
        self.http.return_value = {**self.response,"model":"Qwen/Qwen3-4B"}
        with self.assertRaises(ValueError):
            self.gateway.invoke(self.frame,self.start)
        self.http.assert_called_once()
        self.assertEqual(self.gateway.attempts[0]["status"],"response_unverified")
        self.assertEqual(self.gateway.verified,[])

    def test_call_limit_prevents_additional_http_requests(self):
        self.gateway.max_calls = 1
        self.gateway.invoke(self.frame,self.start)
        with self.assertRaisesRegex(RuntimeError,"budget exhausted"):
            self.gateway.invoke(self.frame,self.start)
        self.http.assert_called_once()

    def test_model_branch_never_calls_the_offline_rule_engine(self):
        parent = self.root/"checkpoint.npz"
        parent.write_bytes(b"identity-only-fixture")
        gateway = Mock()
        gateway.invoke.return_value = ("actual-provider-id",self.response,self.frame["messages"])
        host = live.RealModelHost(self.root,parent,measurements=Mock(),gateway=gateway)
        host.readline()  # Consume start before the first RPC.
        with patch.object(live.EvidenceHost,"plan",side_effect=AssertionError("Offline planning forbidden")), \
                patch.object(live.EvidenceHost,"selection",side_effect=AssertionError("Offline selection forbidden")):
            host.write(json.dumps(self.frame))
        reply = json.loads(host.readline())
        self.assertEqual(reply["type"],"model_response")
        self.assertEqual(reply["call_id"],"actual-provider-id")
        self.assertEqual(reply["response"],self.response)
        self.assertEqual(host.calls,["actual-provider-id"])
        gateway.invoke.assert_called_once()


if __name__ == "__main__":
    unittest.main()
