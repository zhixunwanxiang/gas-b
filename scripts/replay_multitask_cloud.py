"""用已保存的真实模型回复重放A/B/C数值训练；不读取密钥，不调用模型，不写云端。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import contest_submission
from tests.test_submission import OfflineHost
from tests.test_regression import RegressionHost
from scripts.test_multitask_cloud import LOCAL_PROMPT, MODEL
from scripts.test_cloud_suite import DEFAULT_IMAGE, DEFAULT_IMAGE_ID
from scripts.docker_rehearsal import DockerRuntime
from scripts.validate_multitask import run_dataset
from scripts.verify_multitask_cloud_run import read, rows
from gas_a.sensor_training.io import digest, write_json, append_jsonl


class RecordedReplies:
    def __init__(self, source):
        self.responses = [read(p / "response.json") for p in sorted((source / "normal/provider-calls").iterdir())]
        self.used = 0
        self.verified, self.attempts = [], []  # 没有任何新增提供方请求；不能把重放冒充真实调用。

    def next(self):
        # 按原调用顺序消费真实响应；一旦耗尽就失败，绝不回退到联网模型或离线规则补齐。
        if self.used >= len(self.responses):
            raise ValueError("Recorded response sequence exhausted; no model fallback is available")
        response = self.responses[self.used]
        self.used += 1
        return response["id"], response


class ReplayMixin:
    def write(self, text):
        frame = json.loads(text)
        if frame["type"] != "model_request":
            return super().write(text)
        if frame["id"] != self.next_sequence:
            raise ValueError("Unexpected replay RPC sequence")
        self.next_sequence += 1
        self.frames.append(frame)
        self.messages.append(frame["messages"])
        call_id, response = self.gateway.next()
        # 保留原回执号以对应证据，但明确标记 real_remote_model_call=False，避免把重放当新增实验。
        self.calls.append(call_id)
        append_jsonl(self.store / "model-calls.jsonl", {"call_id": call_id,
            "real_remote_model_call": False, "source": "recorded_original_provider_response", "response": response})
        self.pending.append({"type": "model_response", "id": frame["id"], "call_id": call_id, "response": response})
        return len(text)


class ReplayClassificationPeer(ReplayMixin, OfflineHost):
    pass


class ReplayRegressionPeer(ReplayMixin, RegressionHost):
    pass


class ReplayFactory:
    def __init__(self, source):
        self.gateway = RecordedReplies(source)

    def __call__(self, root, parent, dataset, index, history, state):
        host = (ReplayClassificationPeer(root, parent, index, history=history, state=state) if dataset == "A" else
                ReplayRegressionPeer(root, parent, dataset, index, history=history, state=state))
        host.gateway = self.gateway
        host.frame.update(model_name=MODEL, prompt=LOCAL_PROMPT, prompt_version="local-abc-real-sdk104-v1")
        return host


def compare(source, replay, original, report):
    # 验证的不只是最终分数：每轮选版、候选提交行为、检查点/路由文件和全部逐条预测都要一致。
    # 完整输入证据在本地保留；本重放证明的是已保存模型回复下的数值可复现性。
    if len(original["rounds"]) != len(report["rounds"]):
        raise ValueError("Replay round count differs from original")
    files_checked = 0
    for old, new in zip(original["rounds"], report["rounds"]):
        if old["selected_artifact_id"] != new["selected_artifact_id"]:
            raise ValueError("Replay selected a different artifact")
        index = old["round"]
        if old["candidate_submitted"] != new["candidate_submitted"]:
            raise ValueError("Replay candidate submission differs")
        if not old["candidate_submitted"]:
            continue
        transcript = rows(source / f"normal/host-{index}/protocol.jsonl")
        artifact = next(r["frame"]["artifact"] for r in transcript if r["frame"].get("type") == "candidate_request")
        names = list(artifact["checkpoint_paths"])
        if (source / f"normal/host-{index}/candidate/routing.json").is_file():
            names.append("routing.json")
        for name in names:
            if digest(source / f"normal/host-{index}/candidate" / name) != digest(replay / f"host-{index}/candidate" / name):
                raise ValueError("Replay checkpoint/route bytes differ: " + name)
            files_checked += 1
    predictions_checked = 0
    old_measurements = sorted((source / "normal").glob("measurement-*"))
    new_measurements = sorted(replay.glob("measurement-*"))
    if len(old_measurements) != len(new_measurements):
        raise ValueError("Replay measurement count differs")
    for old, new in zip(old_measurements, new_measurements):
        if digest(old / "predictions.jsonl") != digest(new / "predictions.jsonl"):
            raise ValueError("Replay predictions differ from recorded run")
        predictions_checked += 1
    return {"checkpoint_and_route_files_byte_equal": files_checked,
            "prediction_groups_byte_equal": predictions_checked, "selected_artifacts_equal": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--expected-image-id", default=DEFAULT_IMAGE_ID)
    parser.add_argument("--docker-executable", default="docker")
    parser.add_argument("--docker-context", default="desktop-linux")
    parser.add_argument("--datasets", nargs="+", choices=("A", "B", "C"), default=["A", "B", "C"])
    args = parser.parse_args()
    args.output.resolve().mkdir(parents=True, exist_ok=False)
    runtime = DockerRuntime(args.image, executable=args.docker_executable, context=args.docker_context,
                            expected_id=args.expected_image_id, required_tasks=("classification", "regression"))
    write_json(args.output.resolve() / "image-preflight.json", runtime.preflight())
    results = []
    for key in args.datasets:
        source = args.source.resolve() / key
        original = read(source / "normal/report.json")
        factory = ReplayFactory(source)
        output = args.output.resolve() / key
        report = run_dataset(source / "official-baseline", output, key, len(original["rounds"]), runtime, peer_factory=factory)
        report["scope"] = "recorded_reply_numerical_replay_in_docker"
        report["new_real_model_calls"] = 0
        write_json(output / "report.json", report)
        if factory.gateway.used != len(factory.gateway.responses):
            raise ValueError("Not all original responses were consumed")
        result = {"dataset": key, "verified": True, "new_real_model_calls": 0,
                  "recorded_replies_used": factory.gateway.used, **compare(source, output, original, report)}
        results.append(result)
        write_json(args.output.resolve() / "verification.json", {"verified": True, "datasets": results,
                    "new_real_model_calls": 0, "cloud_writes": 0})
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
