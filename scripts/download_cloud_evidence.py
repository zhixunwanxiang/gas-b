"""Download and check an experiment's evidence ZIP; no model calls or cloud writes."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.cloud_experiment import CloudJournal, media_items
from scripts.live_platforms import SWANLAB_HOST, check_run, load_env


def download_bundle(remote, output):
    """Validate all chunk metadata and hashes before creating a new local file."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("Output already exists; choose a new file")
    payload = remote.medias(keys=["evidence/bundle_manifest", "evidence/bundle_chunk"], all=True)
    manifests, blocks = [], []
    for key, _, item in media_items(payload):
        url = item.get("url", "")
        if not url.startswith("https://"):
            raise ValueError("Cloud media must have an HTTPS download URL")
        # Signed media URLs carry their own authorization. No API key is sent here.
        with urllib.request.urlopen(url, timeout=45) as response:
            value = json.load(response)
        if key == "evidence/bundle_manifest":
            manifests.append(value)
        elif key == "evidence/bundle_chunk":
            blocks.append(value)
    if len(manifests) != 1:
        raise ValueError("Expected exactly one evidence manifest")
    manifest, chunks = manifests[0], {}
    if manifest.get("format") != "base64-zip-chunks-v1" or manifest["chunks"] != len(blocks):
        raise ValueError("Evidence format or chunk count is invalid")
    for value in blocks:
        index = value["index"]
        if (type(index) is not int or index in chunks or not 0 <= index < manifest["chunks"]
                or value["count"] != manifest["chunks"] or value["bundle_sha256"] != manifest["sha256"]):
            raise ValueError("Evidence chunk identity is invalid")
        data = base64.b64decode(value["data_base64"], validate=True)
        if hashlib.sha256(data).hexdigest() != value["chunk_sha256"]:
            raise ValueError("Evidence chunk digest mismatch")
        chunks[index] = data
    bundle = b"".join(chunks[i] for i in range(manifest["chunks"]))
    if len(bundle) != manifest["bytes"] or hashlib.sha256(bundle).hexdigest() != manifest["sha256"]:
        raise ValueError("Complete evidence ZIP digest mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(bundle)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", help="workspace/project/run_id; download from cloud only")
    mode.add_argument("--verify-local-run", type=Path, help="Recheck texts, scalars and ZIP against local event journal")
    parser.add_argument("--output", type=Path, help="New ZIP path, required with --run")
    args = parser.parse_args()
    if args.run and args.output is None:
        parser.error("--output is required with --run")
    if args.verify_local_run and args.output is not None:
        parser.error("--output only applies to --run")
    load_env(args.env_file)
    import swanlab
    if args.verify_local_run:
        result = CloudJournal.resume_verification(swanlab, args.verify_local_run.resolve()).verify_cloud()
    else:
        api = swanlab.Api(api_key=os.environ["SWANLAB_API_KEY"], host=SWANLAB_HOST)
        state = check_run(api, args.run)
        if not state["terminal"]:
            raise ValueError("Experiment is still running; evidence may be incomplete")
        manifest = download_bundle(api.run(state["run_path"]), args.output)
        result = dict(state, evidence_zip=manifest, output=str(args.output.resolve()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
