"""Read-only SDK 1.0.4 wire-order and timing audit; never sends protocol frames."""
import base64
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from contest_harness.errors import HarnessError
from contest_harness.round import _MODEL_OPTIONS, _validate_start, _declaration, _text


RESPONSES = {
    "history_request": "history_response", "model_request": "model_response",
    "upload_begin_request": "upload_begin_response", "upload_chunk_request": "upload_chunk_response",
    "upload_finish_request": "upload_finish_response", "candidate_request": "candidate_response",
    "state_request": "state_response", "artifact_read_request": "artifact_read_response",
    "round_complete": "round_ack",
}


def audit_events(events):
    """Validate observed order against the shipped SDK, without inventing an API order.

    save_state and submit_candidate may appear in either order; every upload must
    finish before its transaction and every request must receive its own reply.
    """
    errors, spans, uploads, declarations = [], [], {}, []
    counts, host_wait = Counter(), defaultdict(float)
    pending, previous_reply, sequence, acknowledged = None, None, 0, False
    call_ids, history_ids, allowed_artifacts = set(), set(), set()
    start = events[0].get("frame", {}) if events else {}
    try:
        if not events or events[0].get("direction") != "host":
            raise ValueError("First frame must be the host start")
        _validate_start(start)
        allowed_artifacts.add(start["parent_artifact"]["artifact_id"])
        allowed_artifacts.update(item["artifact_id"] for item in start["artifacts"])
    except Exception as error:
        errors.append("invalid_start: " + str(error))
    previous_time = -math.inf
    timed = True
    for position, event in enumerate(events):
        elapsed = event.get("elapsed_seconds")
        if elapsed is None:
            timed = False
        elif isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < previous_time:
            errors.append("nonmonotonic_or_invalid_timestamp:%d" % position)
        else:
            previous_time = elapsed
        if position == 0:
            previous_reply = elapsed
            continue
        frame, direction = event.get("frame", {}), event.get("direction")
        kind = frame.get("type")
        try:
            if acknowledged:
                raise ValueError("frame_after_round_ack")
            if direction == "harness":
                if pending is not None:
                    raise ValueError("request_before_previous_reply")
                if kind not in RESPONSES:
                    raise ValueError("unknown_request:" + str(kind))
                if kind != "round_complete":
                    if type(frame.get("id")) is not int or frame["id"] != sequence:
                        raise ValueError("nonsequential_rpc_id")
                    sequence += 1
                if kind == "model_request":
                    if set(frame) - {"type", "id"} - _MODEL_OPTIONS:
                        raise ValueError("unsupported_model_option")
                if kind in ("candidate_request", "round_complete"):
                    declaration = frame["record"] if kind == "round_complete" else frame
                    try:
                        _declaration(declaration.get("hypothesis"), declaration.get("change_summary"),
                                     declaration.get("configuration"), declaration.get("seed"))
                    except HarnessError as error:
                        raise ValueError("sdk_declaration_invalid:" + str(error)) from error
                    if kind == "round_complete" and not _text(declaration.get("reason")):
                        raise ValueError("sdk_reason_invalid")
                    declarations.append({"request_type": kind,
                        **{key + "_utf8_bytes": len(declaration[key].encode("utf-8"))
                           for key in ("hypothesis", "change_summary", "reason") if key in declaration},
                        "configuration_serialized_bytes": len(json.dumps(declaration["configuration"],
                            sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8"))})
                if kind in ("upload_chunk_request", "upload_finish_request", "candidate_request", "state_request"):
                    upload = uploads.get(frame.get("upload_id"))
                    if upload is None:
                        raise ValueError("unknown_upload")
                    if kind in ("candidate_request", "state_request"):
                        expected_kind = "candidate" if kind == "candidate_request" else "state"
                        if not upload["finished"] or upload["kind"] != expected_kind:
                            raise ValueError("transaction_before_upload_finish_or_wrong_kind")
                    elif upload["finished"]:
                        raise ValueError("upload_changed_after_finish")
                    elif kind == "upload_chunk_request":
                        item = upload["files"][frame["path"]]
                        raw = base64.b64decode(frame["data_base64"], validate=True)
                        if frame["offset"] != item["received"] or not raw or item["received"] + len(raw) > item["size_bytes"]:
                            raise ValueError("upload_chunk_offset_or_size")
                    else:
                        if any(f["received"] != f["size_bytes"] or "sha256:" + f["hash"].hexdigest() != f["sha256"] for f in upload["files"].values()):
                            raise ValueError("upload_finish_before_all_bytes_or_digest_mismatch")
                if kind == "artifact_read_request" and frame.get("artifact_id") not in allowed_artifacts:
                    raise ValueError("historical_artifact_not_authorized")
                if kind == "round_complete":
                    record = frame["record"]
                    refs = record.get("model_call_ids", [])
                    histories = record.get("history_ids", [])
                    if not refs or len(set(refs)) != len(refs) or not set(refs) <= call_ids:
                        raise ValueError("finish_uses_unverified_model_calls")
                    if len(set(histories)) != len(histories) or not set(histories) <= history_ids:
                        raise ValueError("finish_uses_unread_history")
                    if record.get("selected_artifact_id") not in allowed_artifacts:
                        raise ValueError("finish_selects_unknown_artifact")
                counts[kind] += 1
                pending = (frame, elapsed, previous_reply)
            elif direction == "host":
                if pending is None:
                    raise ValueError("unsolicited_host_reply")
                request, requested_at, prior_reply = pending
                request_kind = request["type"]
                if kind != RESPONSES[request_kind] or (request_kind != "round_complete" and frame.get("id") != request["id"]):
                    raise ValueError("response_type_or_id_mismatch")
                if request_kind == "upload_begin_request":
                    identifier = frame.get("upload_id")
                    if not identifier or identifier in uploads:
                        raise ValueError("invalid_or_duplicate_upload_id")
                    files = request["files"]
                    if len(files) > 64 or len({f["path"] for f in files}) != len(files):
                        raise ValueError("upload_manifest_file_count_or_duplicate")
                    uploads[identifier] = {"kind": request["kind"], "finished": False,
                        "files": {f["path"]: {**f, "received": 0, "hash": hashlib.sha256()} for f in files}}
                elif request_kind == "upload_chunk_request":
                    item = uploads[request["upload_id"]]["files"][request["path"]]
                    raw = base64.b64decode(request["data_base64"], validate=True)
                    if frame.get("offset") != item["received"] + len(raw):
                        raise ValueError("upload_ack_offset_mismatch")
                    item["received"] += len(raw)
                    item["hash"].update(raw)
                elif request_kind == "upload_finish_request":
                    if frame.get("upload_id") != request["upload_id"]:
                        raise ValueError("upload_finish_id_mismatch")
                    uploads[request["upload_id"]]["finished"] = True
                elif request_kind == "model_request":
                    identifier = frame.get("call_id")
                    if not isinstance(identifier, str) or not identifier or identifier in call_ids or not isinstance(frame.get("response"), dict):
                        raise ValueError("invalid_or_duplicate_model_receipt")
                    call_ids.add(identifier)
                elif request_kind == "history_request":
                    rows = frame["records"]
                    ids = [r["id"] for r in rows]
                    if len(set(ids)) != len(ids):
                        raise ValueError("duplicate_history_ids")
                    history_ids = set(ids)
                elif request_kind == "candidate_request":
                    identifier = frame.get("artifact_id")
                    if not isinstance(identifier, str) or not identifier:
                        raise ValueError("invalid_candidate_receipt")
                    allowed_artifacts.add(identifier)
                elif request_kind == "round_complete":
                    if frame.get("accepted") is not True:
                        raise ValueError("round_ack_not_accepted")
                    acknowledged = True
                wait = None if elapsed is None or requested_at is None else elapsed - requested_at
                gap = None if requested_at is None or prior_reply is None else requested_at - prior_reply
                spans.append({"request_type": request_kind, "id": request.get("id"),
                    "request_elapsed_seconds": requested_at, "response_elapsed_seconds": elapsed,
                    "host_wait_seconds": wait, "participant_gap_seconds": gap})
                if wait is not None:
                    host_wait[request_kind] += wait
                pending, previous_reply = None, elapsed
            else:
                raise ValueError("invalid_direction")
        except (ValueError, KeyError, TypeError) as error:
            errors.append("frame_%d:%s" % (position, error))
            break
    if pending is not None:
        errors.append("outstanding_request_at_end:" + str(pending[0].get("type")))
    if not acknowledged:
        errors.append("missing_accepted_round_ack")
    first_request = next((e.get("elapsed_seconds") for e in events if e.get("direction") == "harness"), None)
    return {"format": "sdk104.protocol-timing.v1", "order_valid": not errors, "errors": errors,
        "acknowledged": acknowledged, "timestamps_available": timed,
        "first_protocol_request_seconds": first_request,
        "elapsed_seconds_to_ack": events[-1].get("elapsed_seconds") if acknowledged else None,
        "round_trips": sum(counts.values()), "request_counts": dict(counts),
        "host_wait_seconds_by_request": dict(host_wait),
        "participant_gap_seconds_total": sum(s["participant_gap_seconds"] or 0 for s in spans),
        "declaration_sizes": declarations, "spans": spans}


def audit_file(path):
    return audit_events([json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()])
