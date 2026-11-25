"""Completed model-response faults; no SDK transport errors or external calls."""
import json


CASES = (
    "plain", "fence_crlf", "repair_once", "truncated", "leading_text",
    "duplicate_key", "extra_field", "nonfinite", "wrong_type", "long_text",
    "wrong_parent", "unknown_experiment", "wrong_selection", "empty_choices",
    "message_string", "content_list", "escaped_surrogate", "raw_surrogate",
    "metadata_surrogate",
)


def inject_reply(response, request, case, phase_attempt):
    """Mutate an already completed peer response; preserve its actual call ID."""
    content = response["choices"][0]["message"]["content"]
    decision = json.loads(content)
    phase = request["phase"]
    if case == "plain":
        return
    if case == "fence_crlf":
        content = "```JSON\r\n" + content + "\r\n```"
    elif case == "repair_once":
        if phase != "plan" or phase_attempt > 1:
            return
        content = '{"action":'
    elif case == "truncated":
        content = content[:-1]
    elif case == "leading_text":
        content = "Here is the decision:\n" + content
    elif case == "duplicate_key":
        content = content[:-1] + ',"reason":"duplicate"}'
    elif case == "extra_field":
        content = content[:-1] + ',"extra":"unexpected"}'
    elif case == "nonfinite":
        content = content[:-1] + ',"reason":NaN}'
    elif case == "wrong_type":
        decision["reason"] = None
        content = json.dumps(decision)
    elif case == "long_text":
        decision["reason"] = "中" * 1334
        content = json.dumps(decision, ensure_ascii=False)
    elif case == "wrong_parent":
        if phase != "plan":
            return
        decision["target_artifact_id"] = "not-the-actual-parent"
        content = json.dumps(decision)
    elif case == "unknown_experiment":
        if phase != "plan":
            return
        decision.update(action="experiment", experiment="outside-menu")
        content = json.dumps(decision)
    elif case == "wrong_selection":
        if phase != "select":
            return
        decision["selected_artifact_id"] = "outside-selection-whitelist"
        content = json.dumps(decision)
    elif case == "empty_choices":
        response["choices"] = []
        return
    elif case == "message_string":
        response["choices"][0]["message"] = "unexpected envelope"
        return
    elif case == "content_list":
        response["choices"][0]["message"]["content"] = [{"text": content}]
        return
    elif case in ("escaped_surrogate", "raw_surrogate"):
        decision["reason"] = "invalid Unicode " + chr(0xD800)
        content = json.dumps(decision, ensure_ascii=case == "escaped_surrogate")
    elif case == "metadata_surrogate":
        response["optional_diagnostic"] = chr(0xD800)
        return
    else:
        raise ValueError("Unknown response fault: " + case)
    response["choices"][0]["message"]["content"] = content
