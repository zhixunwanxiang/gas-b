"""Paired warm single-row CPU checks, explicitly excluding SDK/JSON and startup."""
import statistics
import time
import numpy as np


def measure(parent, child, x, rules, deadline):
    settings = rules.get("prediction_budget")
    if not settings:
        return None
    rows = x[np.linspace(0, len(x) - 1, min(32, len(x)), dtype=int)]
    for model in (parent, child):
        for row in rows[:4]:
            model.predict(row[None, :])
    def block(model):
        count, started = 0, time.process_time()
        elapsed = 0.
        while elapsed < .08:
            for row in rows:
                model.predict(row[None, :])
            count += len(rows)
            elapsed = time.process_time() - started
            if time.monotonic() >= deadline:
                raise TimeoutError("Single-row CPU check exceeded candidate budget")
        return elapsed / count
    observations = []
    for index in range(3):
        if index % 2:
            after, before = block(child), block(parent)
        else:
            before, after = block(parent), block(child)
        observations.append({"parent_cpu_seconds_per_row": before, "candidate_cpu_seconds_per_row": after,
                             "ratio": after / before})
    ratio = statistics.median(r["ratio"] for r in observations)
    return {"scope": "warm_single_row_model_cpu; excludes SDK JSON loading and process startup",
            "paired_order_alternated": True, "repeats": observations, "median_cpu_ratio": ratio,
            "maximum_cpu_ratio": settings["maximum_cpu_ratio"],
            "eligible": bool(np.isfinite(ratio) and ratio <= settings["maximum_cpu_ratio"]),
            "official_resource_score_verified": False}
