"""用于公开验证和回归规划的实验顺序。"""


def ordered_experiments(config, dataset, *, mode="dataset-defaults"):
    menu = {spec["name"]: spec for spec in config["experiments"] if dataset in spec.get("datasets", ("B", "C"))}
    if mode == "configured":
        return menu
    sequence = config.get("preferred_experiment_sequence", {}).get(dataset, ())
    if mode == "dataset-defaults" and sequence:
        if any(name not in menu for name in sequence):
            raise ValueError(f"Preferred experiment is unavailable for {dataset}")
        return {**{name: menu[name] for name in sequence}, **menu}
    preferences = ({"B": "bearing_mlp", "C": "cmapss_condition_mlp"} if mode == "mlp-first"
                   else config.get("default_first_experiment", {}))
    first = preferences.get(dataset)
    if first is None:
        return menu
    if not isinstance(first, str) or first not in menu:
        raise ValueError(f"Default experiment is unavailable for {dataset}: {first}")
    return {first: menu[first], **menu}


def stage_experiment_preference(config, dataset, stage_round, allowed):
    """Recommend the scheduled experiment only when the current menu permits it."""
    sequence = config.get("preferred_experiment_sequence", {}).get(dataset, ())
    if not sequence or stage_round < 1:
        return None
    name = sequence[min(stage_round, len(sequence)) - 1]
    return name if name in allowed else None
