"""Definitions shared by the standard and protocol-informed DQN ablations."""

SENSORS = ("Na", "K", "Ca", "TEMP", "FR")
SENSOR_VARIANTS = tuple(f"No {sensor}" for sensor in SENSORS)
FEATURE_TYPE_VARIANTS = (
    "No current signals",
    "No forecast delta",
    "No forecast slope",
    "No uncertainty",
    "No observed slope",
    "Current only",
)
SENSOR_FIGURE_VARIANTS = ("Full", *SENSOR_VARIANTS)
FEATURE_TYPE_FIGURE_VARIANTS = ("Full", *FEATURE_TYPE_VARIANTS)


def make_feature_sets(columns):
    """Return sweat-sensor observation variants; HR remains reward-only."""
    columns = list(columns)
    policy_columns = [column for column in columns if column != "HR_current"]

    def without(predicate):
        return [column for column in policy_columns if not predicate(column)]

    current_signals = [
        column for column in policy_columns if column.endswith("_current")
    ]
    variants = [("Full", policy_columns)]
    variants.extend(
        (
            f"No {sensor}",
            without(lambda column, prefix=f"{sensor}_": column.startswith(prefix)),
        )
        for sensor in SENSORS
    )
    variants.extend(
        [
            (
                "No current signals",
                without(lambda column: column in current_signals),
            ),
            (
                "No forecast delta",
                without(
                    lambda column: "_pred_end_delta" in column
                    or "_pred_mean_delta" in column
                ),
            ),
            (
                "No forecast slope",
                without(lambda column: "_pred_slope" in column),
            ),
            (
                "No uncertainty",
                without(lambda column: "_endpoint_sd" in column),
            ),
            (
                "No observed slope",
                without(lambda column: "_observed_cumulative_slope" in column),
            ),
            ("Current only", current_signals),
        ]
    )
    return variants


def variant_family(variant):
    if variant == "Full":
        return "Reference"
    if variant in SENSOR_VARIANTS:
        return "Sensor"
    if variant == "Current only":
        return "Compact state"
    return "Feature type"


def figure_subset(frame, variants):
    """Select variants in a stable figure order and assign compact row order."""
    order = {variant: index for index, variant in enumerate(variants)}
    subset = frame.loc[frame["Variant"].isin(variants)].copy()
    subset["Variant_order"] = subset["Variant"].map(order)
    return subset.sort_values("Variant_order").reset_index(drop=True)
