import importlib

import numpy as np

from .p_utils import p_if_min_else_max


def p_ceiling(loop_data, slope, intercept):
    flip_x = loop_data["flip_x"]
    flip_y = loop_data["flip_y"]

    if np.isnan(slope) or np.isnan(intercept):
        return float("nan")

    if (flip_x == flip_y) and slope < 0:
        return float("nan")
    if (flip_x != flip_y) and slope > 0:
        return float("nan")

    theo = loop_data["scope_theo"]
    p_l = [theo[0], slope * theo[0] + intercept]
    p_r = [theo[1], slope * theo[1] + intercept]

    if p_l[1] > theo[3] and p_r[1] > theo[3]:
        return float("nan") if not flip_y else loop_data["scope_area"]
    if p_l[1] < theo[2] and p_r[1] < theo[2]:
        return loop_data["scope_area"] if not flip_y else float("nan")

    area_above = 0

    if slope > 0:
        if p_l[1] < theo[2]:
            p_l = [(theo[2] - intercept) / slope, theo[2]]

        if p_r[1] > theo[3]:
            p_r = [(theo[3] - intercept) / slope, theo[3]]

        area_above = 0.5 * (p_r[0] - p_l[0]) * (p_r[1] - p_l[1])
        area_above += (theo[1] - theo[0]) * (theo[3] - p_r[1])
        area_above += (p_l[0] - theo[0]) * (theo[3] - theo[2])
        area_above -= (p_l[0] - theo[0]) * (theo[3] - p_r[1])

    elif slope < 0:
        if p_l[1] > theo[3]:
            p_l = [(theo[3] - intercept) / slope, theo[3]]

        if p_r[1] < theo[2]:
            p_r = [(theo[2] - intercept) / slope, theo[2]]

        area_above = 0.5 * (p_r[0] - p_l[0]) * (p_l[1] - p_r[1])
        area_above += (theo[1] - theo[0]) * (theo[3] - p_l[1])
        area_above += (theo[1] - p_r[0]) * (theo[3] - theo[2])
        area_above -= (theo[1] - p_r[0]) * (theo[3] - p_l[1])

    elif slope == 0:
        area_above = (theo[1] - theo[0]) * (theo[3] - intercept)

    return area_above if not flip_y else loop_data["scope_area"] - area_above


def p_ce_ceiling(loop_data, peers, method):
    emp = loop_data["scope_emp"]
    theo = loop_data["scope_theo"]
    flip_x = loop_data["flip_x"]
    flip_y = loop_data["flip_y"]

    ceiling = p_scope_ceiling(peers, theo, emp, flip_x, flip_y)

    # Convert DataFrame to numpy array for unique check
    peers_arr = peers.values if hasattr(peers, "values") else peers
    unique_peers = np.unique(peers_arr, axis=0)

    if len(unique_peers) <= 1:
        if method in ["fdh", "vrs"]:
            return ceiling
        if method == "con":
            y_val = peers.iloc[0, 1]
            if (not flip_y and y_val > theo[3]) or (flip_y and y_val < theo[2]):
                return float("nan")
            return ceiling

    for i in range(len(peers) - 1):
        if method == "fdh":
            emp_x = emp[1] if flip_x else emp[0]
            x_length = peers.iloc[i + 1, 0] - emp_x
            y_length = peers.iloc[i + 1, 1] - peers.iloc[i, 1]
            ceiling += abs(x_length * y_length)

        elif method == "vrs":
            part_a = (peers.iloc[i + 1, 1] - peers.iloc[i, 1]) * (
                peers.iloc[i + 1, 0] - peers.iloc[0, 0]
            )
            part_b = (
                0.5
                * (peers.iloc[i + 1, 1] - peers.iloc[i, 1])
                * (peers.iloc[i + 1, 0] - peers.iloc[i, 0])
            )
            ceiling += abs(part_a) - abs(part_b)

        elif method == "con":
            emp_x = emp[1] if flip_x else emp[0]
            x_length = peers.iloc[i + 1, 0] - emp_x

            emp_y = emp[2] if flip_y else emp[3]

            y1 = p_if_min_else_max(not flip_y, emp_y, peers.iloc[i, 1])
            y2 = p_if_min_else_max(not flip_y, emp_y, peers.iloc[i + 1, 1])
            y_length = y2 - y1
            ceiling += abs(x_length * y_length)

    return ceiling


def p_scope_ceiling(peers, theo, emp, flip_x, flip_y):
    """Calculate ceiling for scope area.

    Note: peers, flip_x, flip_y kept for API compatibility.
    """
    _ = (peers, flip_x, flip_y)  # API compatibility
    if emp == theo:
        return 0

    x1 = emp[0] - theo[0]
    x2 = theo[1] - emp[1]
    y1 = emp[2] - theo[2]
    y2 = theo[3] - emp[3]

    left = x1 * (theo[3] - theo[2])
    right = x2 * (theo[3] - theo[2])
    lower = (theo[1] - theo[0]) * y1
    upper = (theo[1] - theo[0]) * y2

    if not flip_x and not flip_y:
        cross = x1 * y2
        return left + upper - cross
    if flip_x and not flip_y:
        cross = x2 * y2
        return right + upper - cross
    if flip_x and flip_y:
        cross = x2 * y1
        return right + lower - cross
    if not flip_x and flip_y:
        cross = x1 * y1
        return left + lower - cross
    return 0


def p_cm_ceiling(loop_data, peers):
    theo = loop_data["scope_theo"]
    flip_x = loop_data["flip_x"]
    flip_y = loop_data["flip_y"]

    ceiling = 0
    for i in range(len(peers)):
        if i < len(peers) - 1:
            next_x = peers.iloc[i + 1, 0]
        else:
            next_x = theo[0] if flip_x else theo[1]

        x_length = abs(peers.iloc[i, 0] - next_x)

        target_y = theo[2] if flip_y else theo[3]
        y_length = abs(peers.iloc[i, 1] - target_y)

        ceiling += x_length * y_length

    return ceiling


def p_nca_wrapper(ceiling, loop_data, bn_data, effect_aggregation):
    module_name = f".p_nca_{ceiling}"
    try:
        module = importlib.import_module(module_name, package=__package__)
        func = getattr(module, f"p_nca_{ceiling}")
    except (ImportError, AttributeError) as exc:
        raise ImportError(f"Could not import p_nca_{ceiling}") from exc

    analysis = func(loop_data, bn_data)

    ld = loop_data.copy()

    ld["flip_x"] = not ld["flip_x"]
    if 2 in effect_aggregation:
        tmp_analysis = func(ld, bn_data)
        analysis["effect"] += tmp_analysis["effect"]

    ld["flip_y"] = not ld["flip_y"]
    if 4 in effect_aggregation:
        tmp_analysis = func(ld, bn_data)
        analysis["effect"] += tmp_analysis["effect"]

    ld["flip_x"] = not ld["flip_x"]
    if 3 in effect_aggregation:
        tmp_analysis = func(ld, bn_data)
        analysis["effect"] += tmp_analysis["effect"]

    return analysis
