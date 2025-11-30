import math

import numpy as np


def p_ineffs(loop_data, slope, intercept):
    flip_x = loop_data["flip_x"]
    flip_y = loop_data["flip_y"]

    # Upper left and lower right
    if (flip_x == flip_y) and (math.isnan(slope) or slope <= 1e-3):
        return {"x": float("nan"), "y": float("nan"), "abs": float("nan"), "rel": float("nan")}

    # Lower left and upper right
    if (flip_x != flip_y) and (math.isnan(slope) or slope >= -1e-3):
        return {"x": float("nan"), "y": float("nan"), "abs": float("nan"), "rel": float("nan")}

    # Get the x-value of the line crossing the upper or low boundry
    y_xlim = loop_data["scope_theo"][3] if not flip_y else loop_data["scope_theo"][2]
    x_lim = (y_xlim - intercept) / slope

    # Get the y-value of the line crossing the left or right boundry
    x_ylim = loop_data["scope_theo"][0] if not flip_x else loop_data["scope_theo"][1]
    y_lim = slope * x_ylim + intercept

    return p_ineff(loop_data, x_lim, y_lim)


def p_ineffs_ce(loop_data, peers):
    # if there is only one peer, the ceiling zone is zero
    peers_arr = peers.values if hasattr(peers, "values") else peers
    unique_peers = np.unique(peers_arr, axis=0)
    if unique_peers.shape[0] == 1:
        return {"x": float("nan"), "y": float("nan"), "abs": float("nan"), "rel": float("nan")}

    # x.lim <- tail(peers, n=1)[1] -> last row, first col (x)
    x_lim = peers.iloc[-1, 0]

    # y.lim <- head(peers, n=1)[2] -> first row, second col (y)
    y_lim = peers.iloc[0, 1]

    return p_ineff(loop_data, x_lim, y_lim)


def p_ineff(loop_data, x_lim, y_lim):
    flip_x = loop_data["flip_x"]
    flip_y = loop_data["flip_y"]
    scope_theo = loop_data["scope_theo"]

    if flip_x:
        x_lim = max(scope_theo[0], x_lim)
        x_eff = x_lim - scope_theo[0]
    else:
        x_lim = min(scope_theo[1], x_lim)
        x_eff = scope_theo[1] - x_lim

    if flip_y:
        y_lim = min(scope_theo[3], y_lim)
        y_eff = scope_theo[3] - y_lim
    else:
        y_lim = max(scope_theo[2], y_lim)
        y_eff = y_lim - scope_theo[2]

    ineffs_x = x_eff / (scope_theo[1] - scope_theo[0])
    ineffs_y = y_eff / (scope_theo[3] - scope_theo[2])
    ineffs_rel = ineffs_x + ineffs_y - ineffs_x * ineffs_y
    ineffs_abs = loop_data["scope_area"] * ineffs_rel

    return {"x": ineffs_x * 100, "y": ineffs_y * 100, "abs": ineffs_abs, "rel": ineffs_rel * 100}
