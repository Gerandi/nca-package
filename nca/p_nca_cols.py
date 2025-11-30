import numpy as np
from scipy.stats import linregress

from .p_above import p_above
from .p_bottleneck import p_bottleneck
from .p_ceiling import p_ceiling
from .p_fit import get_fit
from .p_ineffs import p_ineffs
from .p_peers import p_get_line_peers
from .p_utils import p_accuracy


def p_nca_cols(loop_data, bn_data):
    x = loop_data["x"]
    y = loop_data["y"]
    flip_y = loop_data["flip_y"]

    # Unpacking linregress - only slope and intercept are used
    slope, intercept, _r_value, _p_value, _std_err = linregress(x, y)

    y_pred = slope * x + intercept
    residuals = y - y_pred

    if not flip_y:
        intercept += np.max(residuals)
    else:
        intercept += np.min(residuals)

    ceiling = p_ceiling(loop_data, slope, intercept)
    effect = ceiling / loop_data["scope_area"]
    above = p_above(loop_data, slope, intercept)
    accuracy = p_accuracy(loop_data, above)
    fit = get_fit(ceiling, loop_data.get("ce_fdh_ceiling", float("nan")))
    ineffs = p_ineffs(loop_data, slope, intercept)
    bottleneck = p_bottleneck(loop_data, bn_data, slope, intercept)

    ld_copy = loop_data.copy()
    ld_copy["flip_x"] = ld_copy["flip_y"]
    peers = p_get_line_peers(ld_copy, intercept, slope)

    return {
        "line": [intercept, slope],
        "peers": peers,
        "slope": slope,
        "intercept": intercept,
        "ceiling": ceiling,
        "effect": effect,
        "above": above,
        "accuracy": accuracy,
        "fit": fit,
        "ineffs": ineffs,
        "bottleneck": bottleneck,
    }
