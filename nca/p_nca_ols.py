import numpy as np
from scipy.stats import linregress

from .p_above import p_above
from .p_ceiling import p_ceiling
from .p_ineffs import p_ineffs
from .p_utils import p_accuracy


def p_nca_ols(loop_data, bn_data):
    """OLS ceiling method."""
    _ = bn_data  # API compatibility
    x = loop_data["x"]
    y = loop_data["y"]

    slope, intercept, _r_value, _p_value, _std_err = linregress(x, y)

    ceiling = p_ceiling(loop_data, slope, intercept)
    effect = ceiling / loop_data["scope_area"]
    above = p_above(loop_data, slope, intercept)
    accuracy = p_accuracy(loop_data, above)

    fdh_ceiling = loop_data.get("ce_fdh_ceiling", float("nan"))
    if fdh_ceiling != 0 and not np.isnan(fdh_ceiling):
        fit = 100 * ceiling / fdh_ceiling
    else:
        fit = float("nan")

    ineffs = p_ineffs(loop_data, slope, intercept)

    return {
        "line": [intercept, slope],
        "slope": slope,
        "intercept": intercept,
        "ceiling": ceiling,
        "effect": effect,
        "above": above,
        "accuracy": accuracy,
        "fit": fit,
        "ineffs": ineffs,
        "bottleneck": None,
    }
