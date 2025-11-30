import numpy as np

from .p_above import p_above
from .p_bottleneck import p_bottleneck
from .p_ceiling import p_ceiling
from .p_confidence import p_columns
from .p_fit import get_fit
from .p_ineffs import p_ineffs
from .p_utils import p_accuracy


def p_nca_cr_cm_conf(loop_data, bn_data):
    if "ce_cm_conf_columns" in loop_data and loop_data["ce_cm_conf_columns"] is not None:
        columns = loop_data["ce_cm_conf_columns"]
    else:
        columns = p_columns(loop_data, True)

    if columns.shape[1] > 1:
        x = columns[3, :]
        y = columns[4, :]
        weights = columns[0, :]

        w = np.sqrt(weights)

        slope, intercept = np.polyfit(x, y, 1, w=w)

    else:
        x = np.array([loop_data["scope_theo"][0], loop_data["scope_theo"][1]])
        y = np.array([columns[4, 0], columns[4, 0]])

        slope, intercept = np.polyfit(x, y, 1)

    ceiling = p_ceiling(loop_data, slope, intercept)
    above = p_above(loop_data, slope, intercept)
    effect = ceiling / loop_data["scope_area"]
    accuracy = p_accuracy(loop_data, above)
    fit = get_fit(ceiling, loop_data.get("ce_fdh_ceiling", float("nan")))
    ineffs = p_ineffs(loop_data, slope, intercept)
    bottleneck = p_bottleneck(loop_data, bn_data, slope, intercept)

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
        "bottleneck": bottleneck,
        "columns": columns,
    }
