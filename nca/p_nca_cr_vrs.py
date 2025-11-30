import numpy as np

from .p_above import p_above
from .p_bottleneck import p_bottleneck
from .p_ceiling import p_ceiling
from .p_fit import get_fit
from .p_ineffs import p_ineffs
from .p_peers import p_peers
from .p_utils import p_accuracy


def p_nca_cr_vrs(loop_data, bn_data):
    peers = p_peers(loop_data, vrs=True)

    if peers is None:
        unique_peers = np.empty((0, 2))
    else:
        unique_peers = np.unique(peers.values, axis=0)

    if unique_peers.shape[0] > 1:
        x = peers.iloc[:, 0].values
        y = peers.iloc[:, 1].values

        slope, intercept = np.polyfit(x, y, 1)

        ceiling = p_ceiling(loop_data, slope, intercept)
        above = p_above(loop_data, slope, intercept)

    else:
        ceiling = 0
        intercept = float("nan")
        slope = float("nan")
        above = 0
        peers = np.empty((0, 2))

    effect = ceiling / loop_data["scope_area"]
    accuracy = p_accuracy(loop_data, above)
    fit = get_fit(ceiling, loop_data.get("ce_fdh_ceiling", float("nan")))
    ineffs = p_ineffs(loop_data, slope, intercept)
    bottleneck = p_bottleneck(loop_data, bn_data, slope, intercept)

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
