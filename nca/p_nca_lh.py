import numpy as np

from .p_above import p_above
from .p_bottleneck import p_bottleneck
from .p_ceiling import p_ceiling
from .p_fit import get_fit
from .p_ineffs import p_ineffs
from .p_peers import p_peers
from .p_utils import p_accuracy


def p_nca_lh(loop_data, bn_data):
    peers = p_peers(loop_data)

    if peers is None:
        unique_peers = np.empty((0, 2))
    else:
        unique_peers = np.unique(peers, axis=0)

    if unique_peers.shape[0] > 1:
        x1 = peers[0, 0]
        y1 = peers[0, 1]
        x2 = peers[-1, 0]
        y2 = peers[-1, 1]

        if x2 == x1:
            slope = float("inf") if y2 > y1 else float("-inf")
            intercept = float("nan")
        else:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y2 - (slope * x2)

        line = [intercept, slope]
        ceiling = p_ceiling(loop_data, slope, intercept)
        effect = ceiling / loop_data["scope_area"]
        ineffs = p_ineffs(loop_data, slope, intercept)
        above = p_above(loop_data, slope, intercept)

    else:
        line = None
        slope = float("nan")
        intercept = float("nan")
        ceiling = 0
        effect = 0
        ineffs = {"x": float("nan"), "y": float("nan"), "abs": float("nan"), "rel": float("nan")}
        above = float("nan")
        peers = np.empty((0, 2))

    accuracy = p_accuracy(loop_data, above)
    fit = get_fit(ceiling, loop_data.get("ce_fdh_ceiling", float("nan")))
    bottleneck = p_bottleneck(loop_data, bn_data, slope, intercept)

    return {
        "line": line,
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
