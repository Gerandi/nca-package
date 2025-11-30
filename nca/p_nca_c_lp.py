import numpy as np
from scipy.optimize import linprog

from .p_bottleneck import p_bottleneck
from .p_ceiling import p_ceiling
from .p_fit import get_fit
from .p_ineffs import p_ineffs
from .p_peers import p_best_peers, p_peers
from .p_utils import p_accuracy


def p_nca_c_lp(loop_data, bn_data):
    peers = p_peers(loop_data)

    if peers is None:
        unique_peers = np.empty((0, 2))
    else:
        unique_peers = np.unique(peers.values, axis=0)

    if unique_peers.shape[0] > 1:
        K = unique_peers.shape[0]

        s = int(loop_data["flip_x"]) + int(loop_data["flip_y"])
        factor = -1 if s == 1 else 1

        sum_peers_x = np.sum(unique_peers[:, 0])
        c = np.array([K, -K, factor * sum_peers_x])

        A = np.zeros((K, 3))
        A[:, 0] = 1
        A[:, 1] = -1
        A[:, 2] = factor * unique_peers[:, 0]

        b = unique_peers[:, 1]

        if loop_data["flip_y"]:
            # Maximize c @ x subject to A @ x <= b
            # linprog minimizes. So minimize -c @ x.
            res = linprog(-c, A_ub=A, b_ub=b, bounds=(0, None), method="highs")
        else:
            # Minimize c @ x subject to A @ x >= b
            # -> -A @ x <= -b
            res = linprog(c, A_ub=-A, b_ub=-b, bounds=(0, None), method="highs")

        if res.success:
            sol = res.x
            intercept = sol[0] - sol[1]
            slope = factor * sol[2]

            line = [intercept, slope]
            ceiling = p_ceiling(loop_data, slope, intercept)
            above = 0

            peers = p_best_peers(unique_peers, intercept, slope)
        else:
            # Fallback if LP fails?
            line = None
            slope = float("nan")
            intercept = float("nan")
            ceiling = 0
            above = float("nan")
            peers = np.empty((0, 2))

    else:
        line = None
        slope = float("nan")
        intercept = float("nan")
        ceiling = 0
        above = float("nan")
        peers = np.empty((0, 2))

    effect = ceiling / loop_data["scope_area"]
    accuracy = p_accuracy(loop_data, above)
    fit = get_fit(ceiling, loop_data.get("ce_fdh_ceiling", float("nan")))
    ineffs = p_ineffs(loop_data, slope, intercept)
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
