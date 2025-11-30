import numpy as np
import pandas as pd

from .p_utils import p_is_equal


def p_peers(loop_data, vrs=False):
    x = loop_data["x"]
    y = loop_data["y"]
    flip_x = loop_data["flip_x"]
    flip_y = loop_data["flip_y"]

    if len(x) < 2:
        return None

    df = pd.DataFrame({"x": x, "y": y})

    asc_x = not flip_x
    asc_y = not flip_y

    df_sorted = df.sort_values(by=["x", "y"], ascending=[asc_x, asc_y])

    x_sorted = df_sorted["x"].values
    y_sorted = df_sorted["y"].values
    rownames_org = df_sorted.index.values

    peers = []
    peers.append([x_sorted[0], y_sorted[0]])

    peer_indices = [rownames_org[0]]

    for i in range(1, len(x_sorted)):
        x_curr = x_sorted[i]
        y_curr = y_sorted[i]

        x_prev = peers[-1][0]
        y_prev = peers[-1][1]

        x_equal = p_is_equal(x_prev, x_curr)
        y_equal = p_is_equal(y_prev, y_curr)

        if flip_y:
            next_peer = y_curr < y_prev
        else:
            next_peer = y_curr > y_prev

        if next_peer or (x_equal and y_equal):
            while x_equal and not y_equal:
                peers.pop()
                peer_indices.pop()
                if not peers:
                    break
                x_prev = peers[-1][0]
                y_prev = peers[-1][1]
                x_equal = p_is_equal(x_prev, x_curr)
                y_equal = p_is_equal(y_prev, y_curr)

            if vrs:
                while p_invalid_peers(np.array(peers), x_curr, y_curr, flip_x, flip_y):
                    peers.pop()
                    peer_indices.pop()

            peers.append([x_curr, y_curr])
            peer_indices.append(rownames_org[i])

    peers_df = pd.DataFrame(peers, columns=["x", "y"], index=peer_indices)

    return peers_df


def p_invalid_peers(peers, x3, y3, flip_x, flip_y):
    if len(peers) < 2:
        return False

    # peers is list of lists here
    x1 = peers[-2][0]
    y1 = peers[-2][1]
    x2 = peers[-1][0]
    y2 = peers[-1][1]

    if (p_is_equal(x1, x2) and p_is_equal(y1, y2)) or (p_is_equal(x2, x3) and p_is_equal(y2, y3)):
        return False

    if x2 == x1:
        slope0 = float("inf") if y2 > y1 else float("-inf")
    else:
        slope0 = (y2 - y1) / (x2 - x1)

    if x3 == x2:
        slope1 = float("inf") if y3 > y2 else float("-inf")
    else:
        slope1 = (y3 - y2) / (x3 - x2)

    if flip_x == flip_y:
        return slope0 <= slope1
    return slope0 >= slope1


def p_aggregate_peers(model_peers, x):
    peers_list = []
    # model_peers is dict: ceiling -> {x_name -> peers_df}
    # x passed here is likely the index (1) or name?
    # In nca_outliers.py: p_aggregate_peers(model['peers'], 1)
    # But model['peers'] keys are ceilings.
    # model['peers'][ceiling] is dict keyed by x_name.
    # Wait, nca_outliers passes 1?
    # In R: p_aggregate_peers(model$peers, 1)
    # In R, model$peers is a list of lists?
    # model$peers[[ceiling]][[x]]

    # In Python nca.py: peers[ceiling][x_name] = analysis['peers']
    # So model['peers'] is dict[ceiling][x_name].

    # If x is passed as 1, it might mean the first variable?
    # But p_aggregate_peers in R takes 'index'.
    # Let's assume x is the variable name or index.

    # If x is 1 (int), we need to find the name?
    # But nca_outliers calls it with 1.
    # In nca_outliers.py:
    # summary = model['summaries'][x] (x is name)
    # ...
    # 'peers': p_aggregate_peers(model['peers'], 1)
    # This looks like a bug in my port of nca_outliers.py or p_aggregate_peers.

    # In R nca_outliers.R:
    # params <- list(..., peers = p_aggregate_peers(model$peers, 1))
    # In R, model$peers is a list of lists.
    # p_aggregate_peers(peers, index)

    # If I look at R code, p_aggregate_peers iterates over ceilings and extracts peers[[ceiling]][[index]].

    # In Python, model['peers'] is dict[ceiling][x_name].
    # If I pass 1, I can't access by key if key is name.
    # I should pass x_name to p_aggregate_peers in nca_outliers.py.

    # But first let's fix p_peers return type.

    for ceiling in model_peers:
        # We need to handle x being name or index
        # If x is int, we need to get the x-th key?
        # But dicts are unordered (mostly).
        # We should pass the name.

        current_peers = None
        if isinstance(x, str):
            if x in model_peers[ceiling]:
                current_peers = model_peers[ceiling][x]
        elif isinstance(x, int):
            # Assuming x is 1-based index from R?
            # keys = list(model_peers[ceiling].keys())
            # if len(keys) >= x:
            #    current_peers = model_peers[ceiling][keys[x-1]]
            pass

        if current_peers is not None:
            peers_list.append(current_peers)

    if not peers_list:
        return None

    peers = pd.concat(peers_list)
    # Remove duplicates based on index (names)
    peers = peers[~peers.index.duplicated(keep="first")]
    return peers


def p_get_line_peers(loop_data, intercept, slope):
    peers = p_peers(loop_data, vrs=True)
    if intercept is None or slope is None:
        return None
    return p_best_peers(peers, intercept, slope)


def p_best_peers(peers, intercept, slope):
    peers_arr = peers.values if hasattr(peers, "values") else peers
    diff = np.abs(intercept + slope * peers_arr[:, 0] - peers_arr[:, 1])

    indices = np.argsort(diff)
    peers_sorted = peers_arr[indices]
    diff_sorted = diff[indices]

    delta = abs(peers_sorted[0, 1] / 1e6)

    mask = diff_sorted < delta
    return peers_sorted[mask]
