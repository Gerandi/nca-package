import numpy as np
import pandas as pd

from .p_constants import EPSILON
from .p_utils import p_if_min_else_max, p_pretty_number


def p_bottleneck(loop_data, bn_data, slope, intercept):
    if bn_data is None:
        return None

    theo = loop_data["scope_theo"]
    flip_x = loop_data["flip_x"]
    precision_x = 1 if bn_data["bn_x_id"] in [1, 2, 4] else 3

    if np.isnan(intercept) or np.isnan(slope):
        mpx = p_mpx_single_peer(bn_data, theo, flip_x)
    else:
        mpx = (bn_data["mpy"] - intercept) / slope
        mpx = p_edge_cases(mpx, bn_data, theo, flip_x)

    cases = p_cases(loop_data, mpx)

    nn_value = p_nn_value(mpx, loop_data, bn_data)
    na_value = p_na_value(mpx, loop_data, bn_data)

    mpx_actual = mpx.copy()
    mpx = p_transform_mpx(loop_data, mpx, bn_data["bn_x_id"])
    pretty_mpx = p_pretty_mpx(loop_data, mpx, nn_value, na_value, precision_x)

    df = pd.DataFrame(pretty_mpx, columns=[loop_data["names"][loop_data["idx"]]])
    df.attrs["mpx_actual"] = mpx_actual

    if bn_data["bn_x_id"] == 4:
        df.attrs["cases"] = cases

    return df


def p_bottleneck_ce(loop_data, bn_data, peers, type_):
    if bn_data is None:
        return None

    theo = loop_data["scope_theo"]
    flip_x = loop_data["flip_x"]
    flip_y = loop_data["flip_y"]
    precision_x = 1 if bn_data["bn_x_id"] in [1, 2, 4] else 3

    if type_ == "fdh":
        mpx = p_bottleneck_fdh(bn_data, peers, flip_y)
    elif type_ == "vrs":
        mpx = p_bottleneck_vrs(bn_data, peers, flip_y)
    else:
        # Default to fdh if unknown type
        mpx = p_bottleneck_fdh(bn_data, peers, flip_y)

    cases = p_cases(loop_data, mpx)

    mpx = p_edge_cases(mpx, bn_data, theo, flip_x, False)
    nn_value = p_nn_value(mpx, loop_data, bn_data)
    na_value = p_na_value(mpx, loop_data, bn_data)

    mpx_actual = mpx.copy()
    mpx = p_transform_mpx(loop_data, mpx, bn_data["bn_x_id"])
    pretty_mpx = p_pretty_mpx(loop_data, mpx, nn_value, na_value, precision_x)

    df = pd.DataFrame(pretty_mpx, columns=[loop_data["names"][loop_data["idx"]]])
    df.attrs["mpx_actual"] = mpx_actual

    if bn_data["bn_x_id"] == 4:
        df.attrs["cases"] = cases

    return df


def p_bottleneck_fdh(bn_data, peers, flip_y):
    mpy = bn_data["mpy"]
    mpx = np.full((len(mpy), 1), np.nan)
    x_peers = peers.iloc[:, 0].values
    y_peers = peers.iloc[:, 1].values

    for j in range(len(mpy)):
        if flip_y:
            indices = np.where(y_peers < (mpy[j, 0] + EPSILON))[0]
        else:
            indices = np.where(y_peers > (mpy[j, 0] - EPSILON))[0]

        if len(indices) == 0:
            mpx[j, 0] = np.nan
        else:
            mpx[j, 0] = x_peers[indices[0]]

    return mpx


def p_bottleneck_vrs(bn_data, peers, flip_y):
    mpy = bn_data["mpy"]
    mpx = np.full((len(mpy), 1), np.nan)
    x_peers = peers.iloc[:, 0].values
    y_peers = peers.iloc[:, 1].values
    peers_arr = peers.values

    def calculate_x(y, peers_data, index):
        p1 = peers_data[index - 1]
        p2 = peers_data[index]
        return p1[0] + (y - p1[1]) * (p1[0] - p2[0]) / (p1[1] - p2[1])

    for j in range(len(mpy)):
        if flip_y:
            indices = np.where(y_peers < (mpy[j, 0] + EPSILON))[0]
        else:
            indices = np.where(y_peers > (mpy[j, 0] - EPSILON))[0]

        if len(indices) == 0:
            mpx[j, 0] = np.nan
        else:
            idx = indices[0]
            if idx == 0:
                mpx[j, 0] = x_peers[idx]
            else:
                mpx[j, 0] = calculate_x(mpy[j, 0], peers_arr, idx)

    return mpx


def p_mpx_single_peer(bn_data, theo, flip_x):
    mpy = bn_data["mpy"]
    cutoff = bn_data["cutoff"]

    if cutoff == 0:
        mpx = np.full((len(mpy), 1), np.inf)
    else:
        val = theo[1] if flip_x else theo[0]
        mpx = np.full((len(mpy), 1), val)

    return mpx


def p_transform_mpx(loop_data, mpx, bn_x_id):
    flip_x = loop_data["flip_x"]
    theo = loop_data["scope_theo"]

    if bn_x_id == 1:
        mpx = 100 * (mpx - theo[0]) / (theo[1] - theo[0])
    elif bn_x_id == 2:
        mpx = 100 * mpx / theo[1]
    elif bn_x_id == 4:
        x_sorted = np.sort(loop_data["x"])
        if flip_x:
            x_ref = np.sort(-loop_data["x"])
            tmp = -mpx - EPSILON
        else:
            x_ref = x_sorted
            tmp = mpx - EPSILON

        indices = np.searchsorted(x_ref, tmp.flatten(), side="right")
        percentile = indices / len(x_ref)
        mpx = (100 * percentile).reshape(-1, 1)
        mpx[mpx == 0] = np.inf

    return mpx


def p_nn_value(mpx, loop_data, bn_data):
    if bn_data["cutoff"] in [0, 1]:
        return "NN"

    flip_x = loop_data["flip_x"]
    theo = loop_data["scope_emp"]
    nn_value = p_if_min_else_max(not flip_x, mpx, na_rm=True)

    return p_transform_value(loop_data, nn_value, theo, bn_data["bn_x_id"])


def p_na_value(mpx, loop_data, bn_data):
    if bn_data["cutoff"] == 0:
        return "NA"

    flip_x = loop_data["flip_x"]
    theo = loop_data["scope_theo"]
    na_value = p_if_min_else_max(flip_x, mpx, na_rm=True)

    return p_transform_value(loop_data, na_value, theo, bn_data["bn_x_id"])


def p_transform_value(loop_data, value, theo, bn_x_id):
    if value is None or isinstance(value, str):
        return value

    if bn_x_id == 1:
        value = 100 * (value - theo[0]) / (theo[1] - theo[0])
    elif bn_x_id == 2:
        value = 100 * value / theo[1]
    elif bn_x_id == 4:
        x_sorted = np.sort(loop_data["x"])
        idx = np.searchsorted(x_sorted, value, side="right")
        value = 100 * idx / len(x_sorted)

    return value


def p_pretty_mpx(loop_data, mpx, nn_value, na_value, precision_x):
    """Format mpx values for display.

    Note: loop_data kept for API compatibility.
    """
    _ = loop_data  # API compatibility

    def subst(x):
        if np.isnan(x):
            return p_pretty_number(na_value, str(na_value), prec=precision_x)
        if np.isinf(x):
            nn_val = nn_value
            if isinstance(nn_val, float) and np.isinf(nn_val):
                nn_val = "NN"
            return p_pretty_number(nn_val, str(nn_val), prec=precision_x)
        return p_pretty_number(x, prec=precision_x)

    flat_mpx = mpx.flatten()
    res = [subst(x) for x in flat_mpx]
    return np.array(res).reshape(-1, 1)


def p_edge_cases(mpx, bn_data, theo, flip_x, use_epsilon=False):
    tmp = EPSILON if use_epsilon else 0

    mask_low = mpx < (theo[0] + EPSILON)
    mpx[mask_low] = np.nan if flip_x else -np.inf

    if bn_data["cutoff"] == 0:
        mask_high = mpx > (theo[1] - tmp)
        mpx[mask_high] = np.inf if flip_x else np.nan
    elif bn_data["cutoff"] == 1:
        mask_high = mpx > (theo[1] - tmp)
        mpx[mask_high] = theo[1]

    return mpx


def p_cases(loop_data, mpx):
    x = loop_data["x"]
    flip_x = loop_data["flip_x"]
    mpx_cases = mpx + (EPSILON if flip_x else -EPSILON)

    x_sorted = np.sort(x)
    indices = np.searchsorted(x_sorted, mpx_cases.flatten(), side="right")
    cases = indices
    return cases.reshape(-1, 1)


def p_bottleneck_id(name):
    if name == "percentage.range":
        return 1
    if name == "percentage.max":
        return 2
    if name == "percentile":
        return 4
    return 3
