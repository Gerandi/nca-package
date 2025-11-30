import math

import numpy as np
from scipy.stats import iqr

from .p_utils import p_if_min_else_max


def p_columns(loop_data, is_confidence):
    flip_x = loop_data["flip_x"]
    flip_y = loop_data["flip_y"]

    x = loop_data["x"]
    y = loop_data["y"]

    if flip_x:
        indices = np.argsort(x)[::-1]
    else:
        indices = np.argsort(x)

    x_sorted = x.iloc[indices].values if hasattr(x, "iloc") else x[indices]
    y_sorted = y.iloc[indices].values if hasattr(y, "iloc") else y[indices]

    columns = p_initial_columns(x_sorted, y_sorted, loop_data, flip_x, flip_y)
    columns = p_merge_columns(columns, flip_x, flip_y)

    if is_confidence:
        columns = p_bootstrap(y_sorted, columns, loop_data)
        columns = p_con_ce(columns, loop_data)

    return columns


def p_initial_columns(x, y, loop_data, flip_x, flip_y):
    # x is sorted
    # pd.unique preserves order, np.unique sorts.
    # Since x is sorted (asc or desc), pd.unique will return unique values in that order.
    import pandas as pd

    ux = pd.unique(x)

    k = len(ux)
    if k <= 1:
        colwidth = np.array([])
    else:
        colwidth = ux[1:] - ux[:-1]

    if k > 1:
        boundaries = ux[:-1] + colwidth / 2
    else:
        boundaries = np.array([])

    scope_theo = loop_data["scope_theo"]

    if not flip_x:
        if k > 1:
            first = min(scope_theo[0], ux[0] - colwidth[0] / 2)
            second = max(scope_theo[1], ux[-1] + colwidth[-1] / 2)
        else:
            first = scope_theo[0]
            second = scope_theo[1]
    else:
        if k > 1:
            first = max(scope_theo[1], ux[0] - colwidth[0] / 2)
            second = min(scope_theo[0], ux[-1] + colwidth[-1] / 2)
        else:
            first = scope_theo[1]
            second = scope_theo[0]

    boundaries = np.concatenate(([first], boundaries, [second]))

    columns = np.zeros((5, k))

    for i in range(k):
        b1 = boundaries[i]
        b2 = boundaries[i + 1]

        if not flip_x:
            mask = (x >= b1) & (x < b2)
            # Handle last bin edge case if needed, but usually boundaries cover it.
            # If x includes max value and b2 > max value, it's fine.
        else:
            mask = (x <= b1) & (x > b2)

        count = np.sum(mask)
        columns[0, i] = count
        columns[1, i] = b1
        columns[2, i] = b2

        if count == 0:
            y_max = scope_theo[2] if flip_y else scope_theo[3]
            x_max = scope_theo[1] if flip_x else scope_theo[0]
        else:
            y_subset = y[mask]
            x_subset = x[mask]

            if flip_y:
                y_max = np.min(y_subset)
            else:
                y_max = np.max(y_subset)

            indices = np.where(y_subset == y_max)[0]
            if not flip_x:
                idx = indices[0]
            else:
                idx = indices[-1]
            x_max = x_subset[idx]

        columns[3, i] = x_max
        columns[4, i] = y_max

    return columns


def p_merge_columns(columns, flip_x, flip_y):
    total_count = np.sum(columns[0, :])
    min_count = round(math.sqrt(total_count / 2))

    while True:
        if columns.shape[1] <= 1:
            break

        if np.min(columns[0, :]) >= min_count:
            break

        counts = columns[0, :]
        below_min = counts[counts < min_count]
        max_below_min = np.max(below_min)

        min_col = np.where(counts == max_below_min)[0][0]

        if min_col in (0, columns.shape[1] - 1):
            min_neighbor = min_col + (1 if min_col == 0 else -1)
        else:
            neighbor_cols = [min_col - 1, min_col + 1]
            values = counts[neighbor_cols]

            if np.min(values) >= min_count:
                min_neighbor = min_col + (1 if values[0] > values[1] else -1)
            else:
                min_neighbor = min_col + (1 if values[1] > values[0] else -1)

        columns = p_merge_2_columns(columns, min_col, min_neighbor, flip_x, flip_y)

    return columns


def p_merge_2_columns(columns, col1, col2, flip_x, flip_y):
    columns[0, col1] += columns[0, col2]

    if col1 > col2:
        columns[1, col1] = columns[1, col2]
    else:
        columns[2, col1] = columns[2, col2]

    val1 = columns[4, col1]
    val2 = columns[4, col2]

    update = False
    if not flip_y and val1 < val2:
        update = True
    elif flip_y and val1 > val2:
        update = True
    elif val1 == val2:
        x1 = columns[3, col1]
        x2 = columns[3, col2]
        best_x = p_if_min_else_max(not flip_x, x1, x2)
        columns[3, col1] = best_x

    if update:
        columns[3, col1] = columns[3, col2]
        columns[4, col1] = columns[4, col2]

    columns = np.delete(columns, col2, axis=1)

    return columns


def p_bootstrap(y, columns, loop_data):
    conf = loop_data["conf"]
    conf_rep = loop_data["conf_rep"]
    flip_y = loop_data["flip_y"]

    y_min = np.min(y) if not flip_y else np.max(y)
    y_range = (np.max(y) - np.min(y)) if not flip_y else (np.min(y) - np.max(y))

    if y_range == 0:
        y_norm = y - y_min  # All zeros
    else:
        y_norm = (y - y_min) / y_range

    start = 0
    for col in range(columns.shape[1]):
        count = int(columns[0, col])
        end = start + count

        z = y_norm[start:end]
        ci = p_bootstrap_column(z, count, conf, conf_rep)

        columns[4, col] = (ci * y_range) + y_min
        start = end

    return columns


def p_bootstrap_column(z, n, conf, nrep):
    n2 = n * 2
    zeta_hat = np.max(z)
    zeta_star = np.zeros(nrep)

    zr = np.concatenate((z, 2 * zeta_hat - z))

    hr = p_dpik_safe(zr)
    h = (2**0.2) * hr

    for b in range(nrep):
        ind = np.floor(np.random.uniform(0, 1, n) * n2).astype(int)
        zs = zr[ind]

        t1 = zs + h * np.random.normal(0, 1, n)
        zss = np.where(t1 <= zeta_hat, t1, 2 * zeta_hat - t1)

        t2 = np.mean(zs)

        v_zss = np.var(zss, ddof=1)
        if v_zss == 0:
            zsss = zss
        else:
            zsss = t2 + (zss - t2) / math.sqrt(1 + (h**2) / v_zss)

        zeta_star[b] = np.max(zsss)

    g_star = (n / zeta_hat) * (zeta_hat - zeta_star)
    qq = np.quantile(g_star, conf)
    ci = zeta_hat / (1 - qq / n)

    return ci


def p_dpik_safe(x):
    n = len(x)
    sd = np.std(x, ddof=1)
    iqr_val = iqr(x)

    if iqr_val == 0:
        scale = sd
    else:
        scale = min(sd, iqr_val / 1.34)

    if scale == 0:
        scale = 1.0

    h = 0.9 * scale * (n**-0.2)
    return h


def p_con_ce(columns, loop_data):
    flip_x = loop_data["flip_x"]
    flip_y = loop_data["flip_y"]
    scope_theo = loop_data["scope_theo"]
    peers = loop_data["ce_fdh_peers"]

    columns[4, :] = np.minimum(columns[4, :], scope_theo[3])
    columns[4, :] = np.maximum(columns[4, :], scope_theo[2])

    for col in range(columns.shape[1]):
        x_bound = columns[2, col]

        if flip_x:
            indices = np.where(peers.iloc[:, 0].values >= x_bound)[0]
        else:
            indices = np.where(peers.iloc[:, 0].values <= x_bound)[0]

        if len(indices) > 0:
            peer_max_idx = np.max(indices)
            peer_y = peers.iloc[peer_max_idx, 1]

            if flip_y:
                columns[4, col] = min(columns[4, col], peer_y)
            else:
                columns[4, col] = max(columns[4, col], peer_y)

    return columns


def p_conf_line(columns):
    x_points = []
    y_points = []
    for col in range(columns.shape[1]):
        x_points.extend([columns[1, col], columns[2, col]])
        y_points.extend([columns[4, col], columns[4, col]])

    return [x_points, y_points]
