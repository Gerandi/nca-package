import numpy as np


def p_above(loop_data, slope, intercept):
    x = loop_data["x"]
    y = loop_data["y"]
    flip_x = loop_data["flip_x"]
    flip_y = loop_data["flip_y"]

    # Upper left and lower right
    if (flip_x == flip_y) and (np.isnan(slope) or slope < 0):
        return float("nan")

    # Lower left and upper right
    if (flip_x != flip_y) and (np.isnan(slope) or slope > 0):
        return float("nan")

    # Vertical difference between observations and ceiling line
    y_c = slope * x + intercept

    if not flip_y:
        y_diff = y - y_c
    else:
        y_diff = y_c - y

    # sum(y.diff > 1e-07, na.rm=TRUE)
    # Pandas/Numpy comparison with NaN results in False, which is effectively na.rm=TRUE for counting True.
    return np.sum(y_diff > 1e-07)
