import warnings

import numpy as np
import pandas as pd

from .p_constants import P_NO_BOTTLENECK

p_bottleneck_options = ["percentage.range", "percentage.max", "actual", "percentile"]


def p_bottleneck_data(
    x, y, scope, flip_y, ceilings, bottleneck_x, bottleneck_y, steps, step_size, cutoff
):
    bn_x = p_validate_bottleneck(bottleneck_x, "x")
    bn_y = p_validate_bottleneck(bottleneck_y, "y")
    bn_x_id = p_bottleneck_id(bn_x)
    bn_y_id = p_bottleneck_id(bn_y)

    # Use the first scope vector for Y calculations
    y_scope = scope
    if isinstance(scope, list) and len(scope) > 0:
        if isinstance(scope[0], list):
            y_scope = scope[0]
        elif scope[0] is None:
            y_scope = None
        # else: scope is likely [min, max, min, max] so we use it as is

    mp, mpy = p_mp_mpy(y, y_scope, steps, step_size, bn_y_id, flip_y)

    mp = pd.DataFrame(mp)

    mp.attrs["bn.x"] = bn_x
    mp.attrs["bn.y"] = bn_y
    mp.attrs["bn.y.id"] = bn_y_id
    mp.attrs["size"] = len(x)
    mp.attrs["cutoff"] = cutoff

    bottlenecks = {}
    valid_ceilings = [c for c in ceilings if c not in P_NO_BOTTLENECK]

    for ceil in valid_ceilings:
        bottlenecks[ceil] = mp.copy()

    return {
        "bottlenecks": bottlenecks,
        "bn_x": bn_x,
        "bn_y": bn_y,
        "bn_x_id": bn_x_id,
        "bn_y_id": bn_y_id,
        "mpy": mpy,
        "cutoff": cutoff,
        "steps": steps,
    }


def p_mp_mpy(y, scope, steps, step_size, bn_y_id, flip_y):
    if isinstance(steps, int) and steps < 1:
        steps = 10
    elif isinstance(steps, list) and len(steps) == 1 and steps[0] < 1:
        steps = 10

    if bn_y_id == 3:
        mp, mpy = p_mp_mpy_actual(y, scope, steps, step_size, flip_y)
    else:
        mp, mpy = p_mp_mpy_perc(y, scope, steps, step_size, bn_y_id, flip_y)

    return mp, mpy


def p_low_high(y, scope, bn_y_id):
    if scope is None:
        if hasattr(y, "iloc") and y.ndim > 1:
            y_vals = y.iloc[:, 0]
        else:
            y_vals = y

        py_low = np.min(y_vals)
        py_high = np.max(y_vals)
    else:
        py_low = scope[2]
        py_high = scope[3]

    if bn_y_id == 2:
        py_low = 0

    return py_low, py_high


def p_sanitize_steps(steps, low, high):
    steps = sorted(steps)
    steps = np.array(steps)

    if low > steps[0]:
        warnings.warn("Bottleneck: Some steps below scope, excluded", stacklevel=2)
        steps = steps[steps >= low]

    if high < steps[-1]:
        warnings.warn("Bottleneck: Some steps above scope, excluded", stacklevel=2)
        steps = steps[steps <= high]

    return steps


def p_mp_mpy_actual(y, scope, steps, step_size, flip_y):
    py_low, py_high = p_low_high(y, scope, 3)

    if step_size is None or step_size <= 0:
        if isinstance(steps, (list, np.ndarray)) and len(steps) > 1:
            values = p_sanitize_steps(steps, py_low, py_high)
        else:
            step_count = steps[0] if isinstance(steps, list) else steps
            values = np.linspace(py_low, py_high, int(step_count) + 1)
    else:
        values = []
        value = py_low
        while value <= py_high:
            values.append(value)
            value += step_size
        values = np.array(values)
        if abs(values[-1] - py_high) > 1e-6:
            values = np.append(values, py_high)

    if flip_y:
        mpy = np.flip(values).reshape(-1, 1)
    else:
        mpy = values.reshape(-1, 1)

    return mpy, mpy


def p_mp_mpy_perc(y, scope, steps, step_size, bn_y_id, flip_y):
    py_low, py_high = p_low_high(y, scope, bn_y_id)

    if step_size is None or step_size <= 0:
        if isinstance(steps, (list, np.ndarray)) and len(steps) > 1:
            probs = p_sanitize_steps(steps, 0, 100) / 100
        else:
            step_count = steps[0] if isinstance(steps, list) else steps
            probs = np.linspace(0, 1, int(step_count) + 1)
    else:
        step = min(1000, step_size / 100)
        probs = np.arange(0, 1 + step / 1000, step)

    if bn_y_id == 4:
        y_vals = y.iloc[:, 0] if hasattr(y, "iloc") else y
        values = np.quantile(y_vals, probs)
    else:
        values = py_low + probs * (py_high - py_low)

    if flip_y:
        mpy = np.flip(values).reshape(-1, 1)
    else:
        mpy = values.reshape(-1, 1)

    mp = (100 * probs).reshape(-1, 1)

    return mp, mpy


def p_validate_bottleneck(option, x_or_y):
    if option not in p_bottleneck_options:
        warnings.warn(
            f"Bottleneck option '{option}' for {x_or_y} is not valid, using '{p_bottleneck_options[0]}'",
            stacklevel=2,
        )
        return p_bottleneck_options[0]
    return option


def p_bottleneck_id(option):
    if option not in p_bottleneck_options:
        return 1
    return p_bottleneck_options.index(option) + 1
