import warnings

import pandas as pd

from . import p_loop_data
from .p_constants import CEILINGS


def p_validate_clean(data, x, y, outliers=False):
    """Validate and clean input data.

    Note: outliers parameter is kept for API compatibility with R version.
    """
    _ = outliers  # API compatibility
    if not isinstance(y, str) and len(y) != 1:
        raise ValueError("Dependent variable can only be a single column!\n\n")

    data_clean = data.copy()
    for col in data_clean.columns:
        data_clean[col] = pd.to_numeric(data_clean[col], errors="coerce")

    p_loop_data.scope_warnings_nca = False

    return {"x": data_clean[x], "y": data_clean[y]}


def p_validate_ceilings(methods):
    if methods is None:
        methods = []

    # Handle string input (convert to list)
    if isinstance(methods, str):
        methods = [methods]

    valid_methods = [m for m in methods if m.lower() in CEILINGS]

    if len(valid_methods) == 0:
        warnings.warn("Invalid CEILINGS, using ols, ce_fdh and cr_fdh")
        return ["ols", "ce_fdh", "cr_fdh"]
    if len(valid_methods) != len(methods):
        invalid = set(methods) - set(valid_methods)
        for diff in invalid:
            warnings.warn(f"Ignoring invalid ceiling(s) '{diff}'")

    return valid_methods


def p_validate_flipx(x, flip_x):
    num_x = len(x.columns) if hasattr(x, "columns") else len(x)

    if isinstance(flip_x, bool):
        return [flip_x] * num_x

    if len(flip_x) != num_x:
        if len(flip_x) == 1:
            return [flip_x[0]] * num_x
        raise ValueError(
            "The length of 'flip.x' needs to be equal to the length of x "
            "or a single Boolean!\n"
        )

    return flip_x


def p_validate_corner(x, corner):
    num_x = len(x.columns) if hasattr(x, "columns") else len(x)

    if isinstance(corner, int):
        corner = [corner]

    if len(corner) != num_x:
        if len(corner) == 1:
            corner = [corner[0]] * num_x
        else:
            raise ValueError(
                "The length of 'corner' needs to be equal to the length of x or a integer 1:4!\n"
            )

    if not all(c in [1, 2, 3, 4] for c in corner):
        raise ValueError("All corners must be an integer between 1 and 4 !\n")

    all_upper = all(c in [1, 2] for c in corner)
    all_lower = all(c in [3, 4] for c in corner)

    if not all_upper and not all_lower:
        raise ValueError(
            "All corners need to be in the upper half (1 and 2) \n       "
            "or the lower half (3 and 4).\n       "
            "You can not mix upper and lower !\n"
        )

    return corner
