import warnings

import numpy as np
import pandas as pd

# Global variable to track scope warnings
scope_warnings_nca = False


def p_create_loop_data(x, y, scope, flip_x, flip_y, id_x, qr_tau):
    global scope_warnings_nca

    # x is DataFrame, y is Series or DataFrame
    if isinstance(x, pd.Series):
        x_name = x.name
        x_col = x
    else:
        x_name = x.columns[id_x]
        x_col = x.iloc[:, id_x]

    y_name = y.name if isinstance(y, pd.Series) else y.columns[0]
    names = [x_name, y_name]

    # scope is list of scopes or None
    current_scope = scope[id_x] if scope is not None else None

    current_flip_x = flip_x[id_x]

    # Remove not complete cases
    # Create a temporary DataFrame to handle alignment and dropna
    df = pd.DataFrame({"x": x_col, "y": y})
    df = df.dropna()

    x_clean = df["x"]
    y_clean = df["y"]

    # Define the scope params
    scope_emp = [np.min(x_clean), np.max(x_clean), np.min(y_clean), np.max(y_clean)]

    if current_scope is None:
        scope_theo = scope_emp
    else:
        s = current_scope

        # min_x
        v1 = [scope_emp[0], s[0], s[1]]
        v1 = [v for v in v1 if not np.isnan(v)]
        min_x = np.min(v1)

        # max_x
        v2 = [scope_emp[1], s[0], s[1]]
        v2 = [v for v in v2 if not np.isnan(v)]
        max_x = np.max(v2)

        # min_y
        v3 = [scope_emp[2], s[2], s[3]]
        v3 = [v for v in v3 if not np.isnan(v)]
        min_y = np.min(v3)

        # max_y
        v4 = [scope_emp[3], s[2], s[3]]
        v4 = [v for v in v4 if not np.isnan(v)]
        max_y = np.max(v4)

        scope_theo = [min_x, max_x, min_y, max_y]

        s_x_sorted = np.sort([s[0], s[1]])
        t_x_sorted = np.sort([scope_theo[0], scope_theo[1]])

        s_y_sorted = np.sort([s[2], s[3]])
        t_y_sorted = np.sort([scope_theo[2], scope_theo[3]])

        if not np.array_equal(s_x_sorted, t_x_sorted) or not np.array_equal(s_y_sorted, t_y_sorted):
            if not scope_warnings_nca:
                scope_warnings_nca = True
                warnings.warn("Theorectical scope has been adjusted to include all observations")

    scope_area = (scope_theo[1] - scope_theo[0]) * (scope_theo[3] - scope_theo[2])

    return {
        "x": x_clean,
        "y": y_clean,
        "idx": id_x,
        "scope_emp": scope_emp,
        "scope_theo": scope_theo,
        "scope_area": scope_area,
        "names": names,
        "flip_x": current_flip_x,
        "flip_y": flip_y,
        "qr_tau": qr_tau,
    }
