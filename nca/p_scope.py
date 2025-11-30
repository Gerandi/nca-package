import warnings

import numpy as np


def p_scope(x, scope):
    if isinstance(x, str):
        num_x = 1
    elif hasattr(x, "columns"):
        num_x = len(x.columns)
    else:
        num_x = len(x)

    if scope is None:
        return [None] * num_x

    if isinstance(scope, (list, np.ndarray)):
        if len(scope) == 4 and all(isinstance(v, (int, float, np.number)) for v in scope):
            return p_validate_scope_vector(num_x, scope)

        if len(scope) == 4 * num_x and all(isinstance(v, (int, float, np.number)) for v in scope):
            return p_validate_scope_vector(num_x, scope)

        return p_validate_scope_list(num_x, scope)

    return [None] * num_x


def p_validate_scope_list(num_x, scope):
    if len(scope) != num_x:
        raise ValueError(
            "The length of scope needs to be equal to the length of x \n"
            "       or a single vector with 4 values!\n\n"
        )

    for s in scope:
        if len(s) != 4:
            raise ValueError("The length of each scope segment needs to be 4!\n\n")

    first_y = scope[0][2:4]
    for s in scope:
        current_y = s[2:4]

        match = True
        for j in range(2):
            v1 = current_y[j]
            v2 = first_y[j]
            if np.isnan(v1) and np.isnan(v2):
                continue
            if v1 != v2:
                match = False
                break

        if not match:
            warnings.warn("Scope is using different Ymin and Ymax for the same Y")
            break

    return scope


def p_validate_scope_vector(num_x, scope):
    if len(scope) == 4:
        return [list(scope)] * num_x

    if len(scope) == 4 * num_x:
        return [list(scope[i * 4 : (i + 1) * 4]) for i in range(num_x)]

    raise ValueError(
        "The length of scope needs to be equal to the length of x \n"
        "       or a single vector with 4 values!\n\n"
    )
