import numpy as np
import pandas as pd
from scipy.stats import truncnorm

p_errors = {
    "n": "n should be an integer > 1!",
    "combination": "The combination of slope and intercept does not provide points in the [(0, 0), (1, 1)] area!",
    "length": "The length of the slopes and intercepts should be equal!",
    "corner_23": "Upward slope can not provide empty corners 2 (upper right) or 3 (lower left)!",
    "corner_14": "Downward slope can not provide empty corners 1 (upper left) or 4 (lower right)!",
    "distribution": "The distribution types need to be 'uniform' or 'normal'!",
}


def nca_random(
    n,
    intercepts,
    slopes,
    corner=1,
    distribution_x="uniform",
    distribution_y="uniform",
    mean_x=0.5,
    mean_y=0.5,
    sd_x=0.2,
    sd_y=0.2,
):

    # Validate inputs
    tmp = p_validate_inputs(n, intercepts, slopes, corner, distribution_x, distribution_y)
    error = tmp["error"]
    intercepts = tmp["intercepts"]
    slopes = tmp["slopes"]

    if error is not None:
        raise ValueError(p_errors[error])

    # Prepare data structure
    cols = []
    if len(slopes) == 1:
        cols.append("X")
    else:
        for i in range(len(slopes)):
            cols.append(f"X{i+1}")
    cols.append("Y")

    data_list = []

    # Generate n points
    # We can optimize this loop but rejection sampling is inherently iterative or batch-based.
    # Given n is usually small (20-100 in defaults), loop is fine.

    for _ in range(int(n)):
        while True:
            y_value = p_value(distribution_y, mean_y, sd_y)

            row_vals = []
            all_good = True

            for idx, slope in enumerate(slopes):
                x_value = p_value(distribution_x, mean_x, sd_x)
                row_vals.append(x_value)

                intercept = intercepts[idx]

                val_line = intercept + slope * x_value

                if corner in [1, 2]:
                    if y_value >= min(1, val_line):
                        all_good = False

                if corner in [3, 4]:
                    if y_value <= max(0, val_line):
                        all_good = False

            if all_good:
                row_vals.append(y_value)
                data_list.append(row_vals)
                break

    df = pd.DataFrame(data_list, columns=cols)

    # Sort by Y
    df = df.sort_values(by="Y").reset_index(drop=True)
    df.index = df.index + 1

    # Attributes
    df.attrs["distribution.x"] = distribution_x
    df.attrs["distribution.y"] = distribution_y
    if distribution_x == "normal":
        df.attrs["mean.x"] = mean_x
        df.attrs["sd.x"] = sd_x
    if distribution_y == "normal":
        df.attrs["mean.y"] = mean_y
        df.attrs["sd.y"] = sd_y

    return df


def p_value(distribution, mean, sd):
    if distribution == "uniform":
        return np.random.uniform(0, 1)
    # rtruncnorm(1, a=0, b=1, mean=mean, sd=sd)
    # scipy truncnorm takes a, b as standardized limits
    a, b = (0 - mean) / sd, (1 - mean) / sd
    return truncnorm.rvs(a, b, loc=mean, scale=sd)


def p_validate_inputs(n, intercepts, slopes, corner, distribution_x, distribution_y):
    if n < 1 or n != round(n):
        return {"error": "n", "intercepts": intercepts, "slopes": slopes}

    # Normalize intercepts/slopes to lists
    if np.isscalar(intercepts):
        intercepts = [intercepts]
    if np.isscalar(slopes):
        slopes = [slopes]

    intercepts = list(intercepts)
    slopes = list(slopes)

    if len(intercepts) == 1 and len(slopes) > 1:
        intercepts = intercepts * len(slopes)
    elif len(intercepts) > 1 and len(slopes) == 1:
        slopes = slopes * len(intercepts)

    if len(intercepts) != len(slopes):
        return {"error": "length", "intercepts": intercepts, "slopes": slopes}

    for idx, slope in enumerate(slopes):
        intercept = intercepts[idx]

        cond_1 = slope > 0 and (intercept >= 1 or (intercept + slope) <= 0)
        cond_2 = slope < 0 and (intercept <= 0 or (intercept + slope) >= 1)
        cond_3 = slope == 0 and (intercept <= 0 or intercept >= 1)

        if cond_1 or cond_2 or cond_3:
            return {"error": "combination", "intercepts": intercepts, "slopes": slopes}

        if slope > 0 and corner in [2, 3]:
            return {"error": "corner_23", "intercepts": intercepts, "slopes": slopes}

        if slope < 0 and corner in [1, 4]:
            return {"error": "corner_14", "intercepts": intercepts, "slopes": slopes}

    valid_dist = ["uniform", "normal"]
    if distribution_x not in valid_dist or distribution_y not in valid_dist:
        return {"error": "distribution", "intercepts": intercepts, "slopes": slopes}

    return {"error": None, "intercepts": intercepts, "slopes": slopes}
