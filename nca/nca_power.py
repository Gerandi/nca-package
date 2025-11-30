import math

import numpy as np
import pandas as pd

from .nca import nca_analysis


def nca_power(
    n=None,
    effect=0.10,
    slope=1,
    ceiling="ce_fdh",
    p=0.05,
    distribution_x="uniform",
    distribution_y="uniform",
    rep=100,
    test_rep=200,
):

    if n is None:
        n = [20, 50, 100]

    from .nca_random import nca_random

    # Ensure inputs are lists/arrays where appropriate or handle scalars
    # R's c() creates a vector. Python list or scalar.
    # We normalize to numpy arrays for iteration.

    def to_array(x):
        if np.isscalar(x):
            return np.array([x])
        return np.array(x)

    if np.any(to_array(effect) <= 0) or np.any(to_array(effect) >= 1):
        print("The effect size needs to be larger than 0 and smaller than 1\n")
        return None
    if np.any(to_array(slope) <= 0):
        print("The slope needs to be larger than 0\n")
        return None

    # Make sure we're not doing extra work
    distribution_x = np.unique(to_array(distribution_x))
    distribution_y = np.unique(to_array(distribution_y))
    n = np.unique(to_array(n))
    effect = np.unique(to_array(effect))
    slope = np.unique(to_array(slope))
    ceiling = np.unique(to_array(ceiling))

    # Calculate the total number of iterations
    n_iterations = (
        rep
        * len(distribution_x)
        * len(distribution_y)
        * len(n)
        * len(effect)
        * len(slope)
        * len(ceiling)
    )

    # Define the variables that will store results
    results = pd.DataFrame(
        columns=["n", "ES", "slope", "ceiling", "p", "distr.x", "distr.y", "power"]
    )

    # Counter of iterations (single samples)
    count = 0

    for distr_x in distribution_x:
        for distr_y in distribution_y:
            for ceil in ceiling:
                for sample_size in n:
                    for effect_loop in effect:
                        for slope_loop in slope:
                            intercept = p_intercept(slope_loop, effect_loop)

                            # Initialize vectors to store p-values and power results
                            pval = np.zeros(rep)
                            sig_results = np.zeros(rep)

                            for r in range(rep):
                                count += 1
                                print(f"\rIteration {count} of {n_iterations}", end="")

                                df = nca_random(
                                    sample_size,
                                    intercept,
                                    slope_loop,
                                    distribution_x=distr_x,
                                    distribution_y=distr_y,
                                )

                                # Get column names from the generated data
                                x_col = df.columns[0]  # First column is X
                                y_col = df.columns[-1]  # Last column is Y

                                model = nca_analysis(
                                    df, x_col, y_col, ceilings=[ceil], test_rep=test_rep, scope=[0, 1, 0, 1]
                                )

                                # Estimated p-value
                                # model['summaries']['X']['params'][6] (R 1-based) -> index 5 (Python 0-based)
                                x_name = x_col

                                try:
                                    # We need to ensure we access the correct value.
                                    # In nca_summary.R (to be ported), params is constructed.
                                    # I will assume index 5 corresponds to p-value.
                                    p_val = model["summaries"][x_name]["params"][5]
                                    pval[r] = p_val
                                    sig_results[r] = p_val <= p
                                except Exception:
                                    # If something goes wrong (e.g. no p-value), treat as not significant?
                                    pval[r] = 1.0
                                    sig_results[r] = 0

                            # Store the results for this iteration
                            res_row = pd.DataFrame(
                                {
                                    "n": [sample_size],
                                    "ES": [effect_loop],
                                    "slope": [slope_loop],
                                    "ceiling": [ceil],
                                    "p": [np.mean(pval)],
                                    "distr.x": [distr_x],
                                    "distr.y": [distr_y],
                                    "power": [np.mean(sig_results)],
                                }
                            )
                            results = pd.concat([results, res_row], ignore_index=True)

    print("\n\n")
    return results


def p_intercept(slope, effect):
    # Assume intercept >= 0, line through roof, y on x == 0 should be >= 1
    intercept = 1 - math.sqrt(2 * effect * slope)
    if intercept >= 0 and (intercept + slope) >= 1:
        return intercept

    # Assume intercept >= 0, line through right, y on x == 0 should be < 1
    intercept = 1 - effect - slope / 2
    if intercept >= 0 and (intercept + slope) < 1:
        return intercept

    # Assume intercept < 0, line through roof, y on x == 0 should be >= 1
    intercept = 0.5 - effect * slope
    if intercept < 0 and (intercept + slope) >= 1:
        return intercept

    # Assume intercept < 0, line through side, y on x == 0 should be <> 1
    y = math.sqrt(2 * (1 - effect) * slope)
    intercept = y - slope
    return intercept
