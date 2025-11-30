import math
import multiprocessing
import platform
import warnings

import numpy as np

# Global variable to hold the pool if we use one
_pool = None


def p_generate_title(x_name, y_name):
    return f"{x_name} - {y_name}"


def p_pretty_name(ugly_name):
    if ugly_name:
        return ugly_name.upper().replace("_", "-")
    return ugly_name


def p_is_number(number):
    if isinstance(number, list):
        if not number:
            return False
        # In R, unlist flattens. We check the first element if it's a list.
        try:
            number = number[0]
        except IndexError:
            return False

    if number is None:
        return False

    if isinstance(number, str):
        if number in ("NA", "NN"):
            return False
        # R's is.numeric returns FALSE for strings, even if they look like numbers
        return False

    try:
        if np.isnan(number):
            return False
        if np.isinf(number):
            return False
    except Exception:
        pass

    return isinstance(number, (int, float, np.number))


def p_pretty_number(ugly_number, default="", prec=3, use_spaces=False):
    if not p_is_number(ugly_number):
        return default

    if isinstance(ugly_number, list):
        ugly_number = ugly_number[0]

    # R: if (is.integer(uglyNumber) && !useSpaces)
    # In Python, 1.0 is float, 1 is int. R is looser.
    # We'll check if it's an integer type.
    if isinstance(ugly_number, int) and not use_spaces:
        return f"{ugly_number:d}"

    if prec == "auto":
        if ugly_number == 0:
            prec = 3
        else:
            prec = max(0, 3 - math.floor(math.log10(abs(ugly_number))))

    n_spaces = 0
    if use_spaces:
        n_spaces = 4 if prec == 0 else max(0, 3 - prec)

    # We hate to see -0.0
    if abs(ugly_number) < 0.1 ** max(1, prec):
        ugly_number = 0

    formatted_number = f"{ugly_number:.{prec}f}"
    return f"{formatted_number}{' ' * n_spaces}"


def p_warn_percentage_max(loop_data, bn_data):
    # We need p_bottleneck_id. It is likely in p_bottleneck.py (not yet created).
    # We will import it inside to avoid circular imports if it ends up there.
    # For now, we assume it will be available in the package.
    from .p_bottleneck import p_bottleneck_id

    # R: if (p_bottleneck_id(bn.data$bn.y) == 2 && loop.data$scope.theo[3] < 0)
    # Python: 0-based indexing for scope_theo[3] -> scope_theo[2]
    if p_bottleneck_id(bn_data["bn_y"]) == 2 and loop_data["scope_theo"][2] < 0:
        warnings.warn(
            "Using bottleneck.y with Y values < 0, results might be counterintuitive!", stacklevel=2
        )


def p_if_min_else_max(use_min, *args, na_rm=False):
    # R: dots <- c(...)
    # R's c(...) flattens.
    flat_args = []
    for arg in args:
        if isinstance(arg, (list, tuple, np.ndarray)):
            flat_args.extend(arg)
        else:
            flat_args.append(arg)

    if na_rm:
        flat_args = [
            x for x in flat_args if x is not None and not (isinstance(x, float) and np.isnan(x))
        ]

    if not flat_args:
        return None

    if use_min:
        return min(flat_args)
    return max(flat_args)


def p_is_equal(value_1, value_2):
    max_diff = min(abs(value_1), abs(value_2)) / 1e6
    diff = abs(value_1 - value_2)
    return diff <= max_diff


def p_weights(loop_data, peers):
    x = loop_data["x"]
    flip_x = loop_data["flip_x"]

    weights = []

    # peers is likely a numpy array or list of lists.
    # R: nrow(peers)
    num_rows = len(peers)

    for i in range(num_rows - 1):
        # R: peers[i + 1, 1] -> Python: peers[i + 1][0] (assuming 0-based index for 1st column)
        peer_val = peers[i + 1][0]

        if not flip_x:
            count = x < peer_val
        else:
            count = x > peer_val

        current_weight = np.sum(count) - sum(weights)
        weights.append(current_weight)

    # Add the last column
    weights.append(len(x) - sum(weights))

    return weights


def print_nca_result(x):
    from .nca_plots import p_display_plot
    from .nca_summary import p_display_summary_simple

    p_display_summary_simple(x["summaries"])
    if x.get("show_plots", False):
        for plot in x["plots"]:
            p_display_plot(plot)


def summary_nca_result(obj, columns=None):
    from .nca_output import nca_output

    if columns is not None:
        # Columns can be indexes or names
        if all(isinstance(c, str) for c in columns):
            # Filter by keys
            selected_summaries = {k: v for k, v in obj["summaries"].items() if k in columns}
        elif all(isinstance(c, int) for c in columns):
            # Filter by index
            keys = list(obj["summaries"].keys())
            selected_keys = [keys[i - 1] for i in columns if 0 < i <= len(keys)]
            selected_summaries = {k: obj["summaries"][k] for k in selected_keys}
        else:
            selected_summaries = obj["summaries"]

        if selected_summaries:
            obj["summaries"] = selected_summaries

    nca_output(obj)


def plot_nca_result(x):
    from .nca_output import nca_output

    nca_output(x, plots=True, summaries=False, bottlenecks=False)


def p_get_digits(tmp):
    def get_max_nchar(n):
        # R: nchar(sub("0+$", "", sprintf("%f", n %% 1)))
        # sprintf("%f") in R gives 6 decimal places by default.
        frac = n % 1
        s = f"{frac:.6f}"
        s = s.rstrip("0")
        if s.endswith("."):
            s += "0"  # R's "0." has length 2.
        return len(s)

    # Handle pandas Series, lists, or empty values
    if tmp is None:
        return 0
    if hasattr(tmp, "empty") and tmp.empty:
        return 0
    if hasattr(tmp, "__len__") and len(tmp) == 0:
        return 0

    # R: min(3, max(sapply(tmp, get_max_nchar) - 2))
    max_len = max(get_max_nchar(n) for n in tmp)
    return min(3, max_len - 2)


def p_accuracy(loop_data, above):
    n_observations = min(len(loop_data["x"]), len(loop_data["y"]))
    return 100 * (n_observations - above) / n_observations


def p_start_cluster(condition):
    global _pool
    if condition:
        if "windows" in platform.system().lower():
            print("Preparing the analysis, this might take a few seconds...")

        try:
            cores = multiprocessing.cpu_count()
            _pool = multiprocessing.Pool(processes=cores)
        except Exception as e:
            print(f"Failed to start cluster: {e}")
    else:
        # Do parallel, this prohibits warnings on %dopar%
        # In Python, we just don't use the pool.
        if _pool:
            _pool.close()
            _pool.join()
            _pool = None


def p_cluster_cleanup():
    global _pool
    if _pool:
        _pool.close()
        _pool.join()
        _pool = None
