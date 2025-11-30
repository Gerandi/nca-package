import math
import multiprocessing
import warnings

from .nca_plots import p_plot
from .nca_summary import p_summary
from .nca_tests import p_test, p_test_time
from .p_bottleneck_table import p_bottleneck_data
from .p_ceiling import p_nca_wrapper
from .p_constants import P_NO_BOTTLENECK
from .p_loop_data import p_create_loop_data
from .p_scope import p_scope
from .p_utils import p_cluster_cleanup, p_start_cluster, p_warn_percentage_max
from .p_validate import (
    p_validate_ceilings,
    p_validate_clean,
    p_validate_corner,
    p_validate_flipx,
)


def nca(data, x, y, ceilings=None):
    if ceilings is None:
        ceilings = ["ols", "ce_fdh", "cr_fdh"]
    if ceilings is None:
        ceilings = ["ols", "ce_fdh", "cr_fdh"]
    model = nca_analysis(data, x, y, ceilings=ceilings)
    model["show_plots"] = True

    if "bottlenecks" in model:
        del model["bottlenecks"]
    if "tests" in model:
        del model["tests"]
    if "test_time" in model:
        del model["test_time"]

    return model


def nca_analysis(
    data,
    x,
    y,
    ceilings=None,
    corner=None,
    flip_x=False,
    flip_y=False,
    scope=None,
    bottleneck_x="percentage.range",
    bottleneck_y="percentage.range",
    steps=10,
    step_size=None,
    cutoff=0,
    qr_tau=0.95,
    effect_aggregation=1,
    test_rep=0,
    test_p_confidence=0.95,
    test_p_threshold=0.05,
):

    if ceilings is None:
        ceilings = ["ols", "ce_fdh", "cr_fdh"]

    # Cleans up any cluster registration
    p_cluster_cleanup()

    # Validate and clean data
    cleaned = p_validate_clean(data, x, y)
    data_x = cleaned["x"]
    data_y = cleaned["y"]

    # Validate ceiling types
    ceilings = p_validate_ceilings(ceilings)

    # Overrule flip.x and flip.y if corners is defined
    if corner is not None:
        corner = p_validate_corner(x, corner)
        if flip_x or flip_y:
            warnings.warn("Ignoring 'flip.x' and 'flip.y': 'corner' is defined", stacklevel=2)

        flip_y = all(c in [3, 4] for c in corner)
        flip_x = [c in [2, 4] for c in corner]

    # Always validate flip.x and flip.y
    flip_x = p_validate_flipx(x, flip_x)
    flip_y = bool(flip_y)

    # Validate scope
    scope = p_scope(x, scope)

    # Validate effect size aggregation
    if isinstance(effect_aggregation, int):
        effect_aggregation = [effect_aggregation]

    effect_aggregation = list(set([2, 3, 4]) & set(effect_aggregation))
    if len(effect_aggregation) > 0:
        total = ", ".join(map(str, [1] + effect_aggregation))
        warnings.warn(f"Using corners {total} for effect_aggregation", stacklevel=2)

    # Data object for bottlenecks
    bn_data = p_bottleneck_data(
        data_x,
        data_y,
        scope,
        flip_y,
        ceilings,
        bottleneck_x,
        bottleneck_y,
        steps,
        step_size,
        cutoff,
    )

    # Data for tests
    # Assuming data_y is a pandas DataFrame or similar, len() gives rows.
    n_rows = len(data_y)

    # In R, nrow(data.x) is used for factorial check.
    if n_rows < 16 and test_rep > math.factorial(n_rows):
        test_rep = math.factorial(n_rows)
        print(f"\nLowered test.rep to {test_rep} as it can not be larger than N!\n\n")

    test_params = {
        "rep": test_rep,
        "p_confidence": test_p_confidence,
        "p_threshold": test_p_threshold,
    }

    # Create cluster for parallisation if needed
    # We need number of columns in data_x.
    if hasattr(data_x, "shape"):
        if len(data_x.shape) > 1:
            num_vars = data_x.shape[1]
        else:
            num_vars = 1
    else:
        num_vars = len(data_x[0])

    condition = len(ceilings) * num_vars * test_rep > 6000
    p_start_cluster(multiprocessing.cpu_count() > 1 and condition)

    # Create output lists
    plots = {}
    summaries = {}
    tests = {}
    peers = {}
    test_time = 0

    # Loop the independent varaibles
    for id_x in range(num_vars):
        loop_data = p_create_loop_data(data_x, data_y, scope, flip_x, flip_y, id_x, qr_tau)
        loop_data["conf"] = test_p_confidence
        p_warn_percentage_max(loop_data, bn_data)
        x_name = loop_data["names"][0]  # Index 0 is always the X variable name

        # We need this for the 'FIT' number, regardless of user preference
        analisys_ce_fdh = p_nca_wrapper("ce_fdh", loop_data, bn_data, effect_aggregation)
        loop_data["ce_fdh_ceiling"] = analisys_ce_fdh["ceiling"]
        loop_data["ce_fdh_peers"] = analisys_ce_fdh["peers"]

        # We need to make sure ce_cm_conf (if present) comes before cr_cm_conf
        if "ce_cm_conf" in ceilings:
            analisys_ce_cm_conf = p_nca_wrapper(
                "ce_cm_conf", loop_data, bn_data, effect_aggregation
            )
            loop_data["ce_cm_conf_columns"] = analisys_ce_cm_conf["line"].get("columns")

        analyses = {}
        for ceiling in ceilings:
            if ceiling == "ce_fdh":
                analysis = analisys_ce_fdh
            elif ceiling == "ce_cm_conf":
                analysis = analisys_ce_cm_conf
            else:
                analysis = p_nca_wrapper(ceiling, loop_data, bn_data, effect_aggregation)

            if analysis.get("bottleneck") is not None and ceiling not in P_NO_BOTTLENECK:
                if ceiling not in bn_data["bottlenecks"]:
                    bn_data["bottlenecks"][ceiling] = {}
                bn_data["bottlenecks"][ceiling][x_name] = analysis["bottleneck"]
                # Handle 'cases' attribute if needed
                if "cases" in analysis:
                    bn_data["bottlenecks"][ceiling][x_name]["cases"] = analysis["cases"]

            if "bottleneck" in analysis:
                del analysis["bottleneck"]

            analyses[ceiling] = analysis

            if ceiling not in peers:
                peers[ceiling] = {}
            peers[ceiling][x_name] = analysis.get("peers")

        test_tuple = p_test(analyses, loop_data, test_params, effect_aggregation)
        if test_tuple is not None:
            tests[x_name] = test_tuple["test"]
            test_time += test_tuple["test_time"]

        # Add P-value/accuracy for displaying in summary
        for ceiling in ceilings:
            if x_name in tests and ceiling in tests[x_name]:
                analyses[ceiling]["p"] = tests[x_name][ceiling]["p_value"]
                analyses[ceiling]["p_accuracy"] = tests[x_name][ceiling]["test_params"][
                    "p_accuracy"
                ]
            else:
                analyses[ceiling]["p"] = float("nan")
                analyses[ceiling]["p_accuracy"] = float("nan")

        plots[x_name] = p_plot(analyses, loop_data, corner)
        summaries[x_name] = p_summary(analyses, loop_data)

    # Shut down cluster for parallisation
    p_cluster_cleanup()

    # Add the bottlenecks with mpy attribute
    bottlenecks = bn_data["bottlenecks"]

    model = {
        "plots": plots,
        "summaries": summaries,
        "bottlenecks": bottlenecks,
        "peers": peers,
        "tests": tests,
        "test_time": p_test_time(test_time),
        "mpy": bn_data["mpy"],
        "flip_y": flip_y,
        "show_plots": False,
    }

    return model
