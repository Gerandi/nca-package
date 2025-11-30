import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from .p_graphics import p_new_pdf, p_new_window
from .p_utils import _pool, p_pretty_name


def p_test(analyses, loop_data, test_params, effect_aggregation):
    if test_params["rep"] < 1:
        return None

    test_params["rep"] = round(test_params["rep"])
    test_params["p_confidence"] = max(0, min(1, test_params["p_confidence"]))
    test_params["p_threshold"] = max(0, min(1, test_params["p_threshold"]))

    test = {}
    h = len(loop_data["x"])
    y_org = loop_data["y"].copy()
    effect_sims = {}

    ceilings = [m for m in analyses.keys() if m != "ols"]

    start_time = time.time()

    x_name = loop_data["x"].name if hasattr(loop_data["x"], "name") else "X"
    if len(x_name) > 25:
        x_name = x_name[:25] + "..."

    samples = []
    seen_samples = set()

    # Generate samples
    for _ in range(test_params["rep"]):
        while True:
            s = tuple(np.random.permutation(h))
            if test_params["rep"] <= 720:
                if s not in seen_samples:
                    seen_samples.add(s)
                    samples.append(np.array(s))
                    break
            else:
                samples.append(np.array(s))
                break

    for ceiling in ceilings:
        print(f"Do test for  : {ceiling}-{x_name}")

        tasks = []
        for sample in samples:
            tasks.append((ceiling, loop_data, sample, effect_aggregation, y_org))

        if _pool:
            results = _pool.starmap(p_test_worker, tasks)
        else:
            results = [p_test_worker(*t) for t in tasks]

        valid_results = [r for r in results if r is not None and not np.isnan(r)]
        effect_sims[ceiling] = np.array(valid_results)

        print(f"\rDone test for: {ceiling}-{x_name}      ")

    for ceiling in ceilings:
        observed = analyses[ceiling]["effect"]
        data = effect_sims.get(ceiling)

        if data is None or len(data) == 0:
            print(f"No permutation test for {loop_data['names'][0]} on {ceiling}\n")
            continue

        threshold_value = np.quantile(np.sort(data), 1 - test_params["p_threshold"])

        tmp = data >= observed
        p_value = (np.sum(tmp) + 1) / (len(tmp) + 1)
        p_value = max(p_value, 1 / test_params["rep"])

        uns = test_params["p_confidence"] + 0.5 * (1 - test_params["p_confidence"])
        z_score = norm.ppf(uns)

        p_accuracy = z_score * math.sqrt(p_value * (1 - p_value) / test_params["rep"])

        if h <= 6 and test_params["rep"] == math.factorial(h):
            p_accuracy = 0

        names = [
            loop_data["x"].name if hasattr(loop_data["x"], "name") else "X",
            loop_data["names"][-1] if loop_data["names"] else "Y",
        ]

        test[ceiling] = {
            "data": data,
            "observed": observed,
            "test_params": {
                "rep": test_params["rep"],
                "p_threshold": test_params["p_threshold"],
                "p_confidence": test_params["p_confidence"],
                "p_accuracy": p_accuracy,
            },
            "p_value": p_value,
            "threshold_value": threshold_value,
            "names": names,
        }

    return {"test": test, "test_time": time.time() - start_time}


def p_test_worker(ceiling, loop_data, sample_indices, effect_aggregation, y_org):
    from .p_ceiling import p_nca_wrapper

    ld = loop_data.copy()
    if hasattr(y_org, "iloc"):
        ld["y"] = y_org.iloc[sample_indices].reset_index(drop=True)
    else:
        ld["y"] = y_org[sample_indices]

    if ceiling == "ce_cm_conf":
        analysis = p_nca_wrapper("ce_cm_conf", ld, None, effect_aggregation)
        return analysis["effect"]
    analysis = p_nca_wrapper(ceiling, ld, None, effect_aggregation)
    return analysis["effect"]


def p_test_time(test_time):
    m, s = divmod(test_time, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)

    if d > 0:
        return f"Test done in {int(d)} days, {int(h):02d}:{int(m):02d}:{int(s):02d}"
    return f"Test done in {int(h):02d}:{int(m):02d}:{int(s):02d}"


def p_display_test(test, pdf=False, path=None):
    for ceiling in reversed(list(test.keys())):
        p_display_ceiling_test(ceiling, test[ceiling], pdf, path)


def p_display_ceiling_test(ceiling, ceiling_test, pdf=False, path=None):
    pretty_ceiling = p_pretty_name(ceiling)
    data = ceiling_test["data"]
    observed = ceiling_test["observed"]
    p_value = ceiling_test["p_value"]
    threshold_value = ceiling_test["threshold_value"]
    p_threshold = ceiling_test["test_params"]["p_threshold"]
    names = ceiling_test["names"]

    if data is None or len(data) == 0:
        return

    file_name = None
    if pdf:
        title = f"{'-'.join(names)} random test"
        file_name = p_new_pdf("plot", title, path)
        fig = plt.figure()
    else:
        fig = p_new_window(title=f"Random test {pretty_ceiling}")

    ax = fig.add_subplot(111)

    x_low = -0.05
    x_high = max(np.max(data), observed) + 0.01

    ax.hist(data, bins=30, range=(x_low, x_high), color="white", edgecolor="black")

    label_main = f"X={names[0]}  Y={names[1]}  {pretty_ceiling}\n"
    ax.set_title(label_main)
    ax.set_xlabel("Permutated effect sizes")
    ax.set_ylabel("Frequency")

    if p_threshold > 0:
        ax.axvline(x=threshold_value, color="darkgreen", linestyle=(0, (5, 10)), linewidth=0.5)

    ax.axvline(x=observed, color="red", linestyle="-", linewidth=0.5)

    label_threshold = ""
    if p_threshold > 0:
        label_threshold = (
            f"----  threshold (d = {threshold_value:.2f}, p_threshold {p_threshold:.2f})"
        )

    label_p = ""
    if not pd.isna(p_value):
        rep = ceiling_test["test_params"]["rep"]
        p_accuracy = ceiling_test["test_params"]["p_accuracy"]
        if p_accuracy > 1e-6:
            p_min = max(0, p_value - p_accuracy)
            p_max = min(1, p_value + p_accuracy)
            label_p = f", p = {p_value:.3f} [{p_min:.3f}, {p_max:.3f}], rep = {rep}"
        else:
            label_p = f", p = {p_value:.3f}, rep = {rep}"

    label_observed = f"observed (d = {observed:.2f}{label_p})"

    if p_threshold > 0:
        ax.text(
            0.05,
            0.95,
            label_threshold,
            transform=ax.transAxes,
            color="darkgreen",
            fontsize=11,
            va="top",
        )
        ax.text(
            0.05,
            0.90,
            "___",
            transform=ax.transAxes,
            color="red",
            fontsize=10,
            fontweight="bold",
            va="top",
        )
        ax.text(
            0.08, 0.88, label_observed, transform=ax.transAxes, color="red", fontsize=11, va="top"
        )
    else:
        ax.text(
            0.5,
            0.95,
            label_observed,
            transform=ax.transAxes,
            color="red",
            fontsize=11,
            ha="center",
            va="top",
        )

    if pdf and file_name:
        fig.savefig(file_name, format="pdf")
        plt.close(fig)
    elif not pdf:
        plt.show()
