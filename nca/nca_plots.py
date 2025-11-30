import math

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from .p_constants import LINE_COLORS as default_line_colors
from .p_constants import LINE_TYPES as default_line_types
from .p_constants import LINE_WIDTH as default_line_width
from .p_constants import P_CEILINGS_STEP
from .p_constants import POINT_COLOR as default_point_color
from .p_constants import POINT_TYPE as default_point_type
from .p_graphics import p_new_pdf, p_new_window
from .p_utils import p_generate_title, p_pretty_name


def p_plot(analyses, loop_data, corner):
    plot = {}

    plot["x"] = loop_data["x"]
    plot["y"] = loop_data["y"]
    plot["scope_theo"] = loop_data["scope_theo"]
    plot["names"] = loop_data["names"]
    plot["flip_x"] = loop_data["flip_x"] if corner is None else False
    plot["flip_y"] = loop_data["flip_y"] if corner is None else False
    plot["conf"] = loop_data["conf"]

    # Handle column names
    x_name = plot["x"].name if hasattr(plot["x"], "name") else "X"
    y_name = plot["y"].name if hasattr(plot["y"], "name") else "Y"

    plot["title"] = p_generate_title(x_name, y_name)
    plot["methods"] = list(analyses.keys())
    plot["lines"] = {}

    for method in plot["methods"]:
        analysis = analyses[method]
        # Assuming analysis is a dict
        if "line" not in analysis or analysis["line"] is None:
            continue
        plot["lines"][method] = analysis["line"]

    return plot


def p_display_plot(plot, pdf=False, path=None):
    # Get the params for plotting
    params = get_plot_params()
    line_colors = params[0]
    line_types = params[1]
    line_width = params[2]
    _point_type = params[3]  # Unused but kept for API compatibility
    point_color = params[4]

    # Open new window or PDF
    file_name = None
    if pdf:
        file_name = p_new_pdf("plot", plot["title"], path)
        fig = plt.figure()
    else:
        fig = p_new_window(title=plot["title"])

    ax = fig.add_subplot(111)

    # Plot the data points
    # xlim and ylim
    idx1 = 0 + int(plot["flip_x"])
    idx2 = 1 - int(plot["flip_x"])
    xlim = [plot["scope_theo"][idx1], plot["scope_theo"][idx2]]

    idx3 = 2 + int(plot["flip_y"])
    idx4 = 3 - int(plot["flip_y"])
    ylim = [plot["scope_theo"][idx3], plot["scope_theo"][idx4]]

    # Confidence lines might be outside the scope
    ylim = p_con_lim(ylim, plot, plot["flip_y"])

    # Plot points
    marker = "o"  # Default mapping for now

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    x_label = plot["x"].name if hasattr(plot["x"], "name") else "X"
    y_label = plot["names"][-1] if plot["names"] else "Y"

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.scatter(plot["x"], plot["y"], c=point_color, marker=marker)

    # Plot the scope outline
    p_plot_outline(plot, ax)

    # Plot the legend
    legend_names = []
    legend_handles = []

    # Map R lty to matplotlib linestyle
    ls_map = {1: "-", 2: "--", 3: ":", 4: "-.", 5: (0, (5, 10)), 6: (0, (3, 5, 1, 5))}

    for method in plot["methods"]:
        l_color = line_colors.get(method, "black")
        l_type = line_types.get(method, 1)
        ls = ls_map.get(l_type, "-")

        if method in ["ce_cm_conf", "cr_cm_conf"]:
            long_name = f"{p_pretty_name(method)} {plot['conf']}"
            legend_names.append(long_name)
        else:
            legend_names.append(p_pretty_name(method))

        handle = mlines.Line2D([], [], color=l_color, linestyle=ls, label=legend_names[-1])
        legend_handles.append(handle)

    if legend_names:
        ax.legend(handles=legend_handles, loc="upper left", fontsize="small", frameon=False)

    # Print the lines
    for method in plot["methods"]:
        line = plot["lines"][method]
        l_color = line_colors.get(method, "black")
        l_type = line_types.get(method, 1)
        ls = ls_map.get(l_type, "-")
        l_width = line_width

        if method in P_CEILINGS_STEP:
            # line is [x, y]
            ax.plot(line[0], line[1], color=l_color, linestyle=ls, linewidth=l_width)
        else:
            if is_infinite(line):
                continue
            # line is [intercept, slope]
            intercept = line[0]
            slope = line[1]
            try:
                ax.axline(
                    (0, intercept), slope=slope, color=l_color, linestyle=ls, linewidth=l_width
                )
            except AttributeError:
                x_vals = np.array(ax.get_xlim())
                y_vals = intercept + slope * x_vals
                ax.plot(x_vals, y_vals, color=l_color, linestyle=ls, linewidth=l_width)

    # Title
    title_text = f"NCA Plot : {plot['title']}"
    ax.set_title(title_text)

    if pdf and file_name:
        fig.savefig(file_name, format="pdf")
        plt.close(fig)
    elif not pdf:
        plt.show()


def p_con_lim(ylim, plot, flip_y):
    columns = None
    if "ce_cm_conf" in plot["methods"]:
        line = plot["lines"]["ce_cm_conf"]
        if hasattr(line, "columns"):
            columns = line.columns
        elif isinstance(line, dict) and "columns" in line:
            columns = line["columns"]
    elif "cr_cm_conf" in plot["methods"]:
        line = plot["lines"]["cr_cm_conf"]
        if hasattr(line, "columns"):
            columns = line.columns
        elif isinstance(line, dict) and "columns" in line:
            columns = line["columns"]
    else:
        return ylim

    if columns is None:
        return ylim

    try:
        # Assuming columns is numpy array shape (rows, cols)
        # R: columns[5,] -> index 4
        vals = columns[4, :]
        min_val = np.min(vals)
        max_val = np.max(vals)

        if not flip_y:
            ylim = [min(ylim[0], min_val), max(ylim[1], max_val)]
        else:
            ylim = [max(ylim[0], max_val), min(ylim[1], min_val)]
    except (TypeError, ValueError, IndexError):
        pass

    return ylim


def p_plot_outline(plot, ax):
    col = "grey"
    ls = "--"

    ax.axvline(x=plot["scope_theo"][0], color=col, linestyle=ls)
    ax.axvline(x=plot["scope_theo"][1], color=col, linestyle=ls)
    ax.axhline(y=plot["scope_theo"][2], color=col, linestyle=ls)
    ax.axhline(y=plot["scope_theo"][3], color=col, linestyle=ls)


def p_plot_grid_fixed(plot, size, ax):
    start = size * math.ceil(plot["scope_theo"][0] / size)
    while start < plot["scope_theo"][1]:
        ax.axvline(x=start, color="grey", linestyle="-")
        start += size

    start = size * math.ceil(plot["scope_theo"][2] / size)
    while start < plot["scope_theo"][3]:
        ax.axhline(y=start, color="grey", linestyle="-")
        start += size


def p_plot_grid(plot, size, ax):
    step_x = (plot["scope_theo"][1] - plot["scope_theo"][0]) / size
    start = plot["scope_theo"][0]
    while start < plot["scope_theo"][1]:
        ax.axvline(x=start, color="grey", linestyle="-")
        start += step_x

    step_y = (plot["scope_theo"][3] - plot["scope_theo"][2]) / size
    start = plot["scope_theo"][2]
    while start < plot["scope_theo"][3]:
        ax.axhline(y=start, color="grey", linestyle="-")
        start += step_y


def p_plot_boundaries(line, method, ax):
    columns = None
    if hasattr(line, "columns"):
        columns = line.columns
    elif isinstance(line, dict) and "columns" in line:
        columns = line["columns"]

    if columns is None:
        return

    # R: columns[2, col] -> index 1
    # R: columns[3, col] -> index 2
    try:
        for col in range(columns.shape[1]):
            ax.axvline(x=columns[1, col], color="grey", linestyle="--")

        # Last one from loop in R
        ax.axvline(x=columns[2, columns.shape[1] - 1], color="grey", linestyle="--")

        if method == "cr_cm_conf":
            # R: points(t(columns[4:5,]), col="red") -> indices 3 and 4
            pts = columns[3:5, :].T
            ax.scatter(pts[:, 0], pts[:, 1], color="red")
    except (TypeError, ValueError, IndexError):
        pass


def get_plot_params():
    line_colors = default_line_colors
    line_types = default_line_types
    line_width_val = default_line_width
    pt_type = default_point_type
    pt_color = default_point_color
    return [line_colors, line_types, line_width_val, pt_type, pt_color]


def is_infinite(line):
    try:
        if isinstance(line, (list, tuple, np.ndarray)):
            if not np.all(np.isfinite(line)):
                return True
        if isinstance(line, dict) and "coefficients" in line:
            if not np.all(np.isfinite(line["coefficients"])):
                return True
    except (TypeError, ValueError):
        pass
    return False
