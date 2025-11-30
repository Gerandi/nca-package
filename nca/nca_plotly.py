import warnings

import numpy as np

from .p_constants import EPSILON, LINE_COLORS, LINE_TYPES, LINE_WIDTH, P_CEILINGS_STEP
from .p_utils import p_pretty_number


def p_display_plotly(plot, peers, labels, name="peer", coord_list=None):
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly is not installed. Please install it to use this feature.")
        return

    # Get the params for plotting
    # params <- get_plot_params() -> This function is likely in nca_plots.R or p_graphics.R?
    # It's not in p_constants.R.
    # But we have constants imported.
    # LINE_COLORS, LINE_TYPES, LINE_WIDTH, point_type, point_color

    # Missing (or too much) labels
    # plot['x'] and plot['y'] are DataFrames or Series?
    # In R: plot$x
    x_data = plot["x"]
    y_data = plot["y"]

    # Assuming x_data and y_data are Series or single-column DataFrames
    if hasattr(x_data, "iloc"):
        x_vals = x_data.iloc[:, 0] if x_data.ndim > 1 else x_data
    else:
        x_vals = x_data

    if hasattr(y_data, "iloc"):
        y_vals = y_data.iloc[:, 0] if y_data.ndim > 1 else y_data
    else:
        y_vals = y_data

    n_obs = len(x_vals)

    if labels is None or len(labels) != n_obs or len(set(labels)) > 5:
        labels = ["obs"] * n_obs
        color_list = ["blue"]
    else:
        color_list = ["blue", "green3", "cyan", "magenta", "gray"]

    fig = go.Figure()

    # Add peers as separate trace first
    if peers is not None and len(peers) > 0:
        # peers is a DataFrame/matrix with x, y columns.
        # R: peers[, 1], peers[, 2]
        # R: rownames(peers)

        peer_x = peers.iloc[:, 0]
        peer_y = peers.iloc[:, 1]
        peer_names = peers.index

        # hover text: <b>name</b><br>x, y
        hover_text = [
            f"<b>{name}</b><br>{x}, {y}" for name, x, y in zip(peer_names, peer_x, peer_y)
        ]

        fig.add_trace(
            go.Scatter(
                x=peer_x,
                y=peer_y,
                mode="markers",
                marker={"color": "red", "size": 10},
                text=hover_text,
                hovertemplate="%{text}",
                name=name,
                showlegend=True,
            )
        )

    # Add the scatter plot for the remaining points
    # include <- !(rownames(plot$x) %in% rownames(peers))
    if peers is not None:
        include = ~x_vals.index.isin(peers.index)
    else:
        include = [True] * n_obs

    # Filter
    x_rem = x_vals[include]
    y_rem = y_vals[include]
    labels_rem = np.array(labels)[include]
    names_rem = x_vals.index[include]

    # We need to handle colors based on labels.
    # If labels are all 'obs', just one trace.
    # If multiple labels, multiple traces or use color array.

    unique_labels = sorted(list(set(labels_rem)))

    for i, label in enumerate(unique_labels):
        mask = labels_rem == label
        x_l = x_rem[mask]
        y_l = y_rem[mask]
        names_l = names_rem[mask]

        hover_text_l = [f"<b>{n}</b><br>{x}, {y}" for n, x, y in zip(names_l, x_l, y_l)]

        # Color mapping
        color = color_list[i % len(color_list)]

        fig.add_trace(
            go.Scatter(
                x=x_l,
                y=y_l,
                mode="markers",
                marker={"symbol": "circle", "color": color},
                text=hover_text_l,
                hovertemplate="%{text}",
                name=str(label),
                showlegend=True,
            )
        )

    # Print the lines
    for method in plot["methods"]:
        if method == "ols":
            continue

        line = plot["lines"][method]
        line_color = LINE_COLORS.get(method, "black")
        line_type = LINE_TYPES.get(method, 1)  # 1 is solid

        # Map R lty to plotly dash
        # 1=solid, 2=dash, 3=dot, 4=dashdot, 5=longdash, 6=longdashdot?
        dash_map = {1: "solid", 2: "dash", 3: "dot", 4: "dashdot", 5: "longdash", 6: "longdashdot"}
        dash_style = dash_map.get(line_type, "solid")

        line_props = {"color": line_color, "width": LINE_WIDTH, "dash": dash_style}

        if method in P_CEILINGS_STEP:
            # line is list of x and y coords
            # R: x = c(line[[1]]), y = c(line[[2]])
            lx = line[0]
            ly = line[1]

            fig.add_trace(
                go.Scatter(x=lx, y=ly, mode="lines", line=line_props, name=method, showlegend=True)
            )
        else:
            # Line coefficients
            # if (is_infinite(line) || is.null(line))
            if line is None:  # Check infinite?
                continue

            # if (is.double(line)) -> list/array of 2
            if isinstance(line, (list, tuple, np.ndarray)) and len(line) == 2:
                intercept = line[0]
                slope = line[1]
            else:
                # Assuming dict or object with coefs
                # intercept <- unname(coef(line)["(Intercept)"])
                # slope <- unname(coef(line)["x"])
                # We need to know how line is stored for OLS/QR etc.
                # Usually [intercept, slope] in Python port?
                # Let's assume it is [intercept, slope] for now.
                intercept = line[0]
                slope = line[1]

            # Points from X scope
            scope = plot["scope_theo"]  # [xmin, xmax, ymin, ymax]
            # R: scope[1] -> scope[0]

            y1 = intercept + slope * scope[0]
            y2 = intercept + slope * scope[1]

            # Points from Y scope
            # x3 <- (scope[3] - intercept) / slope
            if abs(slope) > 1e-10:
                x3 = (scope[2] - intercept) / slope
                x4 = (scope[3] - intercept) / slope
            else:
                x3 = x4 = float("inf")  # or handle horizontal line

            # df <- data.frame(x = c(scope[1], scope[2], x3, x4), y = c(y1, y2, scope[3], scope[4]))
            # R: scope[1], scope[2] are Xmin, Xmax.
            # R: scope[3], scope[4] are Ymin, Ymax.

            pts_x = [scope[0], scope[1], x3, x4]
            pts_y = [y1, y2, scope[2], scope[3]]

            # Filter points outside scope
            # df <- df[df$x >= (scope[1] - EPSILON) & df$x <= (scope[2] + EPSILON),]
            # df <- df[df$y >= (scope[3] - EPSILON) & df$y <= (scope[4] + EPSILON),]

            final_x = []
            final_y = []

            for px, py in zip(pts_x, pts_y):
                if (
                    scope[0] - EPSILON <= px <= scope[1] + EPSILON
                    and scope[2] - EPSILON <= py <= scope[3] + EPSILON
                ):
                    final_x.append(px)
                    final_y.append(py)

            # Sort by x to draw line correctly?
            # Usually just 2 points needed for line.
            # If we have more than 2, we should sort.
            if len(final_x) >= 2:
                # Sort by x
                sorted_pts = sorted(zip(final_x, final_y))
                final_x = [p[0] for p in sorted_pts]
                final_y = [p[1] for p in sorted_pts]

                fig.add_trace(
                    go.Scatter(
                        x=final_x,
                        y=final_y,
                        mode="lines",
                        line=line_props,
                        name=method,
                        showlegend=True,
                    )
                )

    # Add the bottleneck lines
    if coord_list:
        done = []
        for coord in coord_list:
            # coord: [x1, x2, y1, y2]?
            # R: x = c(coord[1], coord[2]), y = coord[4]
            # R: x = coord[2], y = c(coord[3], coord[4])
            # coord indices 1,2,3,4 -> 0,1,2,3

            line_style = {"width": 1, "dash": "dot", "color": "lightgrey"}

            # Horizontal line
            fig.add_trace(
                go.Scatter(
                    x=[coord[0], coord[1]],
                    y=[coord[3], coord[3]],
                    mode="lines",
                    line=line_style,
                    showlegend=False,
                )
            )

            # Vertical line
            fig.add_trace(
                go.Scatter(
                    x=[coord[1], coord[1]],
                    y=[coord[2], coord[3]],
                    mode="lines",
                    line=line_style,
                    showlegend=False,
                )
            )

            if coord[3] not in done:
                done.append(coord[3])
                y_pretty = p_pretty_number(coord[3], prec="auto")

                fig.add_annotation(
                    x=coord[0],
                    y=coord[3],
                    xref="x",
                    yref="y",
                    ax=-20,
                    ay=-20,
                    text=y_pretty,
                    font={"size": 10},
                    arrowsize=0.5,
                    arrowwidth=2,
                    arrowcolor="lightgrey",
                )

    # Add title and axis labels
    # title <- list(text = paste0("NCA Plot : ", plot$title), yanchor = "top")
    title_text = f"NCA Plot : {plot['title']}"

    # xaxis <- list(title = colnames(plot$x))
    xaxis_title = x_data.columns[0] if hasattr(x_data, "columns") else "X"

    # yaxis <- list(title = colnames(plot$y))
    yaxis_title = y_data.columns[0] if hasattr(y_data, "columns") else "Y"

    xaxis = {"title": xaxis_title}
    if plot.get("flip_x"):
        xaxis["autorange"] = "reversed"

    yaxis = {"title": yaxis_title}
    if plot.get("flip_y"):
        yaxis["autorange"] = "reversed"

    fig.update_layout(title={"text": title_text, "yanchor": "top"}, xaxis=xaxis, yaxis=yaxis)

    fig.show()


def p_all_color(x):
    # Check if all elements in x are valid colors
    # In Python, matplotlib.colors.is_color_like
    import matplotlib.colors as mcolors

    return all(mcolors.is_color_like(c) for c in x)


def p_suppress_warnings(expr):
    # Python context manager for warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expr()
