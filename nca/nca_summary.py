import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .p_constants import DASH_COUNT, P_GLOBAL_NAMES, P_RESULT_NAMES
from .p_graphics import p_new_pdf
from .p_utils import p_generate_title, p_get_digits, p_is_number, p_pretty_number


def p_display_summary(summary, pdf=False, path=None):
    if pdf:
        p_display_summary_pdf(summary, path)
    else:
        p_display_summary_screen(summary)


def p_display_summary_pdf(summary, path):
    x_name = summary["names"][0]
    y_name = summary["names"][1]

    file_name = p_new_pdf("summary", p_generate_title(x_name, y_name), path, paper="A4r")

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(8.27, 11.69))  # A4 size approx

    # Plot global
    ax1 = axes[0]
    ax1.axis("off")

    # Title
    title = f"NCA Parameters : {p_generate_title(x_name, y_name)}"
    ax1.set_title(title, fontsize=14)

    df_global = p_pretty_global(summary["global"])
    # Render table
    table1 = ax1.table(
        cellText=df_global.values,
        rowLabels=df_global.index,
        colLabels=df_global.columns,
        loc="center",
        cellLoc="center",
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1, 1.5)

    # Plot params
    ax2 = axes[1]
    ax2.axis("off")

    if summary["params"].shape[1] == 0:
        ax2.text(
            0.5,
            0.5,
            " No NCA parameters available because only OLS selected\n",
            ha="center",
            va="center",
            fontsize=12,
        )
    else:
        df_params = p_pretty_params(summary["params"])
        table2 = ax2.table(
            cellText=df_params.values,
            rowLabels=df_params.index,
            colLabels=df_params.columns,
            loc="center",
            cellLoc="center",
        )
        table2.auto_set_font_size(False)
        table2.set_fontsize(10)
        table2.scale(1, 1.5)

    axes[2].axis("off")

    if file_name:
        fig.savefig(file_name, format="pdf")
        plt.close(fig)


def p_display_summary_screen(summary):
    x_name = summary["names"][0]
    y_name = summary["names"][1]
    title = f"NCA Parameters : {p_generate_title(x_name, y_name)}"

    print("\n" + "-" * DASH_COUNT)
    print(title)
    print("-" * DASH_COUNT)
    print(p_pretty_global(summary["global"]))
    print("\n")

    if summary["params"].shape[1] == 0:
        print(" No NCA parameters available because only OLS selected\n")
    else:
        print(p_pretty_params(summary["params"]))
    print("\n")


def p_display_summary_simple(summaries):
    first_key = list(summaries.keys())[0]
    if summaries[first_key]["params"].shape[1] == 0:
        print("\n No effect sizes available because only OLS selected\n")
        return

    rows = list(summaries.keys())
    param_cols = summaries[first_key]["params"].columns
    n_param_cols = len(param_cols)

    simple_data = []

    for x_name in rows:
        tmp = summaries[x_name]["params"]
        row_vals = []
        for j in range(n_param_cols):
            # Effect size (Row 2, index 1)
            es = tmp.iloc[1, j]
            if pd.isna(es):
                row_vals.append(np.nan)
            else:
                row_vals.append(f"{es:.2f}")

            # p-value (Row 6, index 5)
            p = tmp.iloc[5, j]
            if pd.isna(p):
                row_vals.append(np.nan)
            else:
                row_vals.append(f"{p:.3f}")
        simple_data.append(row_vals)

    cols = []
    for col in param_cols:
        cols.append(col)
        cols.append("p")

    simple_df = pd.DataFrame(simple_data, index=rows, columns=cols)

    simple_df = simple_df.replace("nan", np.nan)
    simple_df = simple_df.dropna(axis=1, how="all")

    print("\n" + "-" * DASH_COUNT)
    print("Effect size(s):")
    print(simple_df.to_string(na_rep=""))
    print("-" * DASH_COUNT + "\n\n")


def p_summary(analyses, loop_data):
    obs = min(len(loop_data["x"]), len(loop_data["y"]))
    emp = loop_data["scope_emp"]
    scope_emp_area = (emp[1] - emp[0]) * (emp[3] - emp[2])

    if loop_data["scope_theo"] == loop_data["scope_emp"]:
        mat1 = pd.DataFrame(index=P_GLOBAL_NAMES, columns=[""])
        mat1.iloc[0, 0] = obs
        mat1.iloc[1, 0] = loop_data["scope_area"]
        mat1.iloc[2:6, 0] = loop_data["scope_emp"]
    else:
        mat1 = pd.DataFrame(index=P_GLOBAL_NAMES, columns=["", " "])
        new_index = list(P_GLOBAL_NAMES)
        new_index[1] = "Scope  emp / theo"
        mat1.index = new_index

        mat1.iloc[0, :] = [obs, np.nan]
        mat1.iloc[1, :] = [scope_emp_area, loop_data["scope_area"]]
        mat1.iloc[2:6, 0] = loop_data["scope_emp"]
        mat1.iloc[2:6, 1] = loop_data["scope_theo"]

    methods = [m for m in analyses.keys() if m != "ols"]

    mat2 = pd.DataFrame(index=P_RESULT_NAMES, columns=methods)

    for m in methods:
        a = analyses[m]
        mat2.loc["Ceiling zone", m] = a.get("ceiling", np.nan)
        mat2.loc["Effect size", m] = a.get("effect", np.nan)
        mat2.loc["# above", m] = a.get("above", np.nan)
        mat2.loc["c-accuracy", m] = a.get("accuracy", np.nan)
        mat2.loc["Fit", m] = a.get("fit", np.nan)
        mat2.loc["p-value", m] = a.get("p", np.nan)
        mat2.loc["p-accuracy", m] = a.get("p_accuracy", np.nan)
        mat2.loc[" ", m] = np.nan
        mat2.loc["Slope", m] = a.get("slope", np.nan)
        mat2.loc["Intercept", m] = a.get("intercept", np.nan)
        mat2.loc["Abs. ineff.", m] = a["ineffs"].get("abs", np.nan) if "ineffs" in a else np.nan
        mat2.loc["Rel. ineff.", m] = a["ineffs"].get("rel", np.nan) if "ineffs" in a else np.nan
        mat2.loc["Condition ineff.", m] = a["ineffs"].get("x", np.nan) if "ineffs" in a else np.nan
        mat2.loc["Outcome ineff.", m] = a["ineffs"].get("y", np.nan) if "ineffs" in a else np.nan

    names = [
        loop_data["x"].name if hasattr(loop_data["x"], "name") else "X",
        loop_data["names"][-1] if loop_data["names"] else "Y",
    ]

    return {"global": mat1, "params": mat2, "names": names}


def p_pretty_global(global_df):
    vals = global_df.iloc[2:6, :].values.flatten()
    vals = vals[~pd.isnull(vals)]
    digits = p_get_digits(vals)

    pretty = pd.DataFrame(index=global_df.index, columns=global_df.columns)

    for row in range(6):
        val1 = global_df.iloc[row, 0]
        if row == 0:
            pretty.iloc[row, 0] = p_pretty_number(val1, " ", prec=0, use_spaces=True)
        else:
            pretty.iloc[row, 0] = p_pretty_number(val1, "", digits, True)

        if global_df.shape[1] > 1:
            val2 = global_df.iloc[row, 1]
            if row == 0:
                pretty.iloc[row, 1] = ""
            else:
                pretty.iloc[row, 1] = p_pretty_number(val2, "", digits, True)

    pretty.columns = [" " for _ in range(pretty.shape[1])]
    return pretty


def p_pretty_params(params_df):
    pretty = params_df.copy()

    for row_idx in range(len(params_df.index)):
        for col_idx in range(len(params_df.columns)):
            val = params_df.iloc[row_idx, col_idx]

            if row_idx == 2:
                pretty.iloc[row_idx, col_idx] = p_pretty_number(val, prec=0, use_spaces=True)
            elif row_idx in (3, 4):
                if not p_is_number(val):
                    if row_idx == 4 and pd.isna(val):
                        pretty.iloc[row_idx, col_idx] = "NA    "
                    else:
                        pretty.iloc[row_idx, col_idx] = ""
                elif val % 1 == 0:
                    s = p_pretty_number(val, "", 0)
                    pretty.iloc[row_idx, col_idx] = f"{s}%   "
                else:
                    s = p_pretty_number(val, "", 1)
                    pretty.iloc[row_idx, col_idx] = f"{s}% "
            else:
                pretty.iloc[row_idx, col_idx] = p_pretty_number(val)

    if (pretty.iloc[6, :] == "").all():
        pretty = pretty.drop(pretty.index[[5, 6]])

    return pretty
