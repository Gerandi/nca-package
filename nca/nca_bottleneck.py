import math

import pandas as pd

from .p_constants import DASH_COUNT
from .p_utils import p_get_digits, p_pretty_name, p_pretty_number


def p_display_bottleneck(bottlenecks, title="Bottleneck", pdf=False, path=None):
    if pdf:
        # Put all tables in 1 file
        from .p_graphics import p_close_pdf, p_new_pdf

        # colnames(bottlenecks[[1]])[1] -> bottlenecks is a dict of dicts/dfs?
        # In R: bottlenecks is a list of lists/dfs.
        # bottlenecks[[1]] is the first element.
        # We need to get the first key.
        first_method = list(bottlenecks.keys())[0]
        first_bn = bottlenecks[first_method]
        # Assuming first_bn is a DataFrame
        first_col_name = str(first_bn.columns[0])  # Convert to string for filename

        p_new_pdf("bottlenecks", first_col_name, path, paper="A4r")

        for method in bottlenecks:
            p_display_table_pdf(bottlenecks[method], p_pretty_name(method), title)

        # Close the file
        p_close_pdf()
        print("")
    else:
        for method in bottlenecks:
            p_display_table_screen(bottlenecks[method], p_pretty_name(method), title)


def p_display_table_pdf(bn, method, title):
    import matplotlib.pyplot as plt

    names = bn.columns
    bn_x = bn.attrs.get("bn_x")
    bn_y = bn.attrs.get("bn_y")
    bn_y_id = bn.attrs.get("bn_y_id")
    rows = len(bn)
    x_length = len(bn.columns) - 1
    size = bn.attrs.get("size")
    _cutoff = bn.attrs.get("cutoff")  # prefixed with _ as unused

    # Prepare data for plotting
    tmp = pd.DataFrame(index=bn.index, columns=range(x_length))

    for i in range(x_length):
        col_idx = i + 1  # Skip first column (Y)
        col_data = bn.iloc[:, col_idx]

        if bn_x == "percentile":
            # R: cases <- round(size * as.numeric(bn[,i+1]) / 100, digits = 0)
            numeric_vals = pd.to_numeric(col_data, errors="coerce").fillna(0)
            cases = (size * numeric_vals / 100).round(0).astype(int)

            # R: cases <- ifelse(bn[,i+1] == "NN", 0, cases)
            is_nn = col_data == "NN"
            cases[is_nn] = 0

            # paste0(bn[,i+1], ' (', as.character(cases), ')')
            tmp.iloc[:, i] = col_data.astype(str) + " (" + cases.astype(str) + ")"
        else:
            tmp.iloc[:, i] = col_data

    if tmp.empty:
        return

    # Set y precision
    if bn_y_id in [1, 2]:
        # R: digits <- ifelse((100 / (rows-1)) %% 1 == 0, 0, 1)
        val = 100 / (rows - 1) if rows > 1 else 0
        digits = 0 if val % 1 == 0 else 1
    else:
        digits = p_get_digits(bn.iloc[:, 0])

    col_names = [str(i + 1) for i in range(x_length)]
    # row.names <- sapply(bn[,1], p_pretty_number, "", digits)
    row_names = [p_pretty_number(x, "", digits) for x in bn.iloc[:, 0]]

    if title and title != "":
        full_title = f"{title} {method} : {names[0]}"
    else:
        full_title = ""

    legend = f"{bn_y} / {bn_x}\n"
    for i in range(x_length):
        legend += f"{i+1} {names[i+1]}\n"

    start = 0
    while start < rows:
        end = min(start + 30, rows)

        tmp_part = tmp.iloc[start:end, :]
        tmp_part.columns = col_names
        tmp_part.index = row_names[start:end]

        # Plotting
        # R: textplot(tmp.part, cex=1, halign="left", valign="top", mar=c(0, 0, 3, 0))
        # R: title(title, cex.main=1, sub=legend)

        # Create figure
        fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4r size approx (landscape)
        ax.axis("off")

        # Add title and legend
        # R's title puts main title on top, sub (legend) on bottom.
        plt.suptitle(full_title, fontsize=12)
        plt.figtext(0.5, 0.05, legend, wrap=True, horizontalalignment="center", fontsize=10)

        # Table
        # matplotlib table doesn't support "halign='left', valign='top'" easily
        # for the whole table layout like R's textplot. But we can approximate.
        table = ax.table(
            cellText=tmp_part.values,
            rowLabels=tmp_part.index,
            colLabels=tmp_part.columns,
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # Save to current figure
        plt.savefig(f"{method}_{start}.pdf")

        plt.close(fig)

        start = end


def p_display_table_screen(bn, method, title):
    names = bn.columns
    bn_x = bn.attrs.get("bn_x")
    bn_y = bn.attrs.get("bn_y")
    bn_y_id = bn.attrs.get("bn_y_id")
    rows = len(bn)
    x_length = len(bn.columns) - 1
    size = bn.attrs.get("size")
    cutoff = bn.attrs.get("cutoff")

    tmp = pd.DataFrame(index=bn.index, columns=range(x_length))

    for i in range(x_length):
        col_idx = i + 1
        values = bn.iloc[:, col_idx]

        if bn_x == "percentile":
            # cases <- attr(values, "cases")
            # In Python, series don't have attrs usually preserved well, or we stored it in bn.attrs?
            # In nca.py, I added 'cases' to the bottleneck dict.
            # But here bn is a DataFrame.
            # Maybe 'cases' is in bn.attrs['cases']?
            # Or maybe values has it if we used a custom class.
            # Let's assume we can get cases.
            # If not, we recalculate like in PDF version?
            # R code says: cases <- attr(values, "cases")
            # This implies the column itself has an attribute.
            # Pandas columns (Series) can have attrs.
            cases = values.attrs.get("cases")
            if cases is None:
                # Fallback: recalculate
                numeric_vals = pd.to_numeric(values, errors="coerce").fillna(0)
                cases = (size * numeric_vals / 100).round(0).astype(int)
                is_nn = values == "NN"
                cases[is_nn] = 0

            tmp.iloc[:, i] = values.astype(str) + " (" + cases.astype(str) + ")"
        else:
            tmp.iloc[:, i] = values

    if tmp.empty:
        return

    if bn_y_id in [1, 2]:
        val = 100 / (rows - 1) if rows > 1 else 0
        digits = 0 if val % 1 == 0 else 1
    else:
        digits = p_get_digits(bn.iloc[:, 0])

    tmp.columns = [str(i + 1) for i in range(x_length)]
    tmp.index = [p_pretty_number(x, "", digits, True) for x in bn.iloc[:, 0]]

    # Display header
    # fmt <- sprintf(" %%-%ds", max(nchar(names)))
    max_len = max(len(str(n)) for n in names)
    fmt = f" {{:<{max_len}s}}"

    print("\n" + "-" * DASH_COUNT)
    print(f"{title} {method} (cutoff = {cutoff})")
    print(f"Y{fmt.format(str(names[0]))} ({bn_y})")
    for i in range(x_length):
        print(f"{i+1}{fmt.format(str(names[i+1]))} ({bn_x})")
    print("-" * DASH_COUNT)

    # Display table
    print("Y", end="")
    # Pandas to_string() can print the table
    print(
        tmp.to_string(header=True, index=True)
    )  # This might not match R's print exactly but close enough
    print("")


def p_display_table_screen_tab(bn, method, title):
    names = bn.columns
    bn_x = bn.attrs.get("bn_x")
    bn_y = bn.attrs.get("bn_y")
    bn_y_id = bn.attrs.get("bn_y_id")
    rows = len(bn)
    x_length = len(bn.columns) - 1
    size = bn.attrs.get("size")
    cutoff = bn.attrs.get("cutoff")

    tmp = pd.DataFrame(index=bn.index, columns=range(x_length))

    for i in range(x_length):
        col_idx = i + 1
        values = bn.iloc[:, col_idx]

        if bn_x == "percentile":
            cases = values.attrs.get("cases")
            if cases is None:
                numeric_vals = pd.to_numeric(values, errors="coerce").fillna(0)
                cases = (size * numeric_vals / 100).round(0).astype(int)
                is_nn = values == "NN"
                cases[is_nn] = 0

            tmp.iloc[:, i] = values.astype(str) + " (" + cases.astype(str) + ")"
        else:
            tmp.iloc[:, i] = values

    if tmp.empty:
        return

    if bn_y_id in [1, 2]:
        val = 100 / (rows - 1) if rows > 1 else 0
        digits = 0 if val % 1 == 0 else 1
    else:
        digits = p_get_digits(bn.iloc[:, 0])

    tmp.columns = [str(i + 1) for i in range(x_length)]
    tmp.index = [p_pretty_number(x, "", digits, True) for x in bn.iloc[:, 0]]

    # Display header
    tab = 8
    # tabs <- ceiling((3 + max(nchar(names))) / tab)
    max_name_len = max(len(str(n)) for n in names)
    tabs = math.ceil((3 + max_name_len) / tab)

    # offset <- floor(log10(x.length)) + 2
    offset = math.floor(math.log10(x_length)) + 2 if x_length > 0 else 2

    # fmt <- sprintf("%%%ds ", offset - 1)
    fmt = f"{{:>{offset - 1}s}} "

    print("\n----------------------------------------", end="")
    print("----------------------------------------")
    print(f"{title} {method} (cutoff = {cutoff})")

    # needed <- tabs - floor((offset + nchar(names[1])) / tab)
    needed = tabs - math.floor((offset + len(str(names[0]))) / tab)
    tab_str = '\t' * needed
    print(f"{fmt.format('Y')}{names[0]}{tab_str}({bn_y})")

    for i in range(x_length):
        # needed <- tabs - floor((offset + nchar(names[i + 1])) / tab)
        needed = tabs - math.floor((offset + len(str(names[i + 1]))) / tab)
        tab_str = '\t' * needed
        print(f"{fmt.format(str(i+1))}{names[i+1]}{tab_str}({bn_x})")

    print("----------------------------------------", end="")
    print("----------------------------------------")

    # Display table
    # tabs <- ceiling(max(nchar(tmp)) / tab)
    # This logic in R seems to calculate max length of all cells in tmp?
    # max(nchar(tmp)) in R works on the whole matrix.
    # In Python:
    max_cell_len = max(
        tmp[col].astype(str).map(len).max() for col in tmp.columns
    )

    tabs = math.ceil(max_cell_len / tab)

    # first <- ceiling(max(nchar(colnames(tmp))) / tab)
    max_col_len = max(len(str(c)) for c in tmp.columns)
    first = math.ceil(max_col_len / tab)

    output = ["Y", "\t" * first]
    tab_str = '\t' * tabs
    output.append("".join([f"{c}{tab_str}" for c in tmp.columns]) + "\n")

    for idx_row in range(len(tmp)):
        row_label = tmp.index[idx_row]
        needed = first - math.floor(len(str(row_label)) / tab)
        tab_str = '\t' * needed
        output.append(f"{row_label}{tab_str}")

        for idx_col in range(len(tmp.columns)):
            val = str(tmp.iloc[idx_row, idx_col])
            needed = tabs - math.floor(len(val) / tab)
            tab_str = '\t' * needed
            output.append(f"{val}{tab_str}")

        output.append("\n")

    print("".join(output))
    print("")
