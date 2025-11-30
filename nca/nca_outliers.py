import multiprocessing
import warnings

import numpy as np
import pandas as pd

from .nca import nca_analysis
from .nca_plotly import p_display_plotly
from .p_constants import EPSILON, P_NO_PEER_LINE
from .p_peers import p_aggregate_peers
from .p_utils import p_cluster_cleanup, p_start_cluster
from .p_validate import p_validate_clean

HIDDEN = "hidden"
SHOWN = "shown"


def nca_outliers(
    data,
    x,
    y,
    ceiling=None,
    corner=None,
    flip_x=False,
    flip_y=False,
    scope=None,
    k=1,
    min_dif=1e-2,
    max_results=25,
    plotly=False,
    condensed=False,
):

    # Cleans up any cluster registration
    p_cluster_cleanup()

    input_ok = p_check_input(x, y, ceiling)
    if input_ok is False:
        return None

    if input_ok is None:
        ceiling = "ce_fdh"

    cleaned = p_validate_clean(data, x, y, outliers=True)
    # In R: data <- as.data.frame(p_validate_clean(data, x, y, TRUE))
    # p_validate_clean returns list(x=..., y=...)
    # We need to reconstruct a DataFrame
    x_data = cleaned["x"]
    y_data = cleaned["y"]

    if hasattr(x_data, "iloc") and x_data.ndim > 1:
        x_vals = x_data.iloc[:, 0]
    else:
        x_vals = x_data

    if hasattr(y_data, "iloc") and y_data.ndim > 1:
        y_vals = y_data.iloc[:, 0]
    else:
        y_vals = y_data

    data = pd.DataFrame({x: x_vals, y: y_vals})
    # x and y are column names now

    model = nca_analysis(
        data, x, y, ceilings=ceiling, corner=corner, flip_x=flip_x, flip_y=flip_y, scope=scope
    )

    # model['summaries'] is a dict keyed by x name.
    # In R: model$summaries[[1]]
    # We assume x is the key.
    summary = model["summaries"][x]
    # eff_or = summary['params'][1] # 2nd element in R (1-based) -> index 1 in Python
    # summary['params'] is a DataFrame. Row "Effect size" is index 1.
    # We assume single ceiling, so column 0.
    eff_or = summary["params"].iloc[1, 0]
    global_scope = summary["global"]

    params = {
        "model": model,
        "x": x,
        "y": y,
        "ceiling": ceiling,
        "corner": corner,
        "flip_x": flip_x,
        "flip_y": flip_y,
        "scope": scope,
        "eff_or": eff_or,
        "global": global_scope,
        "min_dif": min_dif,
        "peers": p_aggregate_peers(model["peers"], x),
    }

    org_outliers = p_get_outliers(data, params, 1)
    if k == 1 and org_outliers is None:
        print("\nNo outliers identified")
        return None

    outliers = p_format_outliers(org_outliers, max_results, 1, min_dif, condensed)

    if plotly and outliers is not None:
        # points <- data[outliers[, 1], c(x, y)]
        # outliers is a DataFrame, column 0 is names (indices)
        outlier_indices = outliers.iloc[:, 0]
        points = data.loc[outlier_indices, [x, y]]

        # labels <- paste0(rownames(points), '<br>diff ', outliers[, 5])
        # outliers column 4 (index 4) is dif.rel (5th column in R)
        labels = points.index.astype(str) + "<br>diff " + outliers.iloc[:, 4].astype(str)

        # marks <- paste0(outliers[, 6], outliers[, 7])
        # columns 5 and 6 (indices 5, 6) -> ceiling, scope
        marks = outliers.iloc[:, 5].astype(str) + outliers.iloc[:, 6].astype(str)

        points["labels"] = labels
        points["marks"] = marks

        # points <- points[points[, 4] != '', 1:3]
        # Filter where marks is not empty string?
        # In R: points[, 4] is marks.
        points = points[points["marks"] != ""]
        points = points[[x, y, "labels"]]  # 1:3 in R (x, y, labels)

        # p_display_plotly(model$plots[[1]], points, NULL, name = 'outlier')
        p_display_plotly(model["plots"][x], points, None, name="outlier")

    if k == 1:
        # If k == 1, all outliers are shown in plotly so we have to slice here
        # return(outliers[1:min(nrow(outliers), max.results),])
        return outliers.iloc[: min(len(outliers), max_results)]

    outliers = p_get_outliers(data, params, k, org_outliers)
    if outliers is None or len(outliers) == 0:
        print("\nNo outliers identified")
        return None

    return p_format_outliers(outliers, max_results, k, min_dif, condensed)


def p_check_input(x, y, ceiling):
    # x and y are strings (column names)
    if not isinstance(x, str):
        print()
        print("Outlier detection needs a single independent variable")
        return False
    if not isinstance(y, str):
        print()
        print("Outlier detection needs a single dependent variable")
        return False
    if ceiling is None:
        return None
    if isinstance(ceiling, list) and len(ceiling) != 1:
        print()
        print("Outlier detection needs a single ceiling")
        return False
    if isinstance(ceiling, str):
        # It's fine
        pass
    elif len(ceiling) != 1:
        print()
        print("Outlier detection needs a single ceiling")
        return False
    else:
        ceiling = ceiling[0]

    if ceiling == "ols":
        print()
        print("Outlier detection does not work with OLS")
        return False

    return True


def p_get_outliers(data, params, k, org_outliers=None):
    if k > len(data):
        k = len(data)
        warnings.warn(f"Reduced k to {len(data)}", stacklevel=2)

    combos = p_get_combos(data, params, k)

    # Start a cluster if needed
    # condition <- detectCores() > 2 && nrow(combos) > 250
    condition = multiprocessing.cpu_count() > 2 and len(combos) > 250
    p_start_cluster(condition)

    if condition:
        print(f"Starting the analysis on {multiprocessing.cpu_count()} cores")

    # Parallel execution logic
    # In Python, we can use multiprocessing.Pool
    # But we need to pass function and args.
    # p_get_outlier needs data, combo, params, k.

    # We need to prepare arguments for map
    # args = [(data, combos[i], params, k) for i in range(len(combos))]

    # However, passing 'data' (DataFrame) to every process might be slow if large.
    # But R does it (copy-on-write or serialization).

    # We'll implement sequential for now to ensure correctness,
    # or use the pool if p_start_cluster set it up.
    # But p_start_cluster in p_utils sets a global _pool.
    # We need to access it.
    from .p_utils import _pool

    outliers_list = []

    if condition and _pool:
        # Parallel
        # We need a wrapper function for map because map takes one arg
        # Or use starmap
        args = [(data, combos[i], params, k) for i in range(len(combos))]

        # Progress bar logic is hard with simple map.
        # We'll skip the progress dots for now or implement a callback.

        results = _pool.starmap(p_get_outlier_wrapper, args)
        outliers_list = [r for r in results if r is not None]

        print(f"\rDone{' ' * 50}\n")
    else:
        # Sequential
        ids = list(range(0, len(combos), max(1, round(len(combos) / 50))))
        for idx, combo in enumerate(combos):
            if condition and idx in ids:
                print(".", end="", flush=True)

            res = p_get_outlier(data, combo, params, k)
            if res is not None:
                outliers_list.append(res)

        if condition:
            print(f"\rDone{' ' * 50}\n")

    if not outliers_list:
        return org_outliers

    # Convert list of dicts/lists to DataFrame
    # p_get_outlier returns a list/dict with keys: outliers, eff.or, eff.nw, dif.abs, dif.rel, ceiling, scope, combo
    outliers = pd.DataFrame(outliers_list)

    # Second pass (new_outliers)
    # R: foreach(idx = 1:nrow(outliers), .combine = rbind)

    # Logic: reorder names in combo based on contribution?
    # R:
    # f <- function (n) {
    #   dif.rel <- tmp[tmp[, "combo"] %in% n,]$dif.rel
    #   return (abs(ifelse(is.null(dif.rel), 0, dif.rel)))
    # }
    # ord <- unlist(sapply(unlist(old_combo), f))
    # tmp <- list(unlist(old_combo)[order(-ord)])
    # outlier[1] <- p_get_names(unlist(tmp), k)$outliers

    if org_outliers is not None:
        # We need to process outliers to reorder names
        # This seems complex to port exactly without exact data structures.
        # Let's try.

        # We need to iterate over current outliers and use org_outliers to sort combo elements.
        # org_outliers is a DataFrame.

        new_outliers_list = []
        for idx, outlier in outliers.iterrows():
            old_combo = outlier["combo"]  # This is a list/array of names

            # We need to find dif.rel for each element in old_combo from org_outliers
            # org_outliers has 'combo' column which is a list of 1 element (since k=1 for org_outliers)
            # or just the name?
            # p_get_outlier returns 'combo' as the list of dropped rows.
            # For k=1, it's a list of 1.

            # Helper to get dif_rel for a single name n
            def get_dif_rel(n):
                # Find row in org_outliers where combo contains n
                # org_outliers['combo'] is a column of lists/arrays
                # We assume org_outliers was created with k=1, so combo is [n]

                # We can create a map for speed
                # But let's just search
                match = org_outliers[org_outliers["combo"].apply(lambda x: x[0] == n)]
                if not match.empty:
                    return abs(match.iloc[0]["dif_rel"])
                return 0

            ord_vals = [get_dif_rel(n) for n in old_combo]
            # Sort old_combo based on ord_vals descending
            # zip, sort, unzip
            sorted_combo = [
                x
                for _, x in sorted(zip(ord_vals, old_combo), key=lambda pair: pair[0], reverse=True)
            ]

            # Update name
            outlier["outliers"] = p_get_names(sorted_combo, k)["outliers"]
            new_outliers_list.append(outlier)

        outliers = pd.DataFrame(new_outliers_list)

    p_cluster_cleanup()

    if org_outliers is not None:
        return pd.concat([org_outliers, outliers], ignore_index=True)
    return outliers


def p_get_outlier_wrapper(data, combo, params, k):
    return p_get_outlier(data, combo, params, k)


def p_get_all_names(data, peers, global_scope, params, k):
    """Get all potential outlier names from peers.

    Note: data, global_scope, and k are kept for API compatibility with R version
    but not currently used in the simplified implementation.
    """
    _ = (data, global_scope, params, k)  # Mark as intentionally unused
    all_names = list(peers.index)
    return all_names


def p_get_combos(data, params, k):
    # For COLS and QR we need all the points
    if params["ceiling"][0] in P_NO_PEER_LINE:  # params['ceiling'] is a list?
        # return(t(combn(rownames(data), k)))
        from itertools import combinations

        return list(combinations(data.index, k))

    # all.names <- p_get_all_names(...)
    all_names = p_get_all_names(data, params["peers"], params["global"], params, k)

    counter = k
    while counter > 1:
        # data <- data[!(row.names(data) %in% all.names),]
        data = data.drop(all_names, errors="ignore")

        model = nca_analysis(
            data,
            params["x"],
            params["y"],
            ceilings=params["ceiling"],
            corner=params["corner"],
            flip_x=params["flip_x"],
            flip_y=params["flip_y"],
            scope=params["scope"],
        )

        global_scope = model["summaries"][params["x"]]["global"]
        # all.names <- c(all.names, p_get_all_names(...))
        new_names = p_get_all_names(
            data, p_aggregate_peers(model["peers"], 1), global_scope, params, k
        )
        all_names.extend(new_names)

        counter -= 1

    all_names = list(set(all_names))  # unique
    if k == 1:
        # return (matrix(all.names, ncol = 1))
        return [[n] for n in all_names]

    # combos <- NULL
    # counter <- k
    # while (counter > 1) ...
    # This logic generates combinations of size k from all_names,
    # but also combinations of size < k padded with NA?
    # R: tmp <- cbind(tmp, matrix(NA, ncol = k - counter, nrow = nrow(tmp)))
    # This suggests we look for outliers of size k, k-1, ... 2?
    # But the loop goes counter > 1.

    combos = []
    from itertools import combinations

    counter = k
    while counter > 1:
        tmp = list(combinations(all_names, counter))
        # Pad with None
        padded = [list(c) + [None] * (k - counter) for c in tmp]
        combos.extend(padded)
        counter -= 1

    return combos


def p_get_outlier(data, combo, params, k):
    # combo is a list/tuple
    combo = [c for c in combo if c is not None]
    # data.new <- data[-(which(rownames(data) %in% combo)),]
    data_new = data.drop(combo, errors="ignore")

    values = p_get_values(data_new, params)
    eff_nw, dif_abs, dif_rel, global_new = values

    if round(abs(dif_rel), 2) < params["min_dif"]:
        return None

    zone_scope = p_zone_scope(combo, params, global_new)
    zone, scope = zone_scope

    # Extra check: 'ceiling' outliers must be on COLS and C_LP lines
    # if (all(k == 1, params$ceiling[1] %in% c("cols", "c_lp"), scope == ""))
    ceiling_type = params["ceiling"]
    if isinstance(ceiling_type, list):
        ceiling_type = ceiling_type[0]

    if k == 1 and ceiling_type in ["cols", "c_lp"] and scope == "":
        # line <- params$model$plots[[1]]$lines[[1]]
        # We need to access the line parameters from the plot object or summary.
        # In Python nca_analysis returns plots dict.
        # We need to know structure of plot object.
        pass  # Placeholder

    names = p_get_names(combo, k)
    result = {
        "outliers": names["outliers"],
        "eff_or": params["eff_or"],
        "eff_nw": eff_nw,
        "dif_abs": dif_abs,
        "dif_rel": dif_rel,
        "ceiling": zone,
        "scope": scope,
        "combo": combo,
    }
    return result


def p_get_values(data_new, params):
    model_new = nca_analysis(
        data_new,
        params["x"],
        params["y"],
        ceilings=params["ceiling"],
        corner=params["corner"],
        flip_x=params["flip_x"],
        flip_y=params["flip_y"],
        scope=params["scope"],
    )

    # eff.nw <- model.new$summaries[[1]]$params[2]
    # summary['params'] is a DataFrame. Row "Effect size" is index 1.
    eff_nw = model_new["summaries"][params["x"]]["params"].iloc[1, 0]

    dif_abs = 0 if eff_nw is None or np.isnan(eff_nw) else eff_nw - params["eff_or"]

    zero_dif_rel = 0 if dif_abs < EPSILON else float("inf")

    if params["eff_or"] < EPSILON:
        dif_rel = zero_dif_rel
    else:
        dif_rel = 100 * dif_abs / params["eff_or"]

    global_new = model_new["summaries"][params["x"]]["global"]
    return [eff_nw, dif_abs, dif_rel, global_new]


def p_zone_scope(combo, params, global_new):
    """Determine if outlier affects ceiling zone and/or scope."""
    _ = global_new  # Mark as intentionally unused (kept for API compatibility)
    found_scope = False  # Simplified: scope checking not fully implemented

    ceiling_type = params["ceiling"]
    if isinstance(ceiling_type, list):
        ceiling_type = ceiling_type[0]

    if ceiling_type in P_NO_PEER_LINE:
        found_ceiling = True
    else:
        # found_ceiling <- any(combo %in% rownames(params$peers))
        # params['peers'] is a DataFrame of peers
        found_ceiling = any(c in params["peers"].index for c in combo)

    return ["X" if found_ceiling else "", "X" if found_scope else ""]


def p_get_names(combo, k):
    if k == 1:
        name = combo[0]
    else:
        name = " - ".join(str(c) for c in combo)
    return {"outliers": name}


def p_format_outliers(outliers, max_results, k, min_dif, condensed):
    if outliers is None or len(outliers) == 0:
        return None

    # outliers is a DataFrame

    # for (i in 1:7) { outliers[[i]] <- unlist(outliers[[i]]) }
    # In Python, they are already columns.

    # Add length column
    # for (row_idx in 1:nrow(outliers)) {
    #   len <- length(unlist(strsplit(outliers[row_idx, 1], ' - ', fixed=T)))
    #   outliers[row_idx, 9] <- len
    # }
    # Column 1 is 'outliers' (name).
    outliers["len"] = outliers["outliers"].apply(lambda x: len(str(x).split(" - ")))

    # outliers <- outliers[order(-abs(outliers$dif.rel), -abs(outliers$dif.abs), outliers[, 9]),]
    outliers["abs_dif_rel"] = outliers["dif_rel"].abs()
    outliers["abs_dif_abs"] = outliers["dif_abs"].abs()

    outliers = outliers.sort_values(
        by=["abs_dif_rel", "abs_dif_abs", "len"], ascending=[False, False, True]
    )

    org_length = len(outliers)

    if k != 1:
        if len(outliers) > 1 and condensed:
            keep = [0]  # Keep first
            # Remove outliers which are not larger than previous (but keep singles)
            for row_idx in range(1, len(outliers)):
                parts = str(outliers.iloc[row_idx]["outliers"]).split(" - ")
                current = abs(outliers.iloc[row_idx]["dif_rel"])
                prev = abs(outliers.iloc[row_idx - 1]["dif_rel"])

                if len(parts) == 1 or abs(current - prev) > min_dif:
                    keep.append(row_idx)

            outliers = outliers.iloc[keep]

        outliers = outliers.iloc[: min(len(outliers), max_results)]

    # outliers <- outliers[, -c(8, 9)]
    # Remove 'combo' and 'len' and temp cols
    cols_to_drop = ["combo", "len", "abs_dif_rel", "abs_dif_abs"]
    outliers = outliers.drop(columns=[c for c in cols_to_drop if c in outliers.columns])

    # Rounding
    # outliers[, c(2, 3, 4)] <- round(outliers[, c(2, 3, 4)], digits = 2)
    # Columns: outliers, eff.or, eff.nw, dif.abs, dif.rel, ceiling, scope
    # Indices: 0, 1, 2, 3, 4, 5, 6
    # R: 2, 3, 4 -> eff.or, eff.nw, dif.abs
    cols_to_round_2 = ["eff_or", "eff_nw", "dif_abs"]
    for c in cols_to_round_2:
        if c in outliers.columns:
            outliers[c] = outliers[c].astype(float).round(2)

    # outliers[, 5] <- round(outliers[, 5], digits = 1) -> dif.rel
    if "dif_rel" in outliers.columns:
        outliers["dif_rel"] = outliers["dif_rel"].astype(float).round(1)

    if org_length > len(outliers):
        outliers.attrs[SHOWN] = len(outliers)
        outliers.attrs[HIDDEN] = org_length - len(outliers)

    # class(outliers) <- c("outliers", class(outliers))
    # We can't easily change class of DataFrame, but we can use attrs or a wrapper.

    return outliers
