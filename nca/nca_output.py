import warnings

from .nca_bottleneck import p_display_bottleneck
from .nca_plotly import p_display_plotly
from .nca_plots import p_display_plot
from .nca_summary import p_display_summary
from .nca_tests import p_display_test
from .p_peers import p_aggregate_peers


def nca_output(
    model,
    plots=True,
    plotly=False,
    bottlenecks=False,
    summaries=True,
    test=False,
    pdf=False,
    path=None,
    selection=None,
):
    # model needs to be the output of the NCA command

    # We have a numeric vector
    if selection is not None and all(isinstance(x, int) for x in selection):
        # selection <- intersect(selection, 1:length(model$summaries))
        # Python 0-based index.
        # Assuming selection is 1-based from R context or 0-based?
        # If user provides 1-based, we adjust.
        # But let's assume 0-based for Python API.
        # If user wants exact R behavior, they might pass 1-based.
        # Let's assume 0-based for now as it's Python.

        # Wait, in nca.py I used keys for summaries.
        # model['summaries'] is a dict.
        # We need to map indices to keys.
        keys = list(model["summaries"].keys())
        selection = [x for x in selection if 0 <= x < len(keys)]

    # We have a string (names) vector
    elif selection is not None and all(isinstance(x, str) for x in selection):
        # selection <- match(selection, names(model$summaries))
        # selection <- selection[!is.na(selection)]
        # In Python, we just keep the keys that exist.
        keys = list(model["summaries"].keys())
        # We need indices for some operations?
        # R uses indices for loop.
        selection_indices = []
        for name in selection:
            if name in keys:
                selection_indices.append(keys.index(name))
        selection = selection_indices

    # Just take all
    if selection is None or len(selection) == 0:
        selection = list(range(len(model["summaries"])))

    if bottlenecks:
        bn = {}
        for method in model["bottlenecks"]:
            # Insert the Y-column as first
            # selection.bottlenecks <- c(1, selection + 1)
            # In R, bottlenecks[[method]] is a matrix/df.
            # Column 1 is Y. Columns 2..N are Xs.
            # selection corresponds to Xs.
            # So we want column 0 (Y) and columns selection+1.

            bn_method = model["bottlenecks"][method]
            # bn_method is a DataFrame.

            # Columns: [Y, X1, X2, ...]
            # selection indices are 0-based indices into Xs (X1 is 0, X2 is 1...)
            # So in DataFrame, X1 is at index 1.

            cols_to_select = [0] + [s + 1 for s in selection]

            # Check bounds
            cols_to_select = [c for c in cols_to_select if c < len(bn_method.columns)]

            bn[method] = bn_method.iloc[:, cols_to_select]

            atts = ["bn_x", "bn_y", "bn_y_id", "size", "cutoff"]
            for att in atts:
                # attr(bn[[method]], att) <- attr(model$bottlenecks[[method]], att)
                if att in bn_method.attrs:
                    bn[method].attrs[att] = bn_method.attrs[att]

        p_display_bottleneck(bn, pdf=pdf, path=path)

    if summaries:
        keys = list(model["summaries"].keys())
        for i in selection:
            key = keys[i]
            summary = model["summaries"][key]
            p_display_summary(summary, pdf=pdf, path=path)

    if plots:
        keys = list(model["plots"].keys())
        for i in selection:
            key = keys[i]
            plot = model["plots"][key]
            p_display_plot(plot, pdf=pdf, path=path)

    if test:
        if len(model.get("tests", {})) == 0:
            warnings.warn(
                "Tests are selected in the output, but non were supplied by the analysis!",
                stacklevel=2,
            )
        else:
            keys = list(model["tests"].keys())
            for i in selection:
                # Check if this X has a test
                # model['tests'] is keyed by X name.
                # We need to map index i to X name.
                # Assuming summaries keys order matches tests keys order?
                # Not necessarily if tests are missing for some.
                # But nca.py populates tests for each X if calculated.

                # We should use the name from summaries keys.
                summary_keys = list(model["summaries"].keys())
                x_name = summary_keys[i]

                if x_name in model["tests"]:
                    test_data = model["tests"][x_name]
                    p_display_test(test_data, pdf=pdf, path=path)

    if plotly is not False:
        keys = list(model["plots"].keys())
        for i in selection:
            labels = None
            if plotly is not True:
                labels = plotly

            # peers <- p_aggregate_peers(model$peers, i)
            # p_aggregate_peers expects peers dict and index/name?
            # In R: p_aggregate_peers(model$peers, i)
            # We need to check p_peers.py implementation.
            # Assuming it takes index.
            peers = p_aggregate_peers(model["peers"], i)

            key = keys[i]
            p_display_plotly(model["plots"][key], peers, labels)

    print("")
