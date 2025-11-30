from .p_ceiling import p_cm_ceiling
from .p_confidence import p_columns, p_conf_line
from .p_fit import get_fit
from .p_ineffs import p_ineffs_ce


def p_nca_ce_cm(loop_data, bn_data):
    """CE-CM ceiling method."""
    _ = bn_data  # API compatibility
    columns = p_columns(loop_data, False)

    # Python: row 1 (left) and row 4 (y_max).
    cm_peers = columns[[1, 4], :].T

    line = p_conf_line(columns)
    ceiling = p_cm_ceiling(loop_data, cm_peers)
    effect = ceiling / loop_data["scope_area"]
    ineffs = p_ineffs_ce(loop_data, cm_peers)
    fit = get_fit(ceiling, loop_data.get("ce_fdh_ceiling", float("nan")))

    return {
        "line": line,
        "slope": float("nan"),
        "intercept": float("nan"),
        "ceiling": ceiling,
        "effect": effect,
        "above": 0,
        "accuracy": 100,
        "fit": fit,
        "ineffs": ineffs,
        "bottleneck": None,
    }
