from .p_bottleneck import p_bottleneck_ce
from .p_ceiling import p_ce_ceiling
from .p_confidence import p_columns, p_conf_line
from .p_fit import get_fit
from .p_ineffs import p_ineffs_ce


def p_nca_ce_cm_conf(loop_data, bn_data):
    columns = p_columns(loop_data, True)
    line = p_conf_line(columns)

    # Python: row 1 (left) and row 4 (y_max).
    peers = columns[[1, 4], :].T

    ceiling = p_ce_ceiling(loop_data, peers, "con")
    effect = ceiling / loop_data["scope_area"]
    ineffs = p_ineffs_ce(loop_data, peers)
    bottleneck = p_bottleneck_ce(loop_data, bn_data, peers, "fdh")
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
        "bottleneck": bottleneck,
        "columns": columns,
    }
