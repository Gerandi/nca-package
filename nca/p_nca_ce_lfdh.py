from .p_bottleneck import p_bottleneck_ce
from .p_ceiling import p_ce_ceiling
from .p_fit import get_fit
from .p_ineffs import p_ineffs_ce
from .p_nca_ce_vrs import p_vrs_line
from .p_peers import p_peers


def p_nca_ce_lfdh(loop_data, bn_data):
    flip_x = loop_data["flip_x"]
    flip_y = loop_data["flip_y"]

    peers = p_peers(loop_data)

    line = p_vrs_line(loop_data, peers, flip_x, flip_y)
    ceiling = p_ce_ceiling(loop_data, peers, "vrs")
    effect = ceiling / loop_data["scope_area"]
    ineffs = p_ineffs_ce(loop_data, peers)
    bottleneck = p_bottleneck_ce(loop_data, bn_data, peers, "vrs")
    fit = get_fit(ceiling, loop_data.get("ce_fdh_ceiling", float("nan")))

    return {
        "line": line,
        "peers": peers,
        "slope": float("nan"),
        "intercept": float("nan"),
        "ceiling": ceiling,
        "effect": effect,
        "above": 0,
        "accuracy": 100,
        "fit": fit,
        "ineffs": ineffs,
        "bottleneck": bottleneck,
    }
