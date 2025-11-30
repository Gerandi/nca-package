from .p_bottleneck import p_bottleneck_ce
from .p_ceiling import p_ce_ceiling
from .p_fit import get_fit
from .p_ineffs import p_ineffs_ce
from .p_peers import p_peers


def p_nca_ce_vrs(loop_data, bn_data):
    flip_x = loop_data["flip_x"]
    flip_y = loop_data["flip_y"]

    peers = p_peers(loop_data, vrs=True)

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


def p_vrs_line(loop_data, peers, flip_x, flip_y):
    x_start = loop_data["scope_emp"][0] if not flip_x else loop_data["scope_emp"][1]
    x_end = loop_data["scope_theo"][1] if not flip_x else loop_data["scope_theo"][0]

    y_start = loop_data["scope_theo"][2] if not flip_y else loop_data["scope_theo"][3]
    y_end = loop_data["scope_emp"][3] if not flip_y else loop_data["scope_emp"][2]

    x_points = [x_start]
    y_points = [y_start]

    if peers is not None and len(peers) > 0:
        x_points.extend(peers.iloc[:, 0].tolist())
        y_points.extend(peers.iloc[:, 1].tolist())

    x_points.append(x_end)
    y_points.append(y_end)

    return [x_points, y_points]
