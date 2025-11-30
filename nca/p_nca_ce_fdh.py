from .p_bottleneck import p_bottleneck_ce
from .p_ceiling import p_ce_ceiling
from .p_ineffs import p_ineffs_ce
from .p_peers import p_peers


def p_nca_ce_fdh(loop_data, bn_data):
    flip_x = loop_data["flip_x"]
    flip_y = loop_data["flip_y"]

    peers = p_peers(loop_data)

    line = p_fdh_line(loop_data["scope_theo"], peers, flip_x, flip_y)
    ceiling = p_ce_ceiling(loop_data, peers, "fdh")
    effect = ceiling / loop_data["scope_area"]
    ineffs = p_ineffs_ce(loop_data, peers)
    bottleneck = p_bottleneck_ce(loop_data, bn_data, peers, "fdh")

    return {
        "line": line,
        "slope": float("nan"),
        "intercept": float("nan"),
        "ceiling": ceiling,
        "effect": effect,
        "above": 0,
        "accuracy": 100,
        "fit": 100,
        "ineffs": ineffs,
        "bottleneck": bottleneck,
        "peers": peers,
    }


def p_fdh_line(scope, peers, flip_x, flip_y):
    x_points = []
    y_points = []

    y_old = scope[3] if not flip_y else scope[2]

    if peers is not None and len(peers) > 0:
        y_new = y_old  # Initialize y_new in case loop doesn't run (though len>0 check handles it)
        for i in range(len(peers)):
            x_new = peers.iloc[i, 0]
            y_new = peers.iloc[i, 1]
            x_points.extend([x_new, x_new])
            y_points.extend([y_old, y_new])
            y_old = y_new

        x_end = scope[1] if not flip_x else scope[0]

        x_points.append(x_end)
        y_points.append(y_new)
    else:
        # If no peers, maybe just scope line?
        pass

    return [x_points, y_points]
