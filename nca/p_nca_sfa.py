import warnings


def p_nca_sfa(loop_data, bn_data):
    """SFA ceiling method - not yet implemented."""
    _ = (loop_data, bn_data)  # API compatibility
    warnings.warn("SFA is not currently implemented in the Python version of NCA.", stacklevel=2)

    return {
        "line": None,
        "ceiling": float("nan"),
        "slope": float("nan"),
        "effect": float("nan"),
        "intercept": float("nan"),
        "above": float("nan"),
        "accuracy": float("nan"),
        "fit": float("nan"),
        "ineffs": {"x": float("nan"), "y": float("nan"), "abs": float("nan"), "rel": float("nan")},
        "bottleneck": None,
    }
