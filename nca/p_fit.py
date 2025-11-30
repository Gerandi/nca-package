import math


def get_fit(ceiling, fdh_ceiling):
    if math.isnan(ceiling) or math.isnan(fdh_ceiling):
        return float("nan")

    if ceiling > fdh_ceiling:
        return float("nan")

    return 100 - 100 * abs(ceiling - fdh_ceiling) / fdh_ceiling
