P_GLOBAL_NAMES = ["Number of observations", "Scope", "Xmin", "Xmax", "Ymin", "Ymax"]
P_RESULT_NAMES = [
    "Ceiling zone",
    "Effect size",
    "# above",
    "c-accuracy",
    "Fit",
    "p-value",
    "p-accuracy",
    " ",
    "Slope",
    "Intercept",
    "Abs. ineff.",
    "Rel. ineff.",
    "Condition ineff.",
    "Outcome ineff.",
]

P_CEILINGS_STEP = ["ce_vrs", "ce_fdh"]
# , "ce_lfdh", "ce_fdhi", "ce_cm", "ce_cm_conf")
P_CEILINGS_LINE = ["ols", "cols", "qr", "cr_vrs", "cr_fdh", "c_lp"]
# , "ct_fdh", "cr_fdhi",
#  "cr_cm", "cr_cm_conf", "c_lp")
CEILINGS = P_CEILINGS_STEP + P_CEILINGS_LINE
P_NO_BOTTLENECK = ["ols"]
# p_no_bottleneck = ["ols", "ce_cm"]
P_NO_PEER_LINE = ["cols", "qr"]

# Keep in sync with line.colors.Rd and line.type.Rd
LINE_COLORS = {
    "ols": "green",
    "c_lp": "blue",
    "cols": "darkgreen",
    "qr": "lightpink",
    "ce_vrs": "#8B4789",  # orchid4 equivalent
    "cr_vrs": "violet",
    "ce_fdh": "red",
    "cr_fdh": "orange",
}
# ce_fdh="red",       ce_lfdh="red2",       ce_fdhi="purple",
# ce_cm="darkgreen",  cr_fdh="orange",      ct_fdh="lightgreen",
# cr_fdhi="brown",    cr_cm="darkgrey",
# ce_cm_conf="black", cr_cm_conf="black",
# c_lp="lightpink",   sfa="darkgoldenrod")

LINE_TYPES = {
    "ols": 1,
    "c_lp": 2,
    "cols": 3,
    "qr": 4,
    "ce_vrs": 5,
    "cr_vrs": 1,
    "ce_fdh": 6,
    "cr_fdh": 1,
}
# ce_fdh=6,           ce_lfdh=3,            ce_fdhi=7,
# ce_cm=5,            cr_fdh=1,             ct_fdh=2,
# cr_fdhi=4,          cr_cm=7,
# ce_cm_conf=6,       cr_cm_conf=1,
# c_lp=5,             sfa=7)

LINE_WIDTH = 1.5
POINT_TYPE = 21
POINT_COLOR = "blue"

DASH_COUNT = 75

# Used to compare floats
EPSILON = 1e-10
DELTA = 1e6
