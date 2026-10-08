from __future__ import annotations

import logging

import matplotlib as mpl
import matplotlib.colors as mcolors

TITLE_FS = 10
AXIS_LABEL_FS = 9
TICK_FS = 8
LEGEND_FS = 8
COLORBAR_LABEL_FS = 8
COLORBAR_TICK_FS = 7
ANNOTATION_FS = 8

ORANGE = "#DD8D6E"
BEIGE = "#F0E8D1"
GREEN = "#558771"
TEAL = "#82A899"
PURPLE = "#885784"
GOLD = "#C2AB42"
GRAY = "#555555"

CANCER_COLORS = [
    "#a84059",
    "#9571ab",
    "#e0cc9b",
    "#82a899",
    "#c47a45",
    "#8076ab",
    "#4a4270",
    "#e98d7d",
    "#cc8711",
    "#6f9c9a",
    "#f0e8d1",
]


def make_linear_cmap(colors, name="custom_cmap", N=256):
    return mcolors.LinearSegmentedColormap.from_list(name, colors, N=N)


CMAP_ORANGE_GREEN = make_linear_cmap([ORANGE, BEIGE, GREEN], "orange_green")
CMAP_BEIGE_PURPLE = make_linear_cmap([BEIGE, GOLD, PURPLE], "beige_purple")
CMAP_GOLD_PURPLE = make_linear_cmap([GOLD, "#f2dac6", PURPLE], "gold_purple")

mpl.rcParams["figure.facecolor"] = "none"
mpl.rcParams["axes.facecolor"] = "none"
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False

logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
