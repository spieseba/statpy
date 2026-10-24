"""Helpers shared by the plot methods: component selection, labels, legend and the common style."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes


def select_component(values: np.ndarray, component: int | tuple[int, ...]) -> np.ndarray:
    """``values[:, *component]`` as one value per row; raises unless ``component`` selects one component."""
    selected = values[(slice(None),) + np.index_exp[component]]
    if selected[0].size != 1:
        raise ValueError(f"component {component} leaves shape {selected.shape[1:]}; give one index per axis")
    return selected.reshape(len(selected))


def label_top(ax: "Axes", x: float, text: str) -> None:
    """Direct label just right of ``x`` at the top edge of ``ax``."""
    ax.annotate(text, (x, 1), xycoords=("data", "axes fraction"), xytext=(4, -2), textcoords="offset points",
                va="top", fontsize="small")


def legend_above(ax: "Axes", handles: Sequence["Artist | tuple[Artist, ...]"], labels: Sequence[str]) -> None:
    """One-row legend above the axes; its lines are thicker than thin traces, so the colors stay visible."""
    legend = ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0, 1), ncols=len(labels), frameon=False,
                       fontsize="small", handlelength=1.5)
    for line in legend.get_lines():
        line.set_linewidth(2)


def style_axes(ax: "Axes") -> None:
    """Light grid, grey frame and ticks, no top/right frame."""
    ax.grid(color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("0.75")
    ax.tick_params(color="0.75")
