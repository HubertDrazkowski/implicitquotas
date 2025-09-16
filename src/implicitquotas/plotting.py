from __future__ import annotations
import os
import logging
from pathlib import Path
from typing import (
    Optional,
    Tuple
)
import seaborn as sns

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import Normalize

def plot_heatmap(
        df: pd.DataFrame,
        *,
        value_col: str = "T_n(z)",
        p_col: str = "adjusted_p_value",
        var_col: str = "Variable",
        z_col: str = "z",
        title: str = "",
        alpha: float = 0.05,
        vlim: Tuple[float, float] = (-3.0, 3.0),
        cmap: Optional[str],
        save_dir: Optional[os.PathLike] = None,
        filename: str | None = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generic heat–map of a test-statistic with a significance mask.

    Parameters
    ----------
    df : DataFrame
        Must contain at least the four columns given by *var_col*, *z_col*,
        *value_col*, *p_col*.
    value_col, p_col, var_col, z_col : str
        Column names in *df* (override if different).
    title : str
        Y–axis label and figure title.
    alpha : float
        Significance level used to mask non–significant cells.
    vlim : (low, high)
        Colour-bar limits (centre is 0).
    cmap : seaborn / matplotlib colormap
        `sns.diverging_palette(...)` by default.
    save_dir, filename :
        If *save_dir* is given, figure is saved there.  If *filename* is
        omitted a sensible name is generated automatically.

    Returns
    -------
    (fig, ax)
    """
    # ---- guard clauses -------------------------------------------------------
    needed = {value_col, p_col, var_col, z_col}
    missing = needed.difference(df.columns)
    if missing:
        raise KeyError(f"Missing column(s): {missing!r}")

    if df.empty:
        raise ValueError("`df` is empty – nothing to plot.")

    # ---- pivot & mask --------------------------------------------------------
    pivot_val = df.pivot(index=var_col, columns=z_col, values=value_col)
    pivot_p   = df.pivot(index=var_col, columns=z_col, values=p_col)
    mask      = pivot_p > alpha

    if cmap is None:          # default diverging palette
        cmap = sns.diverging_palette(240, 5, s=50, l=60, n=3, as_cmap=True)

    # ---- figure size scales with #z ------------------------------------------
    n_z       = pivot_val.shape[1]
    full_cols = 6             # “max” number of z when agg=5
    base_w    = 12
    fig_w     = max(3, base_w * n_z / full_cols)
    fig_h     = 8

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        pivot_val,
        mask=mask,
        cmap=cmap,
        center=0,
        vmin=vlim[0],
        vmax=vlim[1],
        linewidths=.5,
        cbar=False,
        ax=ax
    )

    # ---- labels & ticks ------------------------------------------------------
    ax.set_xlabel("Number of women (z)", fontsize=16, labelpad=10)
    ax.set_ylabel(title,              fontsize=16, labelpad=10)
    ax.tick_params(axis='x', labelsize=14, rotation=0)
    ax.tick_params(axis='y', labelsize=14)

    # add “+” on last x-tick
    xt = [f"{int(t)}+" if i == len(ax.get_xticks() ) -2 else f"{int(t)}"
          for i, t in enumerate(ax.get_xticks())]
    ax.set_xticklabels(xt)

    # ---- 3-patch legend ------------------------------------------------------
    norm   = plt.Normalize(*vlim)
    few    = cmap(norm(vlim[0]))
    many   = cmap(norm(vlim[1]))
    middle = cmap(norm(0))

    legend_patches = [
        Patch(facecolor=few,    edgecolor='k', label="Improbably few"),
        Patch(facecolor=middle, edgecolor='k', label="Not rejected"),
        Patch(facecolor=many,   edgecolor='k', label="Improbably many"),
    ]
    ax.legend(handles=legend_patches,
              bbox_to_anchor=(1.02, 0.5),
              loc="center left",
              frameon=False,
              fontsize=14)

    fig.tight_layout()

    # ---- optional save -------------------------------------------------------
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"{title.replace(' ', '_').lower()}_heatmap.png"
        fig.savefig(Path(save_dir) / filename, dpi=300, bbox_inches="tight")

    return fig, ax

MAINSEC_CODES = set('BNJMRFGLDCH OASPKIQETU'.split()) | {''}   # quick lookup

def _detect_bucket(var, val):
    """
    Decide which heat-map the slice goes into.
    """
    var_low = str(var).lower()

    # 1) explicit variable name wins
    if var_low in {"year", "years"}:            return "years"
    if var_low in {"mainsec", "sector"}:        return "mainsec"
    if var_low in {"country", "nation"}:        return "countries"

    # 2) otherwise infer from value
    if isinstance(val, (int, np.integer)):      return "years"
    if isinstance(val, str) and val in MAINSEC_CODES:
        return "mainsec"

    # 3) fallback bucket (one per unknown variable)
    return var_low


def plot_zoomed_results(
        all_results: dict,
        *,
        value_col   : str = "T_n",           # <-- change if your column is T_n(z)
        p_col       : str = "adjusted_p_value",
        save_dir    : os.PathLike | None = None,
        file_prefix : str = ""
):
    """
    Build one heat-map per detected bucket (countries / years / mainsec / …).

    Parameters
    ----------
    all_results : dict
        Dict returned by `zoom_in_analysis` (keys of arbitrary length).
    value_col, p_col : str
        Column names inside every result-DataFrame.
    save_dir : path or None
        If given: figures are saved there.
    file_prefix : str
        Prepended to every file name if *save_dir* is supplied.
    """
    # ------------------------------------------------ aggregate rows ----------
    bucket_rows = defaultdict(list)

    for key, df in all_results.items():
        var, val = key[0], key[1]           # first 2 components are guaranteed
        bucket   = _detect_bucket(var, val)

        if {value_col, p_col}.difference(df.columns):
            # silently skip slices that do not yet have adj p-values
            # (or raise, if you prefer)
            continue

        tmp = df[[value_col, p_col, "z"]].copy()
        tmp["Variable"]  = val
        bucket_rows[bucket].append(tmp)

    # -------------- nothing to plot?
    if not bucket_rows:
        raise ValueError("No slice contained both "
                         f"'{value_col}' and '{p_col}' columns.")

    # ------------------------------------------------ plot per bucket ----------
    figs = {}
    for bucket, pieces in bucket_rows.items():
        data = pd.concat(pieces, ignore_index=True)

        fig, ax = _single_heatmap(
            data,
            value_col=value_col,
            p_col=p_col,
            var_col="Variable",
            z_col="z",
            title=bucket.capitalize()
        )
        figs[bucket] = (fig, ax)

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            fig.savefig(Path(save_dir) / f"{file_prefix}{bucket}_heatmap.png",
                        dpi=300, bbox_inches="tight")

    return figs

from collections import defaultdict
# ------------- tiniest wrapper around the generic heat-map above --------------
def _single_heatmap(df, *, value_col, p_col, var_col, z_col,
                    title, alpha=0.05, vlim=(-3, 3)):
    """
    Internal helper – identical idea to your earlier plot_heatmap but with the
    new flexible column names.
    """
    pivot_val = df.pivot(index=var_col, columns=z_col, values=value_col)
    pivot_p   = df.pivot(index=var_col, columns=z_col, values=p_col)
    mask      = pivot_p > alpha

    fig_w = max(3, 12 * pivot_val.shape[1] / 6)
    fig, ax = plt.subplots(figsize=(fig_w, 8))

    cmap = sns.diverging_palette(240, 5, s=50, l=60, n=3, as_cmap=True)
    sns.heatmap(pivot_val, mask=mask, cmap=cmap, center=0,
                vmin=vlim[0], vmax=vlim[1],
                cbar=False, linewidths=.5, ax=ax)

    ax.set_xlabel("Number of women (z)", fontsize=16)
    ax.set_ylabel(title,               fontsize=16)
    ax.tick_params(axis='x', labelsize=14, rotation=0)
    ax.tick_params(axis='y', labelsize=14)

    # add “+” to the last x-tick
    ticks = [f"{int(t)}+" if i == len(ax.get_xticks() ) -1 else f"{int(t)}"
             for i, t in enumerate(ax.get_xticks())]
    ax.set_xticklabels(ticks)

    # three-patch legend
    norm = plt.Normalize(*vlim)
    legend_patches = [
        Patch(facecolor=cmap(norm(vlim[0])), edgecolor='k', label="Improbably few"),
        Patch(facecolor=cmap(norm(0      )), edgecolor='k', label="Not rejected"),
        Patch(facecolor=cmap(norm(vlim[1])), edgecolor='k', label="Improbably many"),
    ]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.02, .5),
              loc="center left", frameon=False, fontsize=14)
    fig.tight_layout()
    return fig, ax