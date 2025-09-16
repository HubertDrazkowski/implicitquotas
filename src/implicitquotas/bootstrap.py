from __future__ import annotations

# ─── standard library ─────────────────────────────────────────────────────
import logging
import os
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List

# ─── third-party ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy.stats import binom
from joblib import Parallel, delayed
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt


# ─── local (same-package) ─────────────────────────────────────────────────
from .core import _validate_columns
#################
def panel_bootstrap(
        df: pd.DataFrame,
        id_col: str,
        female_col: str,
        male_col: str,
        p_col: str,
        year_col: Optional[str] = None,
        appointments_col: Optional[str] = None,
        resignations_col: Optional[str] = None,
        agg: int = 6,
        B: int = 1000,
        init_with_observed: bool = True,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
        panel: bool = True,
        return_empirical: bool = False
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Perform either a cross-sectional or panel-based bootstrap of the sum
        B*(z)_b = sum_{i,t} [ H_{i,t}*(z) - p_{i,t}(z) ]
    for z = 0,...,agg, over B independent draws.

    If panel=True (default), runs the full panel-based bootstrap:
      - Requires year_col, appointments_col, resignations_col.
      - Simulates hires and departures year-by-year per firm.

    If panel=False, runs a simple cross-sectional bootstrap:
      - Ignores year, appointments, and resignations.
      - For each firm, draws #female ~ Binomial(n_s, p_s) once.

    Parameters
    ----------
    df : pd.DataFrame
      Must contain, at least, one row per (firm‐year) with columns:
        • id_col            (unique firm identifier, e.g. 'id_bvd'),
        • year_col          (calendar year),
        • female_col        (# of female members observed at that year),
        • male_col          (# of male members observed at that year),
        • p_col             (probability of hiring a woman at that (firm,year), in [0,1]),
        • appointments_col  (number of new hires in that firm‐year),
        • resignations_col  (number of resignations in that firm‐year).

      The columns `female_col` + `male_col` define the observed unit of observation size
      \(n_{i,t} = \textit{female\_col} + \textit{male\_col}\).  We use those
      \((n_{i,t},p_{i,t})\) to compute the “null” binomial‐probabilities
      \(p_{i,t}(z)\).

    id_col, year_col, female_col, male_col, p_col, appointments_col, resignations_col : str
      Names of the corresponding columns in `df`.

    agg : int, default=6
      We treat “\(z<\textit{agg}\)” via `binom.pmf(z,n,p)` and “\(z=\textit{agg}\)” via
      `binom.sf(agg‐1,n,p)`.

    B : int, default=1000
      Number of bootstrap draws.

    init_with_observed : bool, default=True
      If True, each firm’s initial composition at time = min(year) is set equal to the
      *observed* \((\textit{female},\textit{male})\) in `df`.  If False, we re‐sample
      the initial composition via
         M_{i,1}^F ~ Binomial(n_{i,1}, p_{i,1})
         M_{i,1}^M = n_{i,1} - M_{i,1}^F.

    random_state : int or None
      If not None, used to seed a top‐level `np.random.RandomState(random_state)`.  Each
      of the B draws will get its own child‐seed from that.

    n_jobs : int, default=1
      Number of parallel jobs (over the B draws).  If 1, runs single‐threaded.

    Returns
    -------
    pd.DataFrame of shape (B, agg+1).  Columns are named 0,1,2,...,agg.  Each row `b`
    holds
        [ B*(0)_b, B*(1)_b, ..., B*(agg)_b ].

    Raises
    ------
    ValueError
      • if any of the required columns is missing from `df`,
      • or if `df` is empty,
      • or if `B < 1`, etc.
    """

    # Input validation
    if not is_numeric_dtype(df[p_col]):
        raise ValueError(f"{p_col} must be numeric probability values in [0,1].")
    # Negative probabilities not allowed
    if (df[p_col] < 0).any():
        raise ValueError("Negative probabilities are allowed only in quantum mechanics!")
    # Probabilities above 1: assume percentage, divide by 100
    if (df[p_col] > 1).any():
        import warnings
        warnings.warn("Detected probability values > 1; converting by dividing by 100.")
        df.loc[:,p_col] = df.loc[:,p_col] / 100

    if B < 1:
        raise ValueError("B must be at least 1.")
    if df.empty:
        raise ValueError("Input DataFrame `df` is empty.")

    df = df.dropna(subset=[female_col])
    df.loc[:, female_col] = df.loc[:, female_col].astype(int)

    df = df.dropna(subset=[male_col])
    df.loc[:, male_col] = df.loc[:, male_col].astype(int)

    # Helper to compute empirical B(z)
    # Helper to compute empirical B(z)
    def _compute_empirical(data: pd.DataFrame, per_firm_data=None, cross_section=False) -> pd.Series:
        """
        Compute empirical B(z) statistic.
        If cross_section=True, treats each row as independent; otherwise uses panel aggregation.
        """
        # Cross-sectional empirical
        if cross_section:
            B_emp = np.zeros(agg + 1, dtype=float)
            for _, row in data.iterrows():
                n_s = int(row[female_col] + row[male_col])
                p_s = float(row[p_col])
                m_obs = int(row[female_col])
                for z in range(agg):
                    B_emp[z] += (int(m_obs == z) - binom.pmf(z, n_s, p_s))
                B_emp[agg] += (int(m_obs >= agg) - binom.sf(agg - 1, n_s, p_s))
            return pd.Series(B_emp, index=[str(z) for z in range(agg)] + [str(agg)])

        # Panel empirical: build helper DataFrame
        df2 = data.copy().reset_index(drop=True)
        df2['n_d'] = (df2[female_col] + df2[male_col]).astype(int)
        df2['p_s'] = df2[p_col].astype(float)
        df2['Y_d'] = df2[female_col].astype(int)
        group_vars = [id_col, year_col]
        B_emp = []
        # Compute empirical for each z
        for z in range(agg + 1):
            if z < agg:
                df2['H'] = (df2['Y_d'] == z).astype(int)
                df2['p_z'] = binom.pmf(z, df2['n_d'], df2['p_s'])
            else:
                df2['H'] = (df2['Y_d'] >= agg).astype(int)
                df2['p_z'] = binom.sf(agg - 1, df2['n_d'], df2['p_s'])
            grp = df2.groupby(group_vars).agg(
                H_sum=pd.NamedAgg(column='H', aggfunc='sum'),
                p_sum=pd.NamedAgg(column='p_z', aggfunc='sum')
            )
            total_H = grp['H_sum'].sum()
            total_p = grp['p_sum'].sum()
            B_emp.append(total_H - total_p)
        return pd.Series(B_emp, index=[str(z) for z in range(agg)] + [str(agg)])

    # Cross-sectional logic
    if not panel:
        # Ensure only one row per id in cross-sectional mode
        counts = df.groupby(id_col).size()
        multi = counts[counts > 1]
        if not multi.empty:
            dup_ids = multi.index.tolist()
            raise ValueError(
                f"Cross-sectional mode requires one row per id, but found multiple for ids: {dup_ids}"
            )
        _validate_columns(df, [id_col, female_col, male_col, p_col])
        # Vectorized cross-sectional bootstrap
        # Extract arrays of sizes and probabilities
        firm_df = df.set_index(id_col)
        n_arr = (firm_df[female_col] + firm_df[male_col]).astype(int).to_numpy()
        p_arr = firm_df[p_col].astype(float).to_numpy()
        rng = np.random.RandomState(random_state)
        # Draw matrix of shape (B, num_firms)
        M = rng.binomial(n_arr[np.newaxis, :], p_arr[np.newaxis, :], size=(B, len(n_arr)))
        # Compute null pmfs once per z
        pmf = np.vstack([binom.pmf(z, n_arr, p_arr) for z in range(agg)] +
                        [binom.sf(agg - 1, n_arr, p_arr)])  # shape (agg+1, num_firms)
        # Count occurrences and subtract null
        # counts_z[b] = sum_i 1{M[b,i]==z}; then Bz[b,z] = counts_z[b] - sum_i pmf[z,i]
        boot_arr = np.zeros((B, agg + 1))
        for z in range(agg + 1):
            counts_z = (M == z if z < agg else M >= agg).sum(axis=1)
            boot_arr[:, z] = counts_z - pmf[z].sum()
        boot_df = pd.DataFrame(boot_arr, columns=[str(z) for z in range(agg)] + [str(agg)])
        emp = _compute_empirical(df, cross_section=True) if return_empirical else None

        return boot_df, emp

    # Panel logic

    req = [id_col, year_col, female_col, male_col, p_col, appointments_col, resignations_col]
    req = [id_col, year_col, female_col, male_col, p_col, appointments_col, resignations_col]
    _validate_columns(df, req)
    df_sorted = df.sort_values([id_col, year_col])
    firm_groups = {fid: g.reset_index(drop=True) for fid, g in df_sorted.groupby(id_col)}

    #     years_arr = np.stack([g[year_col].to_numpy() for g in firm_groups.values()])
    #     n_arrs    = np.stack([(g[female_col]+g[male_col]).to_numpy() for g in firm_groups.values()]).astype(int)
    #     p_arrs    = np.stack([g[p_col].to_numpy() for g in firm_groups.values()]).astype(float)
    #     A_arrs    = np.stack([g[appointments_col].to_numpy() for g in firm_groups.values()]).astype(int)
    #     R_arrs    = np.stack([g[resignations_col].to_numpy() for g in firm_groups.values()]).astype(int)
    #     f0_arr    = np.array([int(g[female_col].iloc[0]) for g in firm_groups.values()])
    #     m0_arr    = np.array([int(g[male_col].iloc[0])   for g in firm_groups.values()])

    #     # Generate seeds
    #     rng = np.random.RandomState(random_state)
    #     seeds = rng.randint(0, 2**31-1, size=B)
    #     boot_arr = _draw_panel_jit(
    #         seeds,
    #         years_arr, n_arrs, p_arrs, A_arrs, R_arrs,
    #         f0_arr, m0_arr,
    #         agg,
    #         init_with_observed
    #         )

    # boot_df = pd.DataFrame(boot_arr, columns=[str(z) for z in range(agg+1)])
    # emp = _compute_empirical(df, cross_section=False) if return_empirical else None
    # return boot_df, emp

    per_firm_data = []
    for fid, g in firm_groups.items():
        years = g[year_col].to_numpy()
        n_arr = (g[female_col] + g[male_col]).to_numpy().astype(int)
        p_arr = g[p_col].to_numpy().astype(float)
        A_arr = g[appointments_col].to_numpy().astype(int)
        R_arr = g[resignations_col].to_numpy().astype(int)
        f0 = int(g[female_col].iloc[0])
        m0 = int(g[male_col].iloc[0])
        # Include firm_id for empirical lookup
        per_firm_data.append((fid, years, n_arr, p_arr, A_arr, R_arr, f0, m0))

    def _one_draw(seed):
        rng = np.random.RandomState(seed)
        Bz = np.zeros(agg + 1)
        # Iterate over each firm's data
        for (firm_id, years, n_arr, p_arr, A_arr, R_arr, f0, m0) in per_firm_data:
            M_F = f0 if init_with_observed else int(rng.binomial(n_arr[0], p_arr[0]))
            M_M = m0 if init_with_observed else (n_arr[0] - M_F)
            for t in range(len(years)):
                n_t, p_t, A_t, R_t = n_arr[t], p_arr[t], A_arr[t], R_arr[t]
                if A_t > 0:
                    newf = rng.binomial(A_t, p_t)
                    M_F += newf
                    M_M += (A_t - newf)
                if R_t > 0 and (M_F + M_M) > 0:
                    dropf = rng.binomial(R_t, M_F / (M_F + M_M))
                    M_F = max(0, M_F - dropf)
                    M_M = max(0, M_M - (R_t - dropf))
                # Record contributions
                for z in range(agg):
                    Bz[z] += (int(M_F == z) - binom.pmf(z, n_t, p_t))
                Bz[agg] += (int(M_F >= agg) - binom.sf(agg - 1, n_t, p_t))
        return Bz

    rng_top = (np.random.RandomState(random_state)
               if random_state is not None else np.random.RandomState())
    seeds = rng_top.randint(0, 2 ** 31 - 1, size=B)
    #    pdb.set_trace()
    all_Bz = [_one_draw(s) for s in seeds] if n_jobs == 1 else Parallel(n_jobs=n_jobs)(
        delayed(_one_draw)(s) for s in seeds)
    boot_df = pd.DataFrame(all_Bz, columns=[str(z) for z in range(agg)] + [str(agg)])
    emp = _compute_empirical(df, per_firm_data=per_firm_data, cross_section=False) if return_empirical else None
    return boot_df, emp


def plot_bootstrap(
        boot_df: pd.DataFrame,
        empirical: pd.Series,
        z: int = 0,
        alpha: float = 0.05,
        ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the bootstrap distribution for index z, with whiskers at the (alpha/2, 1-alpha/2) quantiles,
    and a vertical line for the empirical statistic (in red).
    """
    col = str(z)
    data = boot_df[col]
    lower, upper = data.quantile(alpha / 2), data.quantile(1 - alpha / 2)

    fig, ax = (plt.subplots() if ax is None else (ax.figure, ax)[::-1])
    ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(empirical[col], color='red', linestyle='--', linewidth=2, label='Empirical')
    ax.axvline(lower, linestyle=':', label=f'{100 * alpha / 2:.1f}% quantile')
    ax.axvline(upper, linestyle=':', label=f'{100 * (1 - alpha / 2):.1f}% quantile')
    ax.set_title(f'Bootstrap distribution for z={z}')
    ax.set_xlabel('B*(z)')
    ax.set_ylabel('Frequency')
    ax.legend()
    return fig, ax


def analyze_bootstrap(
        df: pd.DataFrame,
        id_col: str,
        female_col: str,
        male_col: str,
        p_col: str,
        agg: int,
        z: Union[int, List[int]],
        alpha: float = 0.05,
        panel: bool = True,
        year_col: Optional[str] = None,
        appointments_col: Optional[str] = None,
        resignations_col: Optional[str] = None,
        B: int = 1000,
        init_with_observed: bool = True,
        random_state: Optional[int] = None,
        n_jobs: int = 1
) -> Dict[str, Any]:
    """
    Run the bootstrap, compute p-value for empirical B(z), decide reject/fail, and plot.

    Returns
    -------
    result : dict with keys:
      - 'boot_df', 'empirical', 'p_value', 'reject', 'fig'
    """
    # Ensure z is a list
    if isinstance(z, int):
        z_list = [z]
    else:
        z_list = list(z)

    # Run bootstrap and get empirical
    boot_df, empirical = panel_bootstrap(
        df, id_col, female_col, male_col, p_col,
        year_col=year_col, appointments_col=appointments_col,
        resignations_col=resignations_col, agg=agg, B=B,
        init_with_observed=init_with_observed, random_state=random_state,
        n_jobs=n_jobs, panel=panel, return_empirical=True)

    results = {}
    figs = {}

    for z_val in z_list:
        sim = boot_df[str(z_val)]
        emp_val = empirical[str(z_val)]

        p_lower = np.mean(sim <= emp_val)
        p_upper = np.mean(sim >= emp_val)
        p_value = 2 * min(p_lower, p_upper)
        reject = p_value < alpha

        results[int(z_val)] = {
                        "empirical": emp_val,
                        "p_lower": p_lower,
                        "p_upper": p_upper,
                        "p_value": p_value,
                        "reject": reject
                        }

        # Make plot for this z
        fig, ax = plot_bootstrap(boot_df, empirical, z=z_val, alpha=alpha)
        figs[z_val] = fig

    return {
        "boot_df": boot_df,
        "empirical": empirical,
        "results": results,
        "figs": figs
    }


def plot_summary(
        boot_df: pd.DataFrame,
        empirical: pd.Series,
        alpha: float = 0.05,
        ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot empirical B(z) with bootstrap confidence bands across z=0..agg.
    Draw only the empirical points (●) and the upper (▲) / lower (▼) quantile markers.
    """
    zs = np.arange(len(empirical))
    lower = boot_df.quantile(alpha / 2)
    upper = boot_df.quantile(1 - alpha / 2)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # plot lower bound as ▼
    ax.plot(zs, lower.values,
            linestyle='None', marker='v', markersize=8,
            label=f'{100 * alpha / 2:.1f}% quantile')
    # empirical as ●
    ax.plot(zs, empirical.values,
            linestyle='None', marker='o', markersize=8,
            color='black', label='Empirical')
    # upper bound as ▲
    ax.plot(zs, upper.values,
            linestyle='None', marker='^', markersize=8,
            label=f'{100 * (1 - alpha / 2):.1f}% quantile')

    # **Here’s the key addition**: force xticks = zs
    ax.set_xticks(zs)
    ax.set_xticklabels([str(int(z)) for z in zs])

    ax.set_xlabel('Number of women per Department')
    ax.set_ylabel('Deviation from expected value')
    ax.set_title(f'Including bootstrapped {100 * (1 - alpha)}% bands')
    ax.legend()
    return fig, ax





















