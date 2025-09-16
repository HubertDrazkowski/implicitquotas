from __future__ import annotations
from typing import (
    List,
    Optional,
    Sequence,
    Tuple,
    Dict,
    Any
)
from .core import _as_list
from .core import _validate_columns
from .core import _build_p_s

import numpy as np
import pandas as pd
from scipy.stats import binom, norm
from joblib import Parallel, delayed


def _agg_for_z(
    df: pd.DataFrame,
    z: int,
    agg: int,
    *,
    group_vars: Sequence[str],
    panel: bool,
    id_col: str
) -> Dict[str, Any]:

    n = df['n_d'].to_numpy(int)
    p = df['p_s'].to_numpy(float)

    if z < agg:
        H = (df['Y_d'] == z).to_numpy(int)
        pz = binom.pmf(z, n, p)
    else:
        H = (df['Y_d'] >= agg).to_numpy(int)
        pz = binom.sf(agg - 1, n, p)

    tmp = (
        df[list(group_vars)]
        .assign(H=H, p=pz)
        .groupby(group_vars)
        .agg(H_sum=('H', 'sum'),
             p_sum=('p', 'sum'),
             sigma_sq=('p', lambda x: (x * (1 - x)).sum()))
    )

    cov = 0.0
    if panel:
        B = H - pz
        VarB = pz * (1 - pz)
        cdf = pd.DataFrame({'B': B, 'VarB': VarB, 'id': df[id_col].astype(str).values})
        cov = (
            cdf.groupby('id')
            .agg(S=('B', 'sum'),
                 B2=('B', lambda x: (x**2).sum()),
                 VarB=('VarB', 'sum'))
            .eval('Var_S = VarB + (S**2 - B2)')['Var_S']
            .sum()
        )

    num = tmp['H_sum'].sum() - tmp['p_sum'].sum()
    var = tmp['sigma_sq'].sum() + cov
    den = np.sqrt(var)
    Tn  = num / den if den > 0 else np.nan
    pval = 2 * (1 - norm.cdf(abs(Tn))) if den > 0 else np.nan

    return dict(
        z=z,
        T_n=Tn,
        numerator=num,
        p_value=pval,
        H_d_sum=tmp['H_sum'].sum(),
        p_d_sum=tmp['p_sum'].sum(),
        sigma_diag=np.sqrt(tmp['sigma_sq'].sum()),
        sigma_d_sq_sum=np.sqrt(var)
    )

def build_p_s(
    df: pd.DataFrame,
    group_vars: Sequence[str],
    *,
    female_col: str,
    male_col: str,
    external_ps_col: str | None = None,
    calculate_ps_before: bool = True,
    candidate_pool: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Return a *Series* indexed by ``group_vars`` with the null probabilities *p_s*.

    Parameters
    ----------

    df: pd.DataFrame
        Pandas data frame with needed columns
    group_vars: Sequence[str],
        Which column values combinations will create the pools of candidates
    female_col: str,
        The name of the column of second category
    male_col:
        The name of the column of one category
    external_ps_col:
        Name of a column that already stores probabilities in **percent** (0–100).
        If given, ``p_s`` is the *mean* of that column / 100 for every group.
    calculate_ps_before :
        If *True* **and** ``external_ps_col`` is *None*, compute p_s from the
        *candidate_pool* (if supplied) **before** filtering; otherwise compute
        directly on *df*.
    candidate_pool :
        A “larger” DataFrame from which to compute p_s when
        *calculate_ps_before* is *True*.

    Returns
    -------
    pd.Series
        MultiIndex = group_vars, values in [0,1].
    """
    group_vars = _as_list(group_vars)

    # ----- branch 1: external column
    if external_ps_col:
        _validate_columns(df, group_vars + [external_ps_col])
        return (
            df.groupby(group_vars)[external_ps_col]
            .mean()
            .div(100)
            .rename("p_s")
        )

    # ----- branch 2: compute from female/(female+male)
    source = candidate_pool if (calculate_ps_before and candidate_pool is not None) else df
    _validate_columns(source, group_vars + [female_col, male_col])

    sums = (
        source.groupby(group_vars)
              .agg(f_total=(female_col, "sum"),
                   m_total=(male_col,   "sum"))
    )
    p_s = (sums["f_total"] / (sums["f_total"] + sums["m_total"])).rename("p_s")
    return p_s




# ------------------------------------------------------------------
def perform_analysis(
    df: pd.DataFrme,
    p_s: pd.DataFrame | pd.Series,
    *,
    group_vars: Sequence[str],
    female_col: str,
    male_col: str,
    id_col: str,
    agg: int = 6,
    panel: bool = False,
    min_group_size: int = 10,
    n_jobs: int = 1
) -> pd.DataFrame:

    group_vars = list(group_vars)

    # --- minimal working copy
    work = df[group_vars + [female_col, male_col, id_col]].copy()
    for c in group_vars + [id_col]:
        work[c] = work[c].astype('category')

    # --- merge p_s
    if isinstance(p_s, pd.Series):
        ps_df = p_s.reset_index(name='p_s')
    else:
        if 'p_s' not in p_s.columns:
            raise ValueError("p_s DataFrame must have a 'p_s' column.")
        ps_df = p_s.copy()

    work = work.merge(ps_df, on=group_vars, how='left')

    # --- derive n_d, Y_d
    work['n_d'] = work[female_col] + work[male_col]
    work['Y_d'] = work[female_col]

    # --- filter small groups
    keep = (
        work.groupby(group_vars)
        .size()
        .loc[lambda s: s >= min_group_size]
        .index
    )
    work = work[work.set_index(group_vars).index.isin(keep)].reset_index(drop=True)
    if work.empty:
        raise ValueError("No subgroup meets min_group_size.")

    # --- loop over z
    zs: List[int] = list(range(agg)) + [agg]
    if n_jobs == 1:
        rows = [_agg_for_z(work, z, agg,
                           group_vars=group_vars,
                           panel=panel,
                           id_col=id_col)
                for z in zs]
    else:
        rows = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_agg_for_z)(work, z, agg,
                                group_vars=group_vars,
                                panel=panel,
                                id_col=id_col)
            for z in zs
        )

    return pd.DataFrame(rows)


def zoom_in_analysis(
    df: pd.DataFrame,
    analysis_vars: Sequence[str],
    group_vars_list: Sequence[Sequence[str]],
    female_col: str,
    male_col: str,
    id_col: str,
    external_ps_col: Optional[str] = None,
    calculate_ps_before: bool = True,
    candidate_pool: Optional[pd.DataFrame] = None,
    agg: int = 6,
    panel: bool = False,
    min_group_size: int = 10,
    n_jobs: int = 1
) -> Dict[Tuple, pd.DataFrame]:
    """
    High-level wrapper that “zooms in” by looping over values of one or more
    `analysis_vars` (e.g. “country”, “mainsec”, etc.) and grouping definitions.

    NOTE
    ----
    We kept `subgroup_defs` in the signature purely for backward compatibility, but
    it is not used inside this function. If you need to filter on some “subgroup”
    string, add that logic before calling this wrapper.

    Parameters
    ----------
    df : DataFrame
      Must contain at least:
        - female_col, male_col, id_col,
        - any column in analysis_vars,
        - any column used inside group_vars_list.

    analysis_vars : list of str
      Column names that you want to “zoom in on.” For each unique value of
      each analysis_var, we will call `build_p_s(...)` + `perform_analysis(...)`.

    subgroup_defs : list of str
      (Deprecated / unused) kept so existing notebooks don’t break.

    group_vars_list : list of lists of str
      Each inner list is a grouping scheme (e.g. ["country","year"], or
      ["country","year","mainsec"], etc.). We will call `build_p_s` and
      `perform_analysis` once per grouping scheme per value of `analysis_var`.

    female_col, male_col, id_col : str
      Column names for “female count,” “male count,” and “firm ID,” respectively.

    external_ps_col : str or None
      If not None, the name of a column in `df` that already holds percentages
      (0–100) for p_s. We compute `p_s = mean(external_ps_col)/100`. If None,
      we compute p_s from sums of female_col/(female_col+male_col).

    calculate_ps_before : bool
      If True (and external_ps_col is None), p_s is computed from `candidate_pool`
      if provided, else from each `df_var` subset. If False, fallback to computing
      p_s from `df_var` itself.

    candidate_pool : DataFrame or None
      When `calculate_ps_before=True` and `external_ps_col=None`, we use this DataFrame
      (rather than the per‐subset df_var) to compute p_s.

    agg, panel, min_group_size, n_jobs : same as in `perform_analysis`.

    Returns
    -------
    A dict whose keys are:
      (analysis_var, value, subgroup_def, tuple(group_vars), calculate_ps_before, external_ps_col)
    and whose values are the DataFrame returned by `perform_analysis` for that slice.

    Example
    -------
    >>> from implicitquotas.analysis import zoom_in_analysis
    >>> results = zoom_in_analysis(
    ...     df             = my_df,
    ...     analysis_vars  = ["country","mainsec"],
    ...     subgroup_defs  = ["all"],                # ignored internally
    ...     group_vars_list= [["country","year"]],
    ...     female_col     = "boards_fem",
    ...     male_col       = "boards_male",
    ...     id_col         = "id_bvd",
    ...     external_ps_col= None,
    ...     calculate_ps_before=True,
    ...     candidate_pool = None,
    ...     agg            = 6,
    ...     panel          = True,
    ...     min_group_size = 10,
    ...     n_jobs         = 2
    ... )
    """
    all_results: Dict[Tuple, pd.DataFrame] = {}

    for var in analysis_vars:
        if var not in df.columns:
            raise ValueError(f"analysis_var {var!r} is not in DataFrame.")

        unique_vals = df[var].dropna().unique()
        for val in unique_vals:
            df_var = df[df[var] == val].copy()
            for group_vars in group_vars_list:
                # 1) build p_s for this subset + grouping
                p_s = _build_p_s(
                    df                = df_var,
                    group_vars        = group_vars,
                    female_col        = female_col,
                    male_col          = male_col,
                    external_ps_col   = external_ps_col,
                    calculate_ps_before=calculate_ps_before,
                    candidate_pool    = candidate_pool
                )

                # 2) perform the main analysis
                result_df = perform_analysis(
                    df             = df_var,
                    p_s            = p_s,
                    group_vars     = group_vars,
                    female_col     = female_col,
                    male_col       = male_col,
                    id_col         = id_col,
                    agg            = agg,
                    panel          = panel,
                    min_group_size = min_group_size,
                    n_jobs         = n_jobs
                )

                key = (
                    var,
                    val,
                    tuple(group_vars),
                    calculate_ps_before,
                    external_ps_col
                )
                all_results[key] = result_df

    return all_results
