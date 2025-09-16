from __future__ import annotations
from typing import (
    List,
    Sequence,
    Union,
    Dict
)
import pandas as pd


def _validate_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    """Raise ValueError if any *cols* missing from *df*."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

def _as_list(x: Union[str, Sequence[str]]) -> List[str]:
    """Ensure *x* is returned as a list of str."""
    return [x] if isinstance(x, str) else list(x)


def _prepare_dataframe(
        df: pd.DataFrame,
        female_col: str,
        male_col: str,
        id_col: str,
        extra_group_cols: Sequence[str]
) -> pd.DataFrame:
    """
    Make a shallow copy of df containing only the columns needed:
      - all grouping columns (extra_group_cols)
      - female_col, male_col, id_col

    Convert each grouping column + id_col to categorical dtype (to save memory
    and speed up groupby).
    """
    needed = list(extra_group_cols) + [female_col, male_col, id_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing these required columns: {missing!r}")

    dd = df[needed].copy()
    for c in extra_group_cols + [id_col]:
        dd[c] = dd[c].astype("category")
    return dd

def extract_ps_simple(
        df: pd.DataFrame,
        group_vars: Sequence[str] | str,
        *,
        female_col: str | None = None,
        male_col: str | None = None,
        external_ps: str | None = None,
        force_percent: bool | None = None
) -> Dict[str, pd.DataFrame]:
    """
    Return two DataFrames (possibly empty) with a single column ``ps``:
      {'calculated': … , 'external': …}

    Parameters
    ----------
    df : DataFrame
    group_vars : column or list of columns that define a group
    female_col , male_col :
        Column names with *counts*.  **Required** if `external_ps` is *not* given.
    external_ps :
        Column with percentages.  If present, *ignores* female/male columns.
    force_percent :
        • None (default) → auto-detect: if max(value)>1 assume 0–100 and divide by 100
        • True  → always divide by 100
        • False → never divide (assume already 0–1)

    Raises
    ------
    KeyError   if requested columns are missing.
    """

    # ------------------------------------------------ normalise inputs
    if isinstance(group_vars, str):
        group_vars = [group_vars]

    # ------------------------------------------------ path A: external column
    if external_ps is not None:
        missing = [c for c in group_vars + [external_ps] if c not in df.columns]
        if missing:
            raise KeyError(f"Missing column(s): {missing!r}")

        out = (
            df.groupby(group_vars, observed=True)[external_ps]
            .mean(min_count=1)
            .rename("ps")
        )

        # 0–100 → 0–1 conversion
        if force_percent is None:
            if out["ps"].max(skipna=True) > 1.001:
                out["ps"] = out["ps"] / 100.0
        elif force_percent:
            out["ps"] = out["ps"] / 100.0

        return {"external": out, "calculated": pd.DataFrame()}

    # ------------------------------------------------ path B: derive from counts
    if female_col is None or male_col is None:
        raise KeyError(
            "When `external_ps` is not provided you *must* pass both "
            "`female_col=` and `male_col=`."
        )

    missing = [c for c in group_vars + [female_col, male_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s): {missing!r}")

    sums = (
        df.groupby(group_vars, observed=True)
        .agg(f_total=(female_col, "sum"),
             m_total=(male_col, "sum"))
        .reset_index()
    )
    denom = sums["f_total"] + sums["m_total"]
    sums["ps"] = sums["f_total"].where(denom > 0, pd.NA) / denom.replace({0: pd.NA})

    return {"calculated": sums[group_vars + ["ps"]], "external": pd.DataFrame()}


def _build_p_s(
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


def adjust_p_values(dataframes, method='B'):
    # Assuming each dataframe represents a separate hypothesis group, calculate the number of tests

    if method == 'HB':
        # Calculate the total number of tests across all dataframes
        total_tests = sum(len(df) for df in dataframes.values())

        # Initialize a list to store all p-values and their indices
        p_values = []
        for key, df in dataframes.items():
            for index, row in df.iterrows():
                p_values.append((row['p_value'], key, index))

        # Sort p-values in ascending order
        p_values_sorted = sorted(p_values, key=lambda x: x[0])

        # Apply Holm-Bonferroni adjustment
        adjusted_p_values = {}
        for i, (p_val, key, index) in enumerate(p_values_sorted):
            if i == 0:
                adjusted_p_values[(key, index)] = p_val * (total_tests - i)
            else:
                adjusted_p_values[(key, index)] = max(p_val * (total_tests - i), adjusted_p_values[prev_key_index])

            prev_key_index = (key, index)  # Store the current key and index for the next iteration

        # Update the dataframes with adjusted p-values
        for key, df in dataframes.items():
            adjusted_p_col = []
            for index, row in df.iterrows():
                adjusted_p_col.append(min(adjusted_p_values[(key, index)], 1))  # Ensure p-value is not greater than 1
            df['adjusted_p_value'] = adjusted_p_col

        return dataframes

    if method == "B":
        total_tests = sum(len(df) for df in dataframes)

        # Create a new dictionary to store adjusted dataframes
        adjusted_dataframes = {}

        for key, df in dataframes.items():
            # Create a copy of the dataframe to avoid modifying the original data
            adjusted_df = df.copy()

            # Apply Bonferroni correction
            adjusted_df['adjusted_p_value'] = df['p_value'] * total_tests

            # Ensure that the adjusted p-values do not exceed 1
            adjusted_df['adjusted_p_value'] = adjusted_df['adjusted_p_value'].clip(upper=1)

            # Store in the new dictionary
            adjusted_dataframes[key] = adjusted_df

        return adjusted_dataframes