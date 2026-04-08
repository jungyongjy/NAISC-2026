"""
drift/ranker.py
===============
Stage 3 — Recency-Weighted Drift Severity Ranking.

Receives per-column drift results from Stage 2 (as a plain Polars DataFrame
built from DriftSummary.drifted) and a serve DataFrame containing a Month
column.  Computes a recency-weighted severity score for each drifted feature
and returns a ranked output DataFrame.

Public API
----------
    rank_drift(drift_results, serve_df, lambda_decay=0.1) -> pl.DataFrame

Do NOT import from detector.py, report.py, or classifier.py.
All inputs arrive as plain Polars DataFrames or Python scalars.
"""

from __future__ import annotations

import polars as pl

# ── Constants ────────────────────────────────────────────────────────────────

# Month strings in chronological order.  Used to assign an integer rank
# (0 = oldest) so exponential decay can be applied as a vectorised multiply.
# Format matches the public dataset: "YY-Mon" e.g. "25-Jan".
# The hidden dataset uses the same primary time column ("Month") but may have
# different month strings; _build_month_rank_map() handles any order found in
# the data by sorting lexicographically after converting to a sortable key.
_MONTH_ABBR_ORDER = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

# Severity → base multiplier applied before decay scaling.
# Ensures SEVERE drifted features always outscore MILD ones even if MILD
# features appear in more-recent rows.
_SEVERITY_WEIGHT = {
    "SEVERE":   3.0,
    "MODERATE": 2.0,
    "MILD":     1.0,
    "NONE":     0.0,
}


# ── Private helpers ───────────────────────────────────────────────────────────

def _build_month_rank_map(serve_df: pl.DataFrame, month_col: str) -> pl.DataFrame:
    """
    Extract unique month strings from serve_df, sort them chronologically,
    and return a small lookup DataFrame:  month_str → integer rank (0=oldest).

    Sorting strategy:
      1. Parse the trailing 3-char month abbreviation (e.g. "Jan" from "25-Jan").
      2. Parse the leading year digits.
      3. Sort by (year, month_num) — fully vectorised, no Python loops.

    Returns a Polars DataFrame with columns [month_col, "month_rank"].
    """
    unique_months = (
        serve_df
        .select(pl.col(month_col).cast(pl.String).unique())
        .rename({month_col: "month_str"})
        # Extract year prefix (everything before the dash)
        .with_columns(
            pl.col("month_str").str.split("-").list.first()
              .cast(pl.Int32, strict=False).alias("year_part"),
            pl.col("month_str").str.split("-").list.last()
              .alias("mon_abbr"),
        )
        # Map month abbreviation → integer 1–12 via a replace expression
        .with_columns(
            pl.col("mon_abbr")
              .replace(
                  old=list(_MONTH_ABBR_ORDER.keys()),
                  new=[str(v) for v in _MONTH_ABBR_ORDER.values()],
              )
              .cast(pl.Int32, strict=False)
              .alias("mon_num"),
        )
    )
    # Fallback: if all mon_num values are null (unrecognised format),
    # sort by month_str lexicographically instead.
    if unique_months["mon_num"].is_null().all():
        unique_months = unique_months.sort("month_str")
    else:
        unique_months = unique_months.sort(["year_part", "mon_num"])
    unique_months = (
        unique_months
        .with_row_index("month_rank")          # 0-based integer rank
        .select(["month_str", "month_rank"])
        .rename({"month_str": month_col})
    )
    return unique_months


def _recency_weights(
    serve_df: pl.DataFrame,
    month_col: str,
    lambda_decay: float,
) -> pl.DataFrame:
    """
    Attach a per-row exponential decay weight to serve_df.

    Formula:  w(t) = exp(λ × t)
      where t = month_rank (integer, 0 for the oldest month in serve_df).

    This is computed entirely in Polars expressions — no Python loops.

    Returns serve_df with two extra columns:
      • "month_rank"     — integer chronological rank of that row's month
      • "decay_weight"   — exp(λ × month_rank), a float64 scalar per month
    """
    rank_map = _build_month_rank_map(serve_df, month_col)

    return (
        serve_df
        .join(rank_map, on=month_col, how="left")
        .with_columns(
            (pl.col("month_rank").cast(pl.Float64) * lambda_decay)
            .exp()
            .alias("decay_weight")
        )
    )


# C7: Month-level recency aggregation (no per-row join)
def _recency_weights_month_level(
    serve_df: pl.DataFrame,
    month_col: str,
    lambda_decay: float,
) -> pl.DataFrame:
    """
    Compute per-month decay weights without a per-row join.

    Steps:
      1. Count rows per month (group_by on one column — fast).
      2. Join rank_map (≤24 rows) to the per-month counts.
      3. Compute decay_weight at month level.

    Returns a small DataFrame [month_col, n_rows, month_rank, decay_weight, w_fraction].
    """
    month_counts = (
        serve_df
        .select(month_col)
        .group_by(month_col)
        .agg(pl.len().alias("n_rows"))
    )
    rank_map = _build_month_rank_map(serve_df.select(month_col), month_col)

    per_month = (
        month_counts
        .join(rank_map, on=month_col, how="left")
        .with_columns(
            (pl.col("month_rank").cast(pl.Float64) * lambda_decay)
            .exp()
            .alias("decay_weight")
        )
        .with_columns(
            (
                pl.col("decay_weight") * pl.col("n_rows")
            ).alias("weighted_count")
        )
    )
    total_weighted = float(per_month["weighted_count"].sum() or 1.0)
    per_month = per_month.with_columns(
        (pl.col("weighted_count") / total_weighted).alias("w_fraction")
    )
    return per_month


def _weighted_row_fraction(
    weighted_df: pl.DataFrame,
    month_col: str,
) -> pl.DataFrame:
    """
    Compute the decay-weighted proportion of rows per month.

    Returns a DataFrame [month_col, "w_fraction"] where w_fraction is the
    fraction of total decay-weighted mass that each month contributes.

    This is the recency-adjusted "importance" of each month's data in the
    serve set — used to scale raw drift scores by how much recent months
    dominate the serve distribution.
    """
    per_month = (
        weighted_df
        .group_by(month_col)
        .agg(
            pl.col("decay_weight").sum().alias("w_sum"),
            pl.len().alias("n_rows"),
        )
        .with_columns(
            (pl.col("w_sum") / pl.col("w_sum").sum()).alias("w_fraction")
        )
    )
    return per_month


# ── Public API ────────────────────────────────────────────────────────────────

def rank_drift(
    drift_results: pl.DataFrame,
    serve_df: pl.DataFrame,
    lambda_decay: float = 0.1,
    month_col: str = "Month",
) -> pl.DataFrame:
    """
    Compute recency-weighted drift severity scores and rank drifted features.

    Parameters
    ----------
    drift_results : pl.DataFrame
        Must contain columns:
          "feature"     — column name (str)
          "col_type"    — feature type string (str)
          "test_used"   — name of statistical test (str)
          "raw_score"   — raw divergence / test statistic (float)
          "drift_severity" — "SEVERE" | "MODERATE" | "MILD" | "NONE" (str)
        Typically built from DriftSummary.drifted in the notebook cell.

    serve_df : pl.DataFrame
        The test / serve dataset.  Must contain `month_col`.
        Used only to compute the recency weight distribution — no feature
        values are read, so this is fast regardless of column count.

    lambda_decay : float, default 0.1
        Exponential decay rate.  Larger λ → more weight on recent months.
        Decay formula:  w(t) = exp(λ × t)
          where t = chronological month rank (0 = oldest month in serve_df).

    month_col : str, default "Month"
        Name of the time column in serve_df.

    Returns
    -------
    pl.DataFrame with columns:
        "feature"        — column name
        "col_type"       — feature type
        "test_used"      — statistical test used
        "raw_score"      — raw divergence / statistic from Stage 2
        "weighted_score" — raw_score × severity_weight × recency_factor
        "drift_rank"     — integer rank (1 = highest weighted_score)

    Weighted score formula
    ----------------------
        recency_factor  = Σ_t [ w(t) × n(t) ] / Σ_t [ n(t) ]
                          (decay-weighted mean row fraction in serve set)
        severity_weight = {SEVERE: 3.0, MODERATE: 2.0, MILD: 1.0, NONE: 0.0}
        weighted_score  = raw_score × severity_weight × recency_factor

    The recency_factor is a scalar in (0, 1] representing how concentrated
    the serve data is in recent months.  When all serve rows are in the most
    recent month, recency_factor → 1.  When rows are spread evenly across
    many old months, recency_factor is smaller.  This scales drift scores up
    when the drifted data is heavily recent (higher urgency) and down when
    most serve rows are from older, more stable periods.

    Notes
    -----
    • Entirely Polars-native: no row-level Python loops over DataFrames.
    • Month ranking and decay weighting are vectorised via join + expression.
    • Only the Month column of serve_df is read; all other columns are ignored,
      so runtime is O(n_rows) for a single group_by, not O(n_rows × n_cols).
    """
    # C7: Use month-level aggregation - no per-row join to 10M rows
    per_month = _recency_weights_month_level(serve_df, month_col, lambda_decay)
    recency_factor = float(
        per_month.select((pl.col("w_fraction") ** 2).sum()).item()
    )

    # ── Step 2: map severity labels to numeric weights ────────────────────
    sev_keys = list(_SEVERITY_WEIGHT.keys())
    sev_vals = [str(v) for v in _SEVERITY_WEIGHT.values()]

    ranked = (
        drift_results
        .with_columns(
            pl.col("drift_severity")
              .replace(old=sev_keys, new=sev_vals)
              .cast(pl.Float64)
              .alias("severity_weight")
        )
        # ── Step 3: compute weighted_score ────────────────────────────────
        .with_columns(
            (
                pl.col("raw_score")
                * pl.col("severity_weight")
                * pl.lit(recency_factor)
            ).alias("weighted_score")
        )
        # ── Step 4: rank descending by weighted_score ─────────────────────
        .sort("weighted_score", descending=True)
        .with_row_index("drift_rank", offset=1)
        # ── Step 5: select and cast output schema ─────────────────────────
        .select([
            pl.col("feature").cast(pl.String),
            pl.col("col_type").cast(pl.String),
            pl.col("test_used").cast(pl.String),
            pl.col("raw_score").cast(pl.Float64),
            pl.col("weighted_score").cast(pl.Float64).round(6),
            pl.col("drift_rank").cast(pl.UInt32),
            pl.col("drift_severity").cast(pl.String),
        ])
    )

    print(f"Stage 3 complete: {len(ranked)} drifted features ranked.")
    return ranked


