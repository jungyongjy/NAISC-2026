"""
drift/mitigator.py
==================
Stage 4 — Drift Mitigation.

Applies targeted transformations to drifted columns identified in Stage 3.
All encoding and scaling parameters are computed from train_df ONLY and
then applied to both train_df and test_df.

Public API
----------
    mitigate(train_df, test_df, ranked_df) -> (mitigated_train, mitigated_test)

Treatment routing (keyed on ranked_df["col_type"]):

    high_cardinality / HIGH_CARD_CAT
        → Train-Anchored Frequency Encoding: replace each city/category string
          with the count of how many train rows share that value.  Gives LightGBM
          a population-density proxy without target leakage.  Test rows whose
          category never appeared in train get frequency 0.  Original string
          column is replaced by a new Int64 column of the same name.

    categorical / LOW_CARD_CAT
        → Laplace-Smoothed Target Encoding (m=20): compute per-category churn
          rate from train_df, smooth toward the global rate to penalise rare
          categories, then map onto both splits.  Smoothing formula:
              smoothed = (n * rate + m * global_rate) / (n + m)
          Unseen test categories fall back to global_rate.  Result is Float64.

    numerical / CONTINUOUS — SEVERE drift
        → Quantile Binning: compute N_BINS=10 decile breakpoints from train_df,
          apply pl.cut() with those fixed breaks to both splits.  Converts a
          distorted continuous distribution into a clean ordinal rank (0–9).
          LightGBM splits on ordinal bins very efficiently.  Breakpoints are
          train-anchored — test values outside the train range land in bin 0 or 9.

    numerical / CONTINUOUS — MILD or MODERATE drift
        → Robust Scaling: (x - median) / IQR, parameters from train_df only.
          If IQR == 0 the column is left unchanged and a warning is printed.

Constraints
-----------
- 100% Polars-native: no pandas, no .apply(), no row-level Python loops
- Fit-on-train-only strictly enforced for all treatments
- All other columns pass through completely unchanged
- High-cardinality columns are frequency-encoded, NOT dropped
"""

from __future__ import annotations

import polars as pl

# Column type aliases — handle both spec names and actual pipeline output names
_HIGH_CARD_TYPES = {"high_cardinality", "high_card_cat"}
_CAT_TYPES       = {"categorical", "low_card_cat"}
_NUM_TYPES       = {"numerical", "continuous"}

_TARGET_COL = "ChurnStatus"
_LAPLACE_M  = 20   # smoothing strength for target encoding
N_BINS      = 10   # number of equal-frequency bins for quantile binning


# ── Frequency Encoding ────────────────────────────────────────────────────────

def _frequency_encode_column(
    col: str,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
) -> tuple[pl.Series, pl.Series]:
    """
    Train-Anchored Frequency Encoding for high-cardinality string columns.

    Replaces each category value with the number of train rows that share it.
    Provides LightGBM with a population-density proxy (e.g. city size) with
    zero target leakage — only row counts, not churn rates, are used.

    Test categories absent from train receive frequency 0.

    Returns (freq_train_series, freq_test_series) as Int64.
    """
    freq_map = (
        train_df
        .select(pl.col(col).cast(pl.String).str.to_lowercase().str.strip_chars().str.replace_all(r'[-_]', ' ').str.replace_all(r'\s+', ' ').alias("__cat__"))
        .group_by("__cat__")
        .agg(pl.len().alias("__freq__"))
    )

    train_encoded = (
        train_df
        .select(pl.col(col).cast(pl.String).str.to_lowercase().str.strip_chars().str.replace_all(r'[-_]', ' ').str.replace_all(r'\s+', ' ').alias("__cat__"))
        .join(freq_map, on="__cat__", how="left")
        .select(
            pl.col("__freq__").fill_null(0).cast(pl.Int64).alias(col)
        )
        [col]
    )

    test_encoded = (
        test_df
        .select(pl.col(col).cast(pl.String).str.to_lowercase().str.strip_chars().str.replace_all(r'[-_]', ' ').str.replace_all(r'\s+', ' ').alias("__cat__"))
        .join(freq_map, on="__cat__", how="left")
        .select(
            pl.col("__freq__").fill_null(0).cast(pl.Int64).alias(col)
        )
        [col]
    )

    return train_encoded, test_encoded


# ── Laplace-Smoothed Target Encoding ─────────────────────────────────────────

def _target_encode_column(
    col: str,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    global_rate: float,
    positive_label: str = "Yes",
) -> tuple[pl.Series, pl.Series]:
    """
    Laplace-Smoothed Target Encoding (m=_LAPLACE_M=20).

    Smoothing formula applied as a single Polars expression (no Python loops):
        smoothed_rate = (n * category_rate + m * global_rate) / (n + m)

    Where n = train row count for the category, m = _LAPLACE_M.

    Rare categories are pulled toward global_rate; high-volume categories
    retain most of their empirical churn rate.  Unseen test categories
    fall back to global_rate.

    Returns (encoded_train_series, encoded_test_series) as Float64.
    """
    rate_map = (
        train_df
        .select([
            pl.col(col).cast(pl.String).str.to_lowercase().str.replace_all(r'[-_\s]+', ' ').str.strip_chars().alias("__cat__"),
            (pl.col(_TARGET_COL) == positive_label).cast(pl.Float64).alias("__target__"),
        ])
        .group_by("__cat__")
        .agg([
            pl.len().alias("__n__"),
            pl.col("__target__").mean().alias("__raw_rate__"),
        ])
        .with_columns(
            (
                (pl.col("__n__") * pl.col("__raw_rate__"))
                + (_LAPLACE_M * global_rate)
            )
            .truediv(pl.col("__n__") + _LAPLACE_M)
            .alias("__smoothed_rate__")
        )
        .select(["__cat__", "__smoothed_rate__"])
    )

    train_encoded = (
        train_df
        .select(pl.col(col).cast(pl.String).str.to_lowercase().str.replace_all(r'[-_\s]+', ' ').str.strip_chars().alias("__cat__"))
        .join(rate_map, on="__cat__", how="left")
        .select(
            pl.col("__smoothed_rate__")
              .fill_null(global_rate)
              .cast(pl.Float64)
              .alias(col)
        )
        [col]
    )

    test_encoded = (
        test_df
        .select(pl.col(col).cast(pl.String).str.to_lowercase().str.replace_all(r'[-_\s]+', ' ').str.strip_chars().alias("__cat__"))
        .join(rate_map, on="__cat__", how="left")
        .select(
            pl.col("__smoothed_rate__")
              .fill_null(global_rate)
              .cast(pl.Float64)
              .alias(col)
        )
        [col]
    )

    return train_encoded, test_encoded


# ── Quantile Binning ──────────────────────────────────────────────────────────

def _quantile_bin_column(
    col: str,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
) -> tuple[pl.Series, pl.Series] | None:
    """
    Quantile Binning for SEVERE numerical drift (N_BINS=10 equal-frequency bins).

    For severely drifted continuous features, scaling is insufficient because the
    entire distribution shape has changed.  Binning converts the raw value into an
    ordinal rank (0 to N_BINS-1), which is invariant to monotonic distributional
    shifts — LightGBM only needs to know which decile a value falls in, not the
    exact value.

    Steps (all train-anchored, Polars-native):
      1. Compute N_BINS-1 decile breakpoints from train_df (10th, 20th … 90th
         percentile).  Deduplicate breaks in case of ties (e.g. many zeros).
      2. Apply pl.cut() with those fixed breakpoints to both train and test.
         pl.cut() assigns each value to a labelled bin; labels are integer
         strings "0" through "N_BINS-1".
      3. Cast labels to Int8 for memory efficiency and LightGBM compatibility.

    Train-anchored guarantee: test values below the train minimum land in bin 0;
    values above the train maximum land in bin N_BINS-1.  This correctly signals
    to LightGBM that those test values are extreme relative to training norms.

    Returns None if fewer than 2 unique breakpoints exist (e.g. a near-constant
    column — should not occur for SEVERE drift but handled defensively).
    Returns (binned_train_series, binned_test_series) as Int8 otherwise.
    """
    quantile_levels = [i / N_BINS for i in range(1, N_BINS)]  # 0.1, 0.2 … 0.9

    # Compute breakpoints from train in a single Polars aggregation
    break_exprs = [
        pl.col(col).quantile(q, interpolation="linear").alias(f"q{i}")
        for i, q in enumerate(quantile_levels)
    ]
    raw_breaks = (
        train_df
        .select(break_exprs)
        .row(0, named=True)
    )

    # Deduplicate and sort — ties can occur with sparse/zero-heavy columns
    breaks = sorted(set(
        float(v) for v in raw_breaks.values() if v is not None
    ))

    if len(breaks) < 2:
        return None  # degenerate column — caller falls back to robust scaling

    # Integer bin labels: "0", "1", … "N_BINS-1"
    # len(labels) must equal len(breaks) + 1
    labels = [str(i) for i in range(len(breaks) + 1)]

    # In Polars 1.x, cut() is a Series method — NOT a module-level function.
    # Correct: series.cut(breaks=..., labels=...)
    # Wrong:   pl.cut(pl.col(...), ...)   ← AttributeError in Polars 1.x
    def _apply_cut(df: pl.DataFrame) -> pl.Series:
        series = df[col].cast(pl.Float64)
        return (
            series
            .cut(breaks=breaks, labels=labels)  # returns Categorical; nulls for out-of-range
            .cast(pl.String)                    # Categorical → String ("0".."N"), null → null
            .fill_null("0")                     # values outside break range → bin 0 (lowest)
            .cast(pl.Int8)                      # String → Int8 ordinal
            .alias(col)
        )

    return _apply_cut(train_df), _apply_cut(test_df)


# ── Robust Scaling ────────────────────────────────────────────────────────────

def _robust_scale_column(
    col: str,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
) -> tuple[pl.Series, pl.Series] | None:
    """
    Robust Scaling: (x - median) / IQR using parameters from train_df only.

    Returns None if IQR == 0 (caller skips and warns).
    Returns (scaled_train_series, scaled_test_series) as Float64 otherwise.
    """
    stats = (
        train_df
        .select([
            pl.col(col).median().alias("median"),
            (
                pl.col(col).quantile(0.75, interpolation="linear")
                - pl.col(col).quantile(0.25, interpolation="linear")
            ).alias("iqr"),
        ])
        .row(0, named=True)
    )

    median = float(stats["median"] or 0.0)
    iqr    = float(stats["iqr"]    or 0.0)

    if abs(iqr) < 1e-9:
        return None

    scale_expr   = (pl.col(col).cast(pl.Float64) - median) / iqr
    train_scaled = train_df.select(scale_expr.alias(col))[col]
    test_scaled  = test_df.select(scale_expr.alias(col))[col]

    return train_scaled, test_scaled


# ── Public API ────────────────────────────────────────────────────────────────

def mitigate(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    ranked_df: pl.DataFrame,
    positive_label: str = "Yes",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Apply drift mitigation to all drifted columns in ranked_df.

    Routing priority:
        1. high_cardinality  → Frequency Encoding          (intercepts DROP)
        2. categorical        → Laplace-Smoothed Target Encoding (m=20)
        3. numerical SEVERE   → Quantile Binning             (10 decile bins)
        4. numerical MILD/MOD → Robust Scaling               ((x-median)/IQR)
        5. anything else      → warn and skip

    Parameters
    ----------
    train_df  : Full training Polars DataFrame (must include ChurnStatus).
    test_df   : Full test/serve Polars DataFrame.
    ranked_df : Stage 3 output with columns "feature" and "col_type".

    Returns
    -------
    (mitigated_train_df, mitigated_test_df) preserving original column order.
    Only drifted columns are transformed; all others pass through unchanged.
    """
    # Pull the feature → (col_type, drift_severity) mapping
    # Iterating over a small metadata dict (at most a few hundred drifted features)
    feature_meta: dict[str, tuple[str, str]] = {
        row["feature"]: (row["col_type"].lower(), (row.get("drift_severity") or "MILD"))
        for row in ranked_df.select(
            ["feature", "col_type", "drift_severity"]
            if "drift_severity" in ranked_df.columns
            else ["feature", "col_type"]
        ).iter_rows(named=True)
    }

    global_rate = float(
        train_df
        .select((pl.col(_TARGET_COL) == positive_label).cast(pl.Float64).mean())
        .item()
    )

    train_cols: dict[str, pl.Series] = {c: train_df[c] for c in train_df.columns}
    test_cols:  dict[str, pl.Series] = {c: test_df[c]  for c in test_df.columns}

    for feature, (col_type, severity) in feature_meta.items():

        if feature not in train_df.columns or feature not in test_df.columns:
            print(f"  ⚠ mitigate: '{feature}' not found in DataFrames — skipped.")
            continue

        if col_type in _HIGH_CARD_TYPES:
            # ── Frequency Encoding (intercepts DROP) ──────────────────────
            tr_enc, te_enc = _frequency_encode_column(feature, train_df, test_df)
            train_cols[feature] = tr_enc
            test_cols[feature]  = te_enc

        elif col_type in _CAT_TYPES:
            # ── Laplace-Smoothed Target Encoding ──────────────────────────
            tr_enc, te_enc = _target_encode_column(
                feature, train_df, test_df, global_rate, positive_label
            )
            train_cols[feature] = tr_enc
            test_cols[feature]  = te_enc
            print(f"  ✔ target-encoded  : {feature}  "
                  f"(global_rate={global_rate:.4f}, m={_LAPLACE_M})")

        elif col_type in _NUM_TYPES:
            if severity == "SEVERE":
                # ── Quantile Binning — SEVERE numerical drift ──────────────
                result = _quantile_bin_column(feature, train_df, test_df)
                if result is None:
                    # Degenerate breaks — fall back to robust scaling
                    print(f"  ⚠ quantile-bin degenerate breaks for '{feature}' "
                          f"— falling back to robust scaling")
                    result = _robust_scale_column(feature, train_df, test_df)
                    if result is None:
                        print(f"  ⚠ robust-scale also skipped (IQR=0): {feature}")
                        continue
                    tr_sc, te_sc = result
                    train_cols[feature] = tr_sc
                    test_cols[feature]  = te_sc
                    print(f"  ✔ robust-scaled   : {feature}  (fallback)")
                else:
                    tr_b, te_b = result
                    train_cols[feature] = tr_b
                    test_cols[feature]  = te_b
                    print(f"  ✔ quantile-binned : {feature}  "
                          f"({N_BINS} bins, SEVERE drift → ordinal rank 0–{N_BINS-1})")
            else:
                # ── Robust Scaling — MILD / MODERATE numerical drift ───────
                result = _robust_scale_column(feature, train_df, test_df)
                if result is None:
                    print(f"  ⚠ robust-scale skipped (IQR=0): {feature}")
                    continue
                tr_sc, te_sc = result
                train_cols[feature] = tr_sc
                test_cols[feature]  = te_sc
                print(f"  ✔ robust-scaled   : {feature}  ({severity} drift)")

        else:
            print(f"  ⚠ mitigate: unrecognised col_type '{col_type}' "
                  f"for '{feature}' — skipped.")

    mitigated_train = pl.DataFrame({c: train_cols[c] for c in train_df.columns})
    mitigated_test  = pl.DataFrame({c: test_cols[c]  for c in test_df.columns})

    del train_cols, test_cols

    return mitigated_train, mitigated_test

