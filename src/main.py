"""
src/main.py
===========
NAISC 2026 — SingTel Churn Prediction Pipeline
Production CLI entrypoint.  Orchestrates Stages 1-5.

Usage:
    python ./src/main.py -train_data_filepath <path> -test_data_filepath <path>

Stages:
    1  Ingestion          — load CSVs with Polars
    2  Classification     — schema typing + drift detection
    3  Ranking            — recency-weighted drift severity
    4  Mitigation         — target encoding + robust scaling
    5  Model + Output     — LightGBM → prediction.csv

Architecture constraints:
    - Polars-native throughout until the explicit pandas handoff in Stage 5
    - No row-level Python loops over DataFrames
    - No pandas before Step 5c
    - String normalisation happens in Polars (Step 5a), before pandas handoff
    - prediction.csv written to project root (parent of src/)
    - detector.py, ranker.py, mitigator.py, classifier.py are NEVER modified

Namespace note:
    detector.py was originally written for a shared Jupyter namespace.  It uses
    FeatureManifest, FeatureType, DriftSeverity, etc. without importing them.
    main.py resolves this by injecting the required symbols into drift.detector's
    module namespace immediately after import, before any DriftDetector call.
    This is the only clean solution that does not require modifying detector.py.
"""

from __future__ import annotations

import io
import argparse
import sys
import time
import subprocess
from pathlib import Path

# Reconfigure stdout/stderr to UTF-8 so emoji in print statements
# don't crash on Windows terminals using CP1252 encoding.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Path setup ────────────────────────────────────────────────────────────────
# Ensure project root (parent of src/) is on sys.path so that
# `drift/` and `schema/` are importable as top-level packages.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Third-party imports ───────────────────────────────────────────────────────
import polars as pl
import lightgbm as lgb
from sklearn.metrics import average_precision_score

# ── Internal imports — order matters for namespace injection ──────────────────
# 1. Import classifier symbols first (FeatureManifest, FeatureType, etc.)
from schema.classifier import (
    classify_features,
    FeatureManifest,
    FeatureType,
    ColumnProfile,
    _is_numeric,
    _is_float,
    _is_integer,
    _is_string,
    _is_boolean,
)

# 2. Import report symbols (DriftSeverity, DriftSummary, etc.)
from drift.report import (
    DriftSeverity,
    DriftSummary,
    ColumnDriftResult,
    MitigationStrategy,
    TestMethod,
)

# 3. Import detector MODULE (not just DriftDetector class) so we can inject
#    the above symbols into its namespace before any DriftDetector is used.
import drift.detector as _detector_module

# Inject classifier symbols into detector's namespace
_detector_module.FeatureManifest   = FeatureManifest
_detector_module.FeatureType       = FeatureType
_detector_module.ColumnProfile     = ColumnProfile
_detector_module._is_numeric       = _is_numeric
_detector_module._is_float         = _is_float
_detector_module._is_integer       = _is_integer
_detector_module._is_string        = _is_string
_detector_module._is_boolean       = _is_boolean

# Inject report symbols into detector's namespace
_detector_module.DriftSeverity     = DriftSeverity
_detector_module.DriftSummary      = DriftSummary
_detector_module.ColumnDriftResult = ColumnDriftResult
_detector_module.MitigationStrategy = MitigationStrategy
_detector_module.TestMethod        = TestMethod

# Now safe to reference DriftDetector
DriftDetector = _detector_module.DriftDetector

# 4. Import Stage 3 and 4 modules (clean — no implicit namespace deps)
from drift.ranker    import rank_drift
from drift.mitigator import mitigate


# ── Constants ─────────────────────────────────────────────────────────────────
_TARGET     = "ChurnStatus"   # actual column name in the dataset
_ID_COL     = "CustomerID"


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    """
    Parse exactly two required single-dash flags.
    Validates file existence before returning.
    """
    parser = argparse.ArgumentParser(
        prog="python ./src/main.py",
        description="NAISC 2026 — SingTel Churn Prediction Pipeline",
        add_help=True,
    )
    parser.add_argument(
        "--train_data_filepath",
        required=True,
        metavar="PATH",
        help="Path to training CSV file",
    )
    parser.add_argument(
        "--test_data_filepath",
        required=True,
        metavar="PATH",
        help="Path to test/serve CSV file",
    )
    parser.add_argument(
        "--skip_dashboard",
        action="store_true",
        help="Skip auto-launching Streamlit dashboard after pipeline completes",
    )
    parser.add_argument(
        "--mild_mod_num_policy",
        choices=["robust", "binning", "delta", "none"],
        default="binning",
        help="Policy for MILD/MODERATE numerical drift columns",
    )

    args = parser.parse_args()

    # File existence validation — exit 1 with clear message if missing
    errors = []
    if not Path(args.train_data_filepath).exists():
        errors.append(f"  Train file not found: {args.train_data_filepath}")
    if not Path(args.test_data_filepath).exists():
        errors.append(f"  Test  file not found: {args.test_data_filepath}")

    if errors:
        print("ERROR: one or more input files could not be found:")
        for e in errors:
            print(e)
        print(f"\nUsage: python ./src/main.py "
              f"-train_data_filepath <path> -test_data_filepath <path>")
        sys.exit(1)

    return args


def _launch_dashboard() -> None:
    """Launch Streamlit dashboard.py in a separate process/console."""
    dashboard_path = Path(__file__).resolve().parent / "dashboard.py"
    if not dashboard_path.exists():
        print(f"⚠ Dashboard file not found at: {dashboard_path}")
        return

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_path),
        "--server.headless",
        "true",
    ]
    print(f"\n🚀 Launching dashboard in a new process: {' '.join(cmd)}")
    print("Pipeline run is complete. Dashboard logs will appear in a separate window/process.")
    try:
        if sys.platform.startswith("win"):
            subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            subprocess.Popen(cmd)
    except FileNotFoundError:
        print("⚠ Could not start Streamlit. Please install requirements and retry.")


def _get_mild_mod_numerical_features(ranked_df: pl.DataFrame) -> list[str]:
    """Return MILD/MODERATE numerical drifted feature names from ranked_df."""
    required = {"feature", "col_type", "drift_severity"}
    if not required.issubset(set(ranked_df.columns)):
        return []

    out: list[str] = []
    for row in ranked_df.select(["feature", "col_type", "drift_severity"]).iter_rows(named=True):
        col_type = str(row["col_type"]).lower()
        severity = str(row["drift_severity"]).upper()
        if col_type in {"numerical", "continuous"} and severity in {"MILD", "MODERATE"}:
            out.append(str(row["feature"]))
    return out


def _quantile_bin_from_train(
    feature: str,
    raw_train: pl.DataFrame,
    raw_test: pl.DataFrame,
    n_bins: int = 10,
) -> tuple[pl.Series, pl.Series] | None:
    """Train-anchored quantile binning for one numeric feature."""
    train_series = raw_train[feature].cast(pl.Float64)
    test_series = raw_test[feature].cast(pl.Float64)
    levels = [i / n_bins for i in range(1, n_bins)]
    q_exprs = [pl.col(feature).quantile(q, interpolation="linear").alias(f"q{i}") for i, q in enumerate(levels)]
    raw_breaks = raw_train.select(q_exprs).row(0, named=True)
    breaks = sorted(set(float(v) for v in raw_breaks.values() if v is not None))
    if len(breaks) < 2:
        return None

    labels = [str(i) for i in range(len(breaks) + 1)]
    tr = (
        train_series.cut(breaks=breaks, labels=labels)
        .cast(pl.String)
        .fill_null("0")
        .cast(pl.Int8)
        .alias(feature)
    )
    te = (
        test_series.cut(breaks=breaks, labels=labels)
        .cast(pl.String)
        .fill_null("0")
        .cast(pl.Int8)
        .alias(feature)
    )
    return tr, te


def _apply_mild_mod_numerical_policy(
    policy: str,
    ranked_df: pl.DataFrame,
    raw_train: pl.DataFrame,
    raw_test: pl.DataFrame,
    mitigated_train: pl.DataFrame,
    mitigated_test: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """Override MILD/MODERATE numerical columns according to selected policy."""
    features = [
        f for f in _get_mild_mod_numerical_features(ranked_df)
        if f in raw_train.columns and f in raw_test.columns
    ]

    if policy == "robust" or not features:
        return mitigated_train, mitigated_test, features

    train_cols = {c: mitigated_train[c] for c in mitigated_train.columns}
    test_cols = {c: mitigated_test[c] for c in mitigated_test.columns}

    for feature in features:
        if policy == "none":
            train_cols[feature] = raw_train[feature]
            test_cols[feature] = raw_test[feature]
        elif policy == "delta":
            med = raw_train[feature].cast(pl.Float64).median()
            if med is None:
                continue
            train_cols[feature] = (raw_train[feature].cast(pl.Float64) - float(med)).alias(feature)
            test_cols[feature] = (raw_test[feature].cast(pl.Float64) - float(med)).alias(feature)
        elif policy == "binning":
            result = _quantile_bin_from_train(feature, raw_train, raw_test, n_bins=10)
            if result is None:
                continue
            tr, te = result
            train_cols[feature] = tr
            test_cols[feature] = te

    adjusted_train = pl.DataFrame({c: train_cols[c] for c in mitigated_train.columns})
    adjusted_test = pl.DataFrame({c: test_cols[c] for c in mitigated_test.columns})
    return adjusted_train, adjusted_test, features


def _print_policy_summary(policy: str, features: list[str]) -> None:
    """Print bordered table for selected mild/moderate numerical policy."""
    print("\nMild/Moderate Numerical Drift Policy")
    h1, h2 = "Field", "Value"
    feat_display = ", ".join(features[:8])
    if len(features) > 8:
        feat_display += f" ... (+{len(features) - 8} more)"
    if not feat_display:
        feat_display = "(none)"
    rows = [
        ("Policy", policy),
        ("Affected columns", str(len(features))),
        ("Columns", feat_display),
    ]
    w1 = max(len(h1), max(len(r[0]) for r in rows))
    w2 = max(len(h2), max(len(r[1]) for r in rows))
    sep = f"+{'-'*(w1+2)}+{'-'*(w2+2)}+"
    print(sep)
    print(f"| {h1:<{w1}} | {h2:<{w2}} |")
    print(sep)
    for c1, c2 in rows:
        print(f"| {c1:<{w1}} | {c2:<{w2}} |")
        print(sep)


# ── Pipeline stages ───────────────────────────────────────────────────────────

def stage1_ingest(
    train_path: str,
    test_path: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load CSVs with Polars.

    Candidate 4 — Resilient CSV reading:
      Three-level fallback handles encoding mismatches, BOM, and string
      "NaN" variants that would otherwise be misclassified as non-null.
      infer_schema_length=None scans the full file — safe for hidden datasets
      where early rows may not represent all dtypes.

    Candidate 5 — Column alignment guard:
      If test is missing columns that train has (excluding structural columns),
      fill with null rather than crash downstream. LightGBM handles null
      natively; a missing column is strictly safer than a KeyError.
    """
    _NULL_VALUES = ["", "NA", "N/A", "NULL", "null", "None", "none", "NaN", "nan"]
    _excluded = {_ID_COL, _TARGET, "Month"}

    def _read_csv(path: str) -> pl.DataFrame:
        # Level 1: normal read
        try:
            return pl.read_csv(
                path,
                infer_schema_length=None,
                null_values=_NULL_VALUES,
            )
        except Exception:
            pass
        # Level 2: force UTF-8 schema inference
        try:
            return pl.read_csv(
                path,
                infer_schema_length=None,
                null_values=_NULL_VALUES,
                encoding="utf8-lossy",
            )
        except Exception:
            pass
        # Level 3: ignore errors — last resort
        return pl.read_csv(
            path,
            infer_schema_length=None,
            null_values=_NULL_VALUES,
            ignore_errors=True,
        )

    train_df = _read_csv(train_path)
    test_df  = _read_csv(test_path)

    # Stop condition: CustomerID must be present
    if _ID_COL not in test_df.columns:
        print(f"STOP: '{_ID_COL}' missing from test file — cannot build output.")
        sys.exit(1)

    # Candidate 5 — Column alignment guard:
    # Fill columns present in train but absent in test with null.
    _missing_in_test = [
        c for c in train_df.columns
        if c not in _excluded and c not in test_df.columns
    ]
    if _missing_in_test:
        print(f"  ⚠ Column alignment: {len(_missing_in_test)} train column(s) "
              f"absent in test — filled with null: {_missing_in_test}")
        test_df = test_df.with_columns([
            pl.lit(None).cast(train_df[c].dtype).alias(c)
            for c in _missing_in_test
        ])

    print(f"✅ Stage 1 complete: {len(train_df):,} train rows, "
          f"{len(test_df):,} test rows loaded")
    return train_df, test_df


def stage2_classify_and_detect(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
) -> tuple[object, object]:
    """Schema classification + two-phase drift detection."""
    _t2_start = time.time()
    manifest     = classify_features(train_df, sample_size=50_000, random_state=42)
    detector_obj = DriftDetector(manifest, random_state=42)
    drift_summary = detector_obj.detect_all(train_df, test_df)
    _t2_elapsed = time.time() - _t2_start

    # ── Phase C summary ───────────────────────────────────────────────────────
    all_results = list(drift_summary.results.values())
    n_temporal_flagged = sum(1 for r in all_results if r.phase_c_drift_is_temporal)
    n_upgraded_by_c    = sum(
        1 for r in all_results
        if r.phase_c_notes and "upgraded" in r.phase_c_notes and r.drift_detected
    )
    print(f"  Phase C: {n_temporal_flagged} feature(s) with temporal drift concentration | "
          f"{n_upgraded_by_c} feature(s) upgraded to MILD by sub-population test")

    n_drifted = len(drift_summary.drifted)
    print(f"✅ Stage 2 complete: {n_drifted} columns flagged for drift")

    # ── Drift summary table ───────────────────────────────────────────────────
    _MIT_LABEL = {
        "robust_scaling":                 "Feature Scaling (Robust)",
        "quantile_binning":               "Quantile Binning (10 deciles)",
        "log_transform + robust_scaling": "Log + Feature Scaling",
        "drop_feature":                   "Drop Feature",
        "binarise (sparse->binary)":      "Binarisation",
        "frequency_encoding":             "Frequency Encoding",
        "none":                           "No Action",
        "no_action (stable)":             "No Action (Stable)",
        "no_action (excluded type)":      "No Action (Excluded)",
    }
    _SEV_ORDER = {"SEVERE": 0, "MODERATE": 1, "MILD": 2}

    if not drift_summary.drifted:
        print("No drift detected across all features.")
    else:
        def _desc(r):
            sev = r.drift_severity.value
            ft  = r.feature_type
            if ft == "numerical":
                if sev == "SEVERE":   return "Feature ranges explode in test set"
                if sev == "MODERATE": return "Feature demonstrates moderate distribution shift in test set"
                return "Feature distribution shifts mildly in test set"
            if ft in ("categorical", "high_cardinality"):
                if sev == "SEVERE":   return "Feature has new set of categories in test set"
                return "Feature category proportions shift in test set"
            return f"{sev} drift detected"

        def _mit(r):
            mv = r.mitigation.value if hasattr(r.mitigation, "value") else str(r.mitigation)
            if mv == "frequency_encoding":
                return "Target Encoding (Laplace, m=10)"
            return _MIT_LABEL.get(mv, mv.replace("_", " ").title())

        rows = sorted(
            drift_summary.drifted,
            key=lambda x: (_SEV_ORDER.get(x.drift_severity.value, 9),
                           -(x.test_statistic or 0)),
        )
        col1 = [r.column       for r in rows]
        col2 = [r.feature_type for r in rows]
        col3 = [_desc(r)       for r in rows]
        col4 = [_mit(r)        for r in rows]

        h1, h2, h3, h4 = "Columns with Drift", "Column Type", "Drift Description", "Drift Mitigation"
        w1 = max(len(h1), max(len(v) for v in col1))
        w2 = max(len(h2), max(len(v) for v in col2))
        w3 = max(len(h3), max(len(v) for v in col3))
        w4 = max(len(h4), max(len(v) for v in col4))

        sep = f"+{'-'*(w1+2)}+{'-'*(w2+2)}+{'-'*(w3+2)}+{'-'*(w4+2)}+"
        hdr = f"| {h1:<{w1}} | {h2:<{w2}} | {h3:<{w3}} | {h4:<{w4}} |"
        print(sep)
        print(hdr)
        print(sep)
        for c1, c2, c3, c4 in zip(col1, col2, col3, col4):
            print(f"| {c1:<{w1}} | {c2:<{w2}} | {c3:<{w3}} | {c4:<{w4}} |")
            print(sep)

    # ── Time taken block ──────────────────────────────────────────────────────
    print(f"\nb. Detection + Mitigation Time Taken")
    print(f"+-----------------+")
    print(f"| Time Taken (s)  |")
    print(f"+-----------------+")
    print(f"| {_t2_elapsed:<15.1f} |")
    print(f"+-----------------+")

    return manifest, drift_summary


def stage3_rank(
    drift_summary: object,
    test_df: pl.DataFrame,
    manifest: object = None,
) -> pl.DataFrame:
    """Build drift_results DataFrame from DriftSummary and rank by recency weight."""
    rows = [
        {
            "feature":        r.column,
            "col_type":       r.feature_type,
            "test_used":      r.test_method.value,
            "raw_score":      float(r.test_statistic) if r.test_statistic is not None else 0.0,
            "drift_severity": r.drift_severity.value,
        }
        for r in drift_summary.drifted
    ]

    if not rows:
        # No drift detected — return empty DataFrame with correct schema
        ranked_df = pl.DataFrame(schema={
            "feature":        pl.String,
            "col_type":       pl.String,
            "test_used":      pl.String,
            "raw_score":      pl.Float64,
            "weighted_score": pl.Float64,
            "drift_rank":     pl.UInt32,
            "drift_severity": pl.String,   # matches non-empty path from rank_drift()
        })
        print("✅ Stage 3 complete: no drifted features to rank")
        return ranked_df

    drift_results_df = pl.DataFrame(rows)
    ranked_df = rank_drift(
        drift_results=drift_results_df,
        serve_df=test_df,
        lambda_decay=0.1,
        month_col=(manifest.time[0] if manifest and manifest.time else 'Month'),
    )

    top_feature = ranked_df["feature"][0]
    print(f"✅ Stage 3 complete: top drifted feature = {top_feature}")
    return ranked_df


def stage4_mitigate(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    ranked_df: pl.DataFrame,
    positive_label: str = "Yes",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Apply target encoding and robust scaling to drifted columns."""
    mitigated_train, mitigated_test = mitigate(train_df, test_df, ranked_df, positive_label=positive_label)
    print(f"✅ Stage 4 complete: {len(ranked_df)} columns mitigated")
    return mitigated_train, mitigated_test


def stage5_model_and_output(
    mitigated_train: pl.DataFrame,
    mitigated_test: pl.DataFrame,
    test_df: pl.DataFrame,
    n_train: int,
    n_test: int,
    t0: float,
    _label_map: dict | None = None,
    _NON_FEAT: set | None = None,
) -> tuple[float, float | None]:
    """
    Step 5a: String normalisation in Polars (must happen before pandas handoff).
    Step 5b: Feature / target split.
    Step 5c: Pandas handoff.
    Step 5d: LightGBM train + predict.
    Step 5e: AU-PRC computation.
    Step 5f: Write prediction.csv.
    Step 5g: Print summary table.
    """
    # ── Step 5a: String normalisation — single .with_columns() pass ──────────
    # Identify all string/categorical columns in both DataFrames at once.
    str_cols_train = [
        c for c in mitigated_train.columns
        if mitigated_train[c].dtype in (pl.Utf8, pl.String, pl.Categorical)
    ]
    str_cols_test = [
        c for c in mitigated_test.columns
        if mitigated_test[c].dtype in (pl.Utf8, pl.String, pl.Categorical)
    ]

    _skip_norm = {_TARGET, _ID_COL}
    str_cols_train = [c for c in str_cols_train if c not in _skip_norm]
    str_cols_test  = [c for c in str_cols_test  if c not in _skip_norm]

    if str_cols_train:
        mitigated_train = mitigated_train.with_columns([
            pl.col(c).str.to_lowercase().str.strip_chars()
              .str.replace_all(r'[-_]', ' ').str.replace_all(r'\s+', ' ')
            for c in str_cols_train
        ])
    if str_cols_test:
        mitigated_test = mitigated_test.with_columns([
            pl.col(c).str.to_lowercase().str.strip_chars()
              .str.replace_all(r'[-_]', ' ').str.replace_all(r'\s+', ' ')
            for c in str_cols_test
        ])

    # ── Step 5b: Feature / target split ──────────────────────────────────────
    if _NON_FEAT is None:
        _NON_FEAT = {_TARGET, _ID_COL}
    feature_cols = [c for c in mitigated_train.columns if c not in _NON_FEAT]

    X_train = mitigated_train.select(feature_cols)
    y_train = mitigated_train.select(_TARGET)
    X_test  = mitigated_test.select([
        c for c in feature_cols if c in mitigated_test.columns
    ])

    # ── Step 5c: Pandas handoff (ONLY here) ──────────────────────────────────
    X_train_pd = X_train.to_pandas()
    y_train_pd = y_train.to_pandas()[_TARGET].astype(str).map(_label_map)
    X_test_pd  = X_test.to_pandas()

    # Align any remaining object columns to CategoricalDtype
    # (after string normalisation, most should already be consistent)
    for col in X_train_pd.select_dtypes(include="object").columns:
        X_train_pd[col] = X_train_pd[col].astype("category")
        if col in X_test_pd.columns:
            # Copy the exact CategoricalDtype from train — including category labels
            X_test_pd[col] = X_test_pd[col].astype(X_train_pd[col].dtype)

    # ── Step 5d: LightGBM — EXACTLY these hyperparameters, no others ─────────
    model = lgb.LGBMClassifier(
        verbosity       = -1,
        objective       = "binary",
        is_unbalance    = True,
        random_state    = 42,
        importance_type = "gain",
    )
    model.fit(X_train_pd, y_train_pd)
    import joblib
    joblib.dump(model, Path(__file__).resolve().parent.parent / "model.joblib")

    # Export gain importance for dashboard
    import pandas as _pd
    _fi_df = _pd.DataFrame({
        "feature": model.booster_.feature_name(),
        "importance": model.booster_.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    _fi_df.index += 1
    _fi_df.index.name = "rank"
    _fi_df.to_csv(Path(__file__).resolve().parent.parent / "feature_importance.csv")
    print(f"  Saved feature_importance.csv → {Path(__file__).resolve().parent.parent / 'feature_importance.csv'}")

    proba_train = model.predict_proba(X_train_pd)[:, 1]
    proba_test  = model.predict_proba(X_test_pd)[:, 1]

    # ── Step 5e: AU-PRC ───────────────────────────────────────────────────────
    au_prc_train = average_precision_score(y_train_pd, proba_train)

    if _TARGET in test_df.columns:
        y_test_pd = test_df.select(_TARGET).to_pandas()[_TARGET].astype(str).map(_label_map)
        au_prc_test = average_precision_score(y_test_pd, proba_test)
        au_prc_display = au_prc_test
    else:
        au_prc_test = None
        au_prc_display = au_prc_train

    # AU-PRC table (bordered)
    print("\nAU-PRC Results")
    _h1, _h2 = "Split", "AU-PRC"
    _rows = [("Train", f"{au_prc_train:.4f}")]
    if au_prc_test is not None:
        _rows.append(("Test", f"{au_prc_test:.4f}"))
    else:
        _rows.append(("Test", "N/A (no ground truth)"))
    _w1 = max(len(_h1), max(len(r[0]) for r in _rows))
    _w2 = max(len(_h2), max(len(r[1]) for r in _rows))
    _sep = f"+{'-'*(_w1+2)}+{'-'*(_w2+2)}+"
    print(_sep)
    print(f"| {_h1:<{_w1}} | {_h2:<{_w2}} |")
    print(_sep)
    for _c1, _c2 in _rows:
        print(f"| {_c1:<{_w1}} | {_c2:<{_w2}} |")
        print(_sep)

    # ── Confusion matrix (interpretability only) ──────────────────────────────
    if _TARGET in test_df.columns:
        from sklearn.metrics import confusion_matrix as _cm_fn
        _preds = (proba_test >= 0.5).astype(int)
        _cm    = _cm_fn(y_test_pd, _preds)
        _tn, _fp, _fn, _tp = _cm.ravel()
        _prec = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0.0
        _rec  = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0
        _f1   = (2 * _prec * _rec / (_prec + _rec)
                 if (_prec + _rec) > 0 else 0.0)
        print("\nConfusion Matrix (threshold = 0.50)")
        _ch = ["Actual \\ Predicted", "Predicted No", "Predicted Yes"]
        _cm_rows = [
            ("Actual No", f"{_tn:,}", f"{_fp:,}"),
            ("Actual Yes", f"{_fn:,}", f"{_tp:,}"),
        ]
        _w1 = max(len(_ch[0]), max(len(r[0]) for r in _cm_rows))
        _w2 = max(len(_ch[1]), max(len(r[1]) for r in _cm_rows))
        _w3 = max(len(_ch[2]), max(len(r[2]) for r in _cm_rows))
        _sep = f"+{'-'*(_w1+2)}+{'-'*(_w2+2)}+{'-'*(_w3+2)}+"
        print(_sep)
        print(f"| {_ch[0]:<{_w1}} | {_ch[1]:<{_w2}} | {_ch[2]:<{_w3}} |")
        print(_sep)
        for _r1, _r2, _r3 in _cm_rows:
            print(f"| {_r1:<{_w1}} | {_r2:<{_w2}} | {_r3:<{_w3}} |")
            print(_sep)

        print("\nConfusion Matrix Metrics")
        _mh1, _mh2 = "Metric", "Value"
        _mrows = [
            ("Precision", f"{_prec:.4f}"),
            ("Recall", f"{_rec:.4f}"),
            ("F1", f"{_f1:.4f}"),
        ]
        _mw1 = max(len(_mh1), max(len(r[0]) for r in _mrows))
        _mw2 = max(len(_mh2), max(len(r[1]) for r in _mrows))
        _msep = f"+{'-'*(_mw1+2)}+{'-'*(_mw2+2)}+"
        print(_msep)
        print(f"| {_mh1:<{_mw1}} | {_mh2:<{_mw2}} |")
        print(_msep)
        for _m1, _m2 in _mrows:
            print(f"| {_m1:<{_mw1}} | {_m2:<{_mw2}} |")
            print(_msep)

    # ── Step 5f: Write prediction.csv to project root ─────────────────────────
    # Project root = parent of src/
    output_path = Path(__file__).resolve().parent.parent / "prediction.csv"

    prediction_df = pl.DataFrame({
        _ID_COL:            test_df[_ID_COL].cast(pl.String),
        "probability_score": pl.Series(proba_test.astype(float)),
    })
    prediction_df.write_csv(str(output_path))
    print(f"  Saved {len(prediction_df):,} predictions → {output_path}")
    print(f"  Saved model      → {Path(__file__).resolve().parent.parent / 'model.joblib'}")

    # ── Step 5g: Summary table ────────────────────────────────────────────────
    elapsed = time.time() - t0

    print()
    print("==========================================")
    print("NAISC 2026 — Pipeline Execution Summary")
    print(f"{'Metric':<20} {'Value':>18}")
    print(f"{'AU-PRC':<20} {au_prc_display:>18.4f}")
    print(f"{'Train Set':<20} {f'{n_train:,} rows':>18}")
    print(f"{'Test Set':<20} {f'{n_test:,} rows':>18}")
    print(f"{'Total Runtime':<20} {f'{elapsed:.1f}s':>18}")
    print("==========================================")

    return au_prc_train, au_prc_test


# ── Dashboard feed export ─────────────────────────────────────────────────────

def export_dashboard_feeds(
    drift_summary: object,
    ranked_df: pl.DataFrame,
    manifest: object,
    n_train: int,
    n_test: int,
    au_prc_train: float,
    au_prc_test: float | None,
    runtime_seconds: float,
) -> None:
    """
    Export CSV/JSON files consumed by dashboard_ver4.py.
    All files are written to the project root (parent of src/),
    alongside prediction.csv and model.joblib.

    Files written:
        drift_results.csv    — one row per drifted feature (Stage 2 output)
        ranked_features.csv  — Stage 3 ranked output with weighted scores
        pipeline_meta.json   — AU-PRC scores, row counts, feature count, runtime
    """
    import pandas as pd
    import json

    root = Path(__file__).resolve().parent.parent

    # ── 1. drift_results.csv ─────────────────────────────────────────────────
    _MIT_LABEL = {
        "robust_scaling":                 "robust_scaling",
        "quantile_binning":               "quantile_binning",
        "log_transform + robust_scaling": "log_transform + robust_scaling",
        "frequency_encoding":             "frequency_encoding",
        "drop_feature":                   "drop_feature",
        "none":                           "none",
        "no_action (stable)":             "no_action (stable)",
        "no_action (excluded type)":      "no_action (excluded type)",
    }
    drift_rows = []
    for r in drift_summary.drifted:
        mit_val = r.mitigation.value if hasattr(r.mitigation, "value") else str(r.mitigation)
        drift_rows.append({
            "feature":        r.column,
            "feature_type":   r.feature_type,
            "drift_severity": r.drift_severity.value,
            "test_used":      r.test_method.value,
            "test_statistic": float(r.test_statistic) if r.test_statistic is not None else None,
            "mitigation":     _MIT_LABEL.get(mit_val, mit_val),
        })
    pd.DataFrame(drift_rows).to_csv(root / "drift_results.csv", index=False)
    print(f"  Saved drift_results.csv     → {root / 'drift_results.csv'}")

    # ── 2. ranked_features.csv ───────────────────────────────────────────────
    ranked_df.to_pandas().to_csv(root / "ranked_features.csv", index=False)
    print(f"  Saved ranked_features.csv   → {root / 'ranked_features.csv'}")

    # ── 3. pipeline_meta.json ────────────────────────────────────────────────
    _excluded_feat_types = {"metadata", "time", "target"}
    # _excluded_feat_types = {"metadata", "constant", "time", "target"}
    n_features = len([
        col for col, profile in manifest.profiles.items()
        if (profile.feature_type.value
            if hasattr(profile.feature_type, "value")
            else str(profile.feature_type)).lower()
        not in _excluded_feat_types
    ])
    meta = {
        "n_train":         n_train,
        "n_test":          n_test,
        "n_features":      n_features,
        "au_prc_train":    round(au_prc_train, 6),
        "runtime_seconds": round(runtime_seconds, 2),
    }
    if au_prc_test is not None:
        meta["au_prc_test"] = round(au_prc_test, 6)

    with open(root / "pipeline_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved pipeline_meta.json    → {root / 'pipeline_meta.json'}")

    print(f"✅ Dashboard feeds exported → {root}")

# ── Main entrypoint ───────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # Record wall-clock start immediately after arg validation
    t0 = time.time()

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    train_df, test_df = stage1_ingest(
        args.train_data_filepath,
        args.test_data_filepath,
    )
    n_train, n_test = len(train_df), len(test_df)

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    manifest, drift_summary = stage2_classify_and_detect(train_df, test_df)

    # ── Build dynamic _NON_FEAT from manifest ────────────────────────────────
    _excluded_types = {FeatureType.METADATA, FeatureType.CONSTANT, FeatureType.TIME, FeatureType.TARGET}
    _NON_FEAT = {col for col, profile in manifest.profiles.items() if profile.feature_type in _excluded_types}
    _NON_FEAT.update({_TARGET, _ID_COL})

    # ── Detect positive label dynamically ──────────────────────────────────
    _target_series_raw = train_df.select(_TARGET).to_series().cast(pl.String)
    _unique_labels = sorted(_target_series_raw.unique().to_list())
    assert len(_unique_labels) == 2, f"Target must be binary, found: {_unique_labels}"
    _pos_priority = ["yes", "1", "true", "y", "churn"]
    _pos_label_found = None
    for hint in _pos_priority:
        for lbl in _unique_labels:
            if lbl.lower().startswith(hint) and _pos_label_found is None:
                _pos_label_found = lbl
                break
        if _pos_label_found:
            break
    if _pos_label_found is None:
        raise ValueError(
            f"Could not identify positive churn label from: {_unique_labels}. "
            f"Add the correct label string to _pos_priority and rerun."
        )
    _label_map = {lbl: (1 if lbl == _pos_label_found else 0) for lbl in _unique_labels}
    print(f"Target label mapping: {_label_map}  (positive={_pos_label_found})")

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    ranked_df = stage3_rank(drift_summary, test_df, manifest=manifest)

    # ── Stage 4 ───────────────────────────────────────────────────────────────
    mitigated_train, mitigated_test = stage4_mitigate(train_df, test_df, ranked_df, positive_label=_pos_label_found)

    mitigated_train, mitigated_test, policy_features = _apply_mild_mod_numerical_policy(
        policy=args.mild_mod_num_policy,
        ranked_df=ranked_df,
        raw_train=train_df,
        raw_test=test_df,
        mitigated_train=mitigated_train,
        mitigated_test=mitigated_test,
    )
    _print_policy_summary(args.mild_mod_num_policy, policy_features)

    # ── Stage 5 ───────────────────────────────────────────────────────────────
    au_prc_train, au_prc_test = stage5_model_and_output(
        mitigated_train, mitigated_test,
        test_df,
        n_train, n_test,
        t0,
        _label_map=_label_map,
        _NON_FEAT=_NON_FEAT,
    )

    # ── Dashboard feed export ─────────────────────────────────────────────────
    runtime = time.time() - t0
    export_dashboard_feeds(
        drift_summary   = drift_summary,
        ranked_df       = ranked_df,
        manifest        = manifest,
        n_train         = n_train,
        n_test          = n_test,
        au_prc_train    = au_prc_train,
        au_prc_test     = au_prc_test,
        runtime_seconds = runtime,
    )

    if not args.skip_dashboard:
        _launch_dashboard()

if __name__ == "__main__":
    main()
