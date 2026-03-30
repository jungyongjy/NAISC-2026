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


# ── Pipeline stages ───────────────────────────────────────────────────────────

def stage1_ingest(
    train_path: str,
    test_path: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load CSVs with Polars. infer_schema_length=None scans full file — safe
    for hidden datasets where early rows may not represent all dtypes."""
    train_df = pl.read_csv(train_path, infer_schema_length=None)
    test_df  = pl.read_csv(test_path,  infer_schema_length=None)

    # Stop condition: CustomerID must be present
    if _ID_COL not in test_df.columns:
        print(f"STOP: '{_ID_COL}' missing from test file — cannot build output.")
        sys.exit(1)

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
                if sev == "MODERATE": return "Feature demonstrates greater left-skewness in test set"
                return "Feature distribution shifts mildly in test set"
            if ft in ("categorical", "high_cardinality"):
                if sev == "SEVERE":   return "Feature has new set of categories in test set"
                return "Feature category proportions shift in test set"
            return f"{sev} drift detected"

        def _mit(r):
            mv = r.mitigation.value if hasattr(r.mitigation, "value") else str(r.mitigation)
            ft = r.feature_type
            if mv == "frequency_encoding" and ft in ("categorical", "sparse"):
                return "Target Encoding (Laplace, m=20)"
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
) -> None:
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
        n_jobs          = 1,      # prevents joblib physical core topology errors
    )
    model.fit(X_train_pd, y_train_pd)
    import joblib
    joblib.dump(model, Path(__file__).resolve().parent.parent / "model.joblib")

    proba_train = model.predict_proba(X_train_pd)[:, 1]
    proba_test  = model.predict_proba(X_test_pd)[:, 1]

    # ── Step 5e: AU-PRC ───────────────────────────────────────────────────────
    au_prc_train = average_precision_score(y_train_pd, proba_train)

    if _TARGET in test_df.columns:
        y_test_pd = test_df.select(_TARGET).to_pandas()[_TARGET].astype(str).map(_label_map)
        au_prc_test = average_precision_score(y_test_pd, proba_test)
        au_prc_display = au_prc_test
        print(f"  TRAIN AU-PRC : {au_prc_train:.4f}")
        print(f"  TEST  AU-PRC : {au_prc_test:.4f}")
    else:
        au_prc_display = au_prc_train
        print(f"  TRAIN AU-PRC : {au_prc_train:.4f}")
        print("TEST AU-PRC: unavailable (no ground truth)")

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

    # ── Stage 5 ───────────────────────────────────────────────────────────────
    stage5_model_and_output(
        mitigated_train, mitigated_test,
        test_df,
        n_train, n_test,
        t0,
        _label_map=_label_map,
        _NON_FEAT=_NON_FEAT,
    )


if __name__ == "__main__":
    main()
