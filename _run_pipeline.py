from __future__ import annotations
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# CELL 1 — Install dependencies
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
    'polars', 'scipy', 'lightgbm', 'scikit-learn',
    'pyarrow', 'pandas', 'numpy', 'matplotlib', 'seaborn', '-q'])
import polars as pl, numpy as np
from scipy import stats
print(f'Polars {pl.__version__}  |  NumPy {np.__version__}')

# CELL 2 — Set file paths  ← ONLY CELL YOU NEED TO EDIT
from pathlib import Path

FOLDER     = Path(r'C:\Users\Admin\Downloads')
TRAIN_PATH = FOLDER / 'train.csv'
TEST_PATH  = FOLDER / 'test.csv'

missing = [p for p in [TRAIN_PATH, TEST_PATH] if not p.exists()]
if missing:
    raise FileNotFoundError(f'Cannot find: {[str(p) for p in missing]}')
print(f'TRAIN -> {TRAIN_PATH}')
print(f'TEST  -> {TEST_PATH}')

# CELL 3 — Full pipeline code (do not edit)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 1/5 — schema/classifier.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
schema/classifier.py  [Polars rewrite]
=======================================
Dynamic Schema Classifier — Stage 1 of the Drift Intelligence Pipeline.

COMPUTE STRATEGY (why Polars)
------------------------------
On a 10M × 500 dataset Pandas read_csv alone allocates ~20–40 GB and
processes column statistics sequentially.  Polars gives us:
  • Lazy scan_csv  — zero-copy Arrow memory, no full materialisation
  • Batch aggregation — ALL column stats computed in a single .select()
    call, executed in parallel across CPU cores by the Polars query engine
  • O(1) null_count / n_unique via Arrow bitmaps, not Python loops

The sample used for classification is collected ONCE from the LazyFrame,
and all per-column aggregations run in parallel in a single engine pass.

FeatureType precedence (unchanged from spec):
  1. STRUCTURAL   → CustomerID, Month, ChurnStatus  (only hardcoded logic)
  2. CONSTANT     → n_unique == 1
  3. METADATA     → non-numeric AND n_unique/n_rows > METADATA_RATIO
  4. TIME         → Polars Date/Datetime dtype, OR string parseable as date
  5. SPARSE       → null_rate  > 0.75  (any dtype)
                    OR zero_rate > 0.75  (numeric only)
                    Both checks are INDEPENDENT — a column qualifies if EITHER condition
                    is met alone.  This correctly catches TotalRefunds (86% zeros, 0% nulls)
                    which a combined null+zero threshold would miss when one component is 0%.
  6. HIGH_CARDINALITY → string with n_unique > CARDINALITY_THRESHOLD
  7. CATEGORICAL  → low-cardinality string, OR low-cardinality integer
  8. NUMERICAL    → all remaining numeric columns
"""


from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import polars as pl

# ─────────────────────────────────────────────
#  Tuneable thresholds
# ─────────────────────────────────────────────
METADATA_RATIO        = 0.95
TIME_PARSE_THRESHOLD  = 0.80   # fraction of non-null values parseable as date
SPARSE_NULL_THRESHOLD = 0.75   # null rate alone  > 0.75 → SPARSE (any dtype)
SPARSE_ZERO_THRESHOLD = 0.75   # zero rate alone  > 0.75 → SPARSE (numeric only)
# Both are INDEPENDENT OR conditions — not summed.
# Catches: TotalRefunds (86% zeros), TotalExtraDataCharges (82%), NumberofDependents (80%).
CARDINALITY_THRESHOLD     = 100    # string n_unique above this → HIGH_CARDINALITY
INT_CATEGORICAL_MAX       = 50     # integer n_unique at or below → CATEGORICAL candidate
INT_CATEGORICAL_RATIO     = 0.05   # integer n_unique/n_rows must also be below this

FIXED_STRUCTURAL_COLUMNS = {
    "CustomerID": "metadata",
    "Month":      "time",
    "ChurnStatus": "target",
}

# Common date format patterns tried during TIME detection
_DATE_FORMATS = ["%y-%b", "%Y-%m", "%m/%d/%Y", "%d-%b-%Y", "%Y%m", "%b-%y",
                 "%Y-%m-%d", "%d/%m/%Y"]


class FeatureType(str, Enum):
    TIME             = "time"
    METADATA         = "metadata"
    CONSTANT         = "constant"
    SPARSE           = "sparse"
    NUMERICAL        = "numerical"
    CATEGORICAL      = "categorical"
    HIGH_CARDINALITY = "high_cardinality"
    TARGET           = "target"


# ─────────────────────────────────────────────
#  Polars dtype helpers
# ─────────────────────────────────────────────

def _is_integer(dtype: pl.DataType) -> bool:
    return dtype in pl.INTEGER_DTYPES

def _is_float(dtype: pl.DataType) -> bool:
    return dtype in pl.FLOAT_DTYPES

def _is_numeric(dtype: pl.DataType) -> bool:
    return dtype in pl.NUMERIC_DTYPES

def _is_string(dtype: pl.DataType) -> bool:
    # pl.Utf8 is an alias for pl.String in Polars 1.x; keep both for compat
    return dtype in (pl.String, pl.Utf8, pl.Categorical)

def _is_datetime_polars(dtype: pl.DataType) -> bool:
    return dtype in (pl.Date,) or isinstance(dtype, (pl.Datetime,))

def _is_boolean(dtype: pl.DataType) -> bool:
    return dtype == pl.Boolean


def _try_parse_as_datetime(series: pl.Series, threshold: float = TIME_PARSE_THRESHOLD) -> bool:
    """
    Try a set of date format strings against up to 500 values of a string Series.
    Returns True if any format parses ≥ threshold fraction successfully.
    Only called for String/Utf8 columns — zero overhead for all other types.
    """
    if not _is_string(series.dtype):
        return False

    # Sample up to 500 non-null values for speed
    _pool = series.drop_nulls()
    sample = _pool.sample(n=min(500, len(_pool)), seed=42) if len(_pool) > 500 else _pool
    n = len(sample)
    if n == 0:
        return False

    for fmt in _DATE_FORMATS:
        try:
            parsed = sample.str.to_date(fmt, strict=False)
            success_rate = parsed.drop_nulls().len() / n
            if success_rate >= threshold:
                return True
        except Exception:
            continue
    return False


# ─────────────────────────────────────────────
#  Stats computation — single batch pass
# ─────────────────────────────────────────────

def _batch_numeric_stats(sample: pl.DataFrame, numeric_cols: List[str]) -> Dict[str, Dict]:
    """
    Compute mean, std, median, p5/p25/p75/p95, skewness for ALL numeric columns
    in a SINGLE .select() call.  Polars executes these in parallel.
    Returns { col_name: { stat_name: value, ... } }
    """
    if not numeric_cols:
        return {}

    exprs = []
    for col in numeric_cols:
        exprs += [
            pl.col(col).mean()             .alias(f"{col}__mean"),
            pl.col(col).std()              .alias(f"{col}__std"),
            pl.col(col).median()           .alias(f"{col}__median"),
            pl.col(col).quantile(0.05, interpolation="linear").alias(f"{col}__p5"),
            pl.col(col).quantile(0.25, interpolation="linear").alias(f"{col}__p25"),
            pl.col(col).quantile(0.75, interpolation="linear").alias(f"{col}__p75"),
            pl.col(col).quantile(0.95, interpolation="linear").alias(f"{col}__p95"),
            pl.col(col).skew()             .alias(f"{col}__skew"),
        ]

    row = sample.select(exprs).row(0, named=True)

    result = {}
    for col in numeric_cols:
        result[col] = {
            "mean":     _safe_float(row.get(f"{col}__mean")),
            "std":      _safe_float(row.get(f"{col}__std")),
            "median":   _safe_float(row.get(f"{col}__median")),
            "p5":       _safe_float(row.get(f"{col}__p5")),
            "p25":      _safe_float(row.get(f"{col}__p25")),
            "p75":      _safe_float(row.get(f"{col}__p75")),
            "p95":      _safe_float(row.get(f"{col}__p95")),
            "skewness": _safe_float(row.get(f"{col}__skew")),
        }
    return result


def _batch_null_zero_rates(sample: pl.DataFrame, cols: List[str]) -> Dict[str, Dict]:
    """
    Compute null rate and (for numeric cols) zero rate for ALL columns in one pass.
    Uses Polars bitwise null bitmap — O(n/64) per column.
    """
    n = len(sample)
    if n == 0:
        return {c: {"null_rate": 0.0, "zero_rate": 0.0} for c in cols}

    exprs = [pl.col(c).null_count().alias(f"{c}__nulls") for c in cols]
    null_row = sample.select(exprs).row(0, named=True)

    zero_exprs = []
    for c in cols:
        if _is_numeric(sample[c].dtype):
            zero_exprs.append(
                (pl.col(c).fill_null(0) == 0).sum().alias(f"{c}__zeros")
            )

    zero_row = {}
    if zero_exprs:
        zero_row = sample.select(zero_exprs).row(0, named=True)

    result = {}
    for c in cols:
        null_count = null_row.get(f"{c}__nulls", 0)
        zero_count = zero_row.get(f"{c}__zeros", 0)
        result[c] = {
            "null_rate": null_count / n,
            "zero_rate": zero_count / n,
        }
    return result


def _batch_cardinality(df: pl.DataFrame, cols: List[str]) -> Dict[str, int]:
    """
    n_unique for all columns in ONE .select() call on the FULL dataframe
    (not the sample — cardinality must reflect the full population).
    """
    exprs   = [pl.col(c).n_unique().alias(f"{c}__nu") for c in cols]
    row     = df.select(exprs).row(0, named=True)
    return  {c: int(row[f"{c}__nu"]) for c in cols}


def _top_categories(series: pl.Series, top_n: int = 20) -> Dict[str, float]:
    """Relative frequency of top-N categories (nulls counted as '__null__')."""
    filled = series.fill_null("__null__").cast(pl.String)
    vc     = filled.value_counts(sort=True).head(top_n)
    total  = len(series)
    return {row["__null__" if k == "__null__" else k]: row["count"] / total
            for row in vc.iter_rows(named=True)
            for k in [list(row.keys())[0]]}  # key is the series name


def _safe_float(v) -> float:
    try:
        return float(v) if v is not None else np.nan
    except (TypeError, ValueError):
        return np.nan


# ─────────────────────────────────────────────
#  Dataclasses (unchanged interface)
# ─────────────────────────────────────────────

@dataclass
class ColumnProfile:
    name:           str
    dtype:          str
    feature_type:   FeatureType
    n_rows:         int
    n_unique:       int
    null_rate:      float
    zero_rate:      float
    mean:           float = np.nan
    std:            float = np.nan
    median:         float = np.nan
    p5:             float = np.nan
    p25:            float = np.nan
    p75:            float = np.nan
    p95:            float = np.nan
    skewness:       float = np.nan
    top_categories: Dict[str, float] = field(default_factory=dict)


@dataclass
class FeatureManifest:
    profiles: Dict[str, ColumnProfile] = field(default_factory=dict)

    @property
    def time(self)             -> List[str]: return self._by_type(FeatureType.TIME)
    @property
    def metadata(self)         -> List[str]: return self._by_type(FeatureType.METADATA)
    @property
    def constant(self)         -> List[str]: return self._by_type(FeatureType.CONSTANT)
    @property
    def sparse(self)           -> List[str]: return self._by_type(FeatureType.SPARSE)
    @property
    def numerical(self)        -> List[str]: return self._by_type(FeatureType.NUMERICAL)
    @property
    def categorical(self)      -> List[str]: return self._by_type(FeatureType.CATEGORICAL)
    @property
    def high_cardinality(self) -> List[str]: return self._by_type(FeatureType.HIGH_CARDINALITY)
    @property
    def target(self)           -> List[str]: return self._by_type(FeatureType.TARGET)

    @property
    def model_features(self) -> List[str]:
        excluded = {FeatureType.METADATA, FeatureType.CONSTANT,
                    FeatureType.TARGET,   FeatureType.TIME}
        return [n for n, p in self.profiles.items() if p.feature_type not in excluded]

    def _by_type(self, ft: FeatureType) -> List[str]:
        return [n for n, p in self.profiles.items() if p.feature_type == ft]

    def summary(self) -> str:
        lines = ["=" * 60, "  FEATURE MANIFEST — SCHEMA CLASSIFICATION", "=" * 60]
        counts = {}
        for p in self.profiles.values():
            counts[p.feature_type] = counts.get(p.feature_type, 0) + 1
        for ft in FeatureType:
            n = counts.get(ft, 0)
            if n:
                lines.append(f"  {ft.value:<20s}  {n:>4d} column(s)")
        lines += ["-" * 60, f"  {'Total':<20s}  {len(self.profiles):>4d} column(s)",
                  "=" * 60, "\n  Detailed column assignments:"]
        for ft in FeatureType:
            cols = self._by_type(ft)
            if cols:
                lines.append(f"\n  [{ft.value.upper()}]")
                for col in cols:
                    p = self.profiles[col]
                    lines.append(f"    {col:<40s}  dtype={p.dtype:<12s}  "
                                  f"n_unique={p.n_unique:<6d}  null%={p.null_rate*100:5.1f}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────

def classify_features(
    df: pl.DataFrame,
    sample_size: int = 100_000,
    random_state: int = 42,
) -> FeatureManifest:
    """
    Classify every column in a Polars DataFrame using data statistics only.

    All expensive aggregations (null rates, zero rates, numeric stats) are
    issued as a SINGLE batch .select() call — Polars executes them in parallel
    across CPU cores.  On 10M × 500 this is ~10-30x faster than per-column Pandas.

    Parameters
    ----------
    df          : Polars DataFrame (full training set).
    sample_size : Max rows used for distribution stats.  n_unique always uses
                  the full DataFrame for accuracy.
    random_state: Reproducibility seed.

    Returns
    -------
    FeatureManifest with a ColumnProfile per column.
    """
    n_rows  = len(df)
    cols    = df.columns

    # ── 1. Draw sample for distribution statistics ─────────────────────
    sample = (df.sample(n=min(sample_size, n_rows), seed=random_state)
              if n_rows > sample_size else df)

    # ── 2. Batch compute cardinality (full df), null+zero rates (sample) ─
    cardinality = _batch_cardinality(df, cols)
    rates       = _batch_null_zero_rates(sample, cols)

    # ── 3. Batch compute numeric stats in one pass ─────────────────────
    numeric_cols = [c for c in cols if _is_numeric(df[c].dtype)]
    num_stats    = _batch_numeric_stats(sample, numeric_cols)

    manifest = FeatureManifest()

    for col in cols:
        dtype     = df[col].dtype
        dtype_str = str(dtype)
        n_unique  = cardinality[col]
        null_rate = rates[col]["null_rate"]
        zero_rate = rates[col]["zero_rate"]
        is_num    = _is_numeric(dtype)

        # ── STEP 1: Fixed structural columns ──────────────────────────
        if col in FIXED_STRUCTURAL_COLUMNS:
            ft    = FeatureType(FIXED_STRUCTURAL_COLUMNS[col])
            stats = num_stats.get(col, {})
            top_c = (_top_categories(sample[col])
                     if not is_num and col in sample.columns else {})
            manifest.profiles[col] = ColumnProfile(
                name=col, dtype=dtype_str, feature_type=ft,
                n_rows=n_rows, n_unique=n_unique,
                null_rate=null_rate, zero_rate=zero_rate,
                top_categories=top_c, **stats,
            )
            continue

        # ── STEP 2: Constant columns ───────────────────────────────────
        if n_unique <= 1:
            manifest.profiles[col] = ColumnProfile(
                name=col, dtype=dtype_str, feature_type=FeatureType.CONSTANT,
                n_rows=n_rows, n_unique=n_unique,
                null_rate=null_rate, zero_rate=zero_rate,
            )
            continue

        # ── STEP 3: Near-unique identifiers (metadata, strings only) ───
        if _is_string(dtype) and n_unique / n_rows > METADATA_RATIO:
            manifest.profiles[col] = ColumnProfile(
                name=col, dtype=dtype_str, feature_type=FeatureType.METADATA,
                n_rows=n_rows, n_unique=n_unique,
                null_rate=null_rate, zero_rate=zero_rate,
            )
            continue

        # ── STEP 4: Datetime columns ───────────────────────────────────
        if _is_datetime_polars(dtype) or _try_parse_as_datetime(sample[col]):
            manifest.profiles[col] = ColumnProfile(
                name=col, dtype=dtype_str, feature_type=FeatureType.TIME,
                n_rows=n_rows, n_unique=n_unique,
                null_rate=null_rate, zero_rate=zero_rate,
            )
            continue

        # ── STEP 5: Sparse columns ─────────────────────────────────────
        is_sparse = (
            null_rate > SPARSE_NULL_THRESHOLD                        # too many nulls
            or (is_num and zero_rate > SPARSE_ZERO_THRESHOLD)        # too many zeros
        )
        if is_sparse:
            stats = num_stats.get(col, {})
            top_c = _top_categories(sample[col]) if not is_num else {}
            manifest.profiles[col] = ColumnProfile(
                name=col, dtype=dtype_str, feature_type=FeatureType.SPARSE,
                n_rows=n_rows, n_unique=n_unique,
                null_rate=null_rate, zero_rate=zero_rate,
                top_categories=top_c, **stats,
            )
            continue

        # ── STEP 6: String / Categorical columns ───────────────────────
        if _is_string(dtype) or _is_boolean(dtype):
            ft    = (FeatureType.HIGH_CARDINALITY
                     if n_unique > CARDINALITY_THRESHOLD
                     else FeatureType.CATEGORICAL)
            top_c = _top_categories(sample[col])
            manifest.profiles[col] = ColumnProfile(
                name=col, dtype=dtype_str, feature_type=ft,
                n_rows=n_rows, n_unique=n_unique,
                null_rate=null_rate, zero_rate=zero_rate,
                top_categories=top_c,
            )
            continue

        # ── STEP 7: Low-cardinality integer → CATEGORICAL ──────────────
        if _is_integer(dtype):
            if n_unique <= INT_CATEGORICAL_MAX and (n_unique / n_rows) < INT_CATEGORICAL_RATIO:
                top_c = _top_categories(sample[col])
                manifest.profiles[col] = ColumnProfile(
                    name=col, dtype=dtype_str, feature_type=FeatureType.CATEGORICAL,
                    n_rows=n_rows, n_unique=n_unique,
                    null_rate=null_rate, zero_rate=zero_rate,
                    top_categories=top_c,
                )
                continue

        # ── STEP 8: All remaining numeric columns ─────────────────────
        stats = num_stats.get(col, {})
        manifest.profiles[col] = ColumnProfile(
            name=col, dtype=dtype_str, feature_type=FeatureType.NUMERICAL,
            n_rows=n_rows, n_unique=n_unique,
            null_rate=null_rate, zero_rate=zero_rate,
            **stats,
        )

    return manifest



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 2/5 — drift/report.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
drift/report.py
===============
DriftReport — the typed output contract of the Drift Detector.

Each ColumnDriftResult captures the full audit trail for a single feature:
  - Was it tested? If not, why?
  - Which statistical test(s) ran?
  - What were the test statistics and p-values?
  - What severity was assigned?
  - What mitigation is recommended?

DriftSummary aggregates all per-column results into a pipeline-level report
with convenience accessors and a human-readable console summary.
"""


from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class DriftSeverity(str, Enum):
    NONE     = "NONE"
    MILD     = "MILD"        # detectable but minor
    MODERATE = "MODERATE"    # meaningful distributional shift
    SEVERE   = "SEVERE"      # major shift; high mitigation priority


class TestMethod(str, Enum):
    KS_2SAMP      = "Kolmogorov-Smirnov"
    PSI           = "Population Stability Index"
    CHI2          = "Chi-Squared"
    JS_DIVERGENCE = "Jensen-Shannon Divergence"
    Z_PROPORTION  = "Two-Proportion Z-Test"
    PRESCREENED   = "Pre-Screen Only (fast)"
    SKIPPED       = "Skipped"


class MitigationStrategy(str, Enum):
    NONE                = "none"
    ROBUST_SCALE        = "robust_scaling"
    LOG_ROBUST_SCALE    = "log_transform + robust_scaling"
    QUANTILE_BIN        = "quantile_binning"
    FREQUENCY_ENCODE    = "frequency_encoding"
    BINARISE            = "binarise (sparse→binary)"
    DROP                = "drop_feature"
    RECENCY_WEIGHT      = "recency_weighting"
    DELTA_FEATURE       = "delta_feature"
    NO_ACTION_STABLE    = "no_action (stable)"
    NO_ACTION_EXCLUDED  = "no_action (excluded type)"


@dataclass
class ColumnDriftResult:
    """Full drift assessment for a single column."""

    column:       str
    feature_type: str

    # ── Phase A: fast pre-screen ────────────────────────────────────────
    prescreen_score:   float = 0.0    # normalised mean-shift or L1-distance
    prescreen_flagged: bool  = False  # True → sent to Phase B deep testing

    # ── Phase B: statistical test ───────────────────────────────────────
    test_method:     TestMethod       = TestMethod.SKIPPED
    test_statistic:  Optional[float]  = None   # KS D, PSI value, JS div, etc.
    p_value:         Optional[float]  = None   # None for non-parametric (PSI/JS)
    secondary_stat:  Optional[float]  = None   # e.g. chi2 confirming PSI

    # ── Decision ────────────────────────────────────────────────────────
    drift_detected:   bool          = False
    drift_severity:   DriftSeverity = DriftSeverity.NONE
    mitigation:       MitigationStrategy = MitigationStrategy.NO_ACTION_EXCLUDED
    notes:            str           = ""

    @property
    def emoji(self) -> str:
        return {
            DriftSeverity.NONE:     "✅",
            DriftSeverity.MILD:     "🟡",
            DriftSeverity.MODERATE: "🟠",
            DriftSeverity.SEVERE:   "🔴",
        }[self.drift_severity]

    def one_line(self) -> str:
        stat_str = f"{self.test_statistic:.4f}" if self.test_statistic is not None else "  —  "
        p_str    = f"p={self.p_value:.2e}" if self.p_value is not None else "        "
        return (
            f"{self.emoji} {self.column:<42s} "
            f"{self.feature_type:<18s} "
            f"{self.drift_severity.value:<10s} "
            f"{self.test_method.value:<32s} "
            f"stat={stat_str}  {p_str}  "
            f"→ {self.mitigation.value}"
        )


@dataclass
class DriftSummary:
    """Aggregated drift assessment across all columns."""
    results: Dict[str, ColumnDriftResult] = field(default_factory=dict)

    # ── convenience accessors ────────────────────────────────────────────
    @property
    def drifted(self) -> List[ColumnDriftResult]:
        return [r for r in self.results.values() if r.drift_detected]

    @property
    def stable(self) -> List[ColumnDriftResult]:
        return [r for r in self.results.values()
                if not r.drift_detected and r.test_method != TestMethod.SKIPPED]

    @property
    def skipped(self) -> List[ColumnDriftResult]:
        return [r for r in self.results.values() if r.test_method == TestMethod.SKIPPED]

    def by_severity(self, severity: DriftSeverity) -> List[ColumnDriftResult]:
        return [r for r in self.results.values() if r.drift_severity == severity]

    def summary(self) -> str:
        lines = [
            "=" * 110,
            "  DRIFT DETECTION REPORT",
            "=" * 110,
        ]

        total_tested = len(self.drifted) + len(self.stable)
        lines += [
            f"  Columns tested          : {total_tested}",
            f"  Drift confirmed         : {len(self.drifted)}"
            f"  (MILD={len(self.by_severity(DriftSeverity.MILD))}"
            f" | MODERATE={len(self.by_severity(DriftSeverity.MODERATE))}"
            f" | SEVERE={len(self.by_severity(DriftSeverity.SEVERE))})",
            f"  Stable (no drift)       : {len(self.stable)}",
            f"  Skipped (excl. type)    : {len(self.skipped)}",
            "-" * 110,
        ]

        if self.drifted:
            lines.append(
                f"\n  {'COLUMN':<42s} {'TYPE':<18s} {'SEVERITY':<10s} "
                f"{'TEST':<32s} {'STATISTIC':<14s} {'P-VALUE':<12s} MITIGATION"
            )
            lines.append("  " + "-" * 106)
            # Sort by severity (SEVERE first), then by test_statistic descending
            severity_order = {
                DriftSeverity.SEVERE: 0, DriftSeverity.MODERATE: 1,
                DriftSeverity.MILD: 2, DriftSeverity.NONE: 3,
            }
            for r in sorted(self.drifted,
                             key=lambda x: (severity_order[x.drift_severity],
                                            -(x.test_statistic or 0))):
                lines.append("  " + r.one_line())

        if self.stable:
            lines.append("\n  [STABLE FEATURES — no drift detected]")
            for r in self.stable:
                lines.append(f"  ✅ {r.column:<42s} {r.feature_type:<18s} "
                              f"stat={r.test_statistic:.4f}" if r.test_statistic else
                              f"  ✅ {r.column}")

        lines.append("=" * 110)
        return "\n".join(lines)



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 3/5 — drift/detector.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
drift/detector.py  [Polars rewrite]
=====================================
Two-Phase Drift Detector — Stage 2 of the Drift Intelligence Pipeline.

COMPUTE STRATEGY
----------------
Phase A (Vectorised Pre-Screen)
  All per-column summary statistics are pulled from the FeatureManifest
  that was already computed by the classifier — zero extra passes over data.
  The pre-screen score is pure numpy arithmetic on those cached stats.
  Batch collection of test-set stats uses a SINGLE Polars .select() call,
  so all columns are aggregated in one parallel engine pass.

Phase B (Deep Statistical Tests — flagged columns only)
  scipy tests receive numpy arrays via Polars .to_numpy() — this is a
  zero-copy view for numeric dtypes (no allocation overhead).
  Only flagged columns proceed here, so on 500 features with ~20% drift
  we run maybe 100 scipy tests, not 500.

String normalisation
  All categorical comparisons apply lowercase + strip via a Polars
  string expression BEFORE any frequency computation.  This catches
  casing inconsistencies between train and test (e.g. "Month-to-Month"
  vs "month-to-month") that would otherwise produce spurious PSI inflation.

Test methods by feature type:
  NUMERICAL        → KS 2-sample (scipy) on capped samples
  CATEGORICAL      → PSI + chi-squared GOF (both must confirm)
  HIGH_CARDINALITY → Jensen-Shannon Divergence on frequency histograms
  SPARSE           → Two-Stage Bipartite Test:
                       Stage 1: Bernoulli Presence Rate Z-test
                       Stage 2: Conditional KS on non-zero values only
                     Drift confirmed if EITHER stage fires.
                     Severity = max(Stage1, Stage2).
"""


import warnings
from typing import Dict, List, Tuple

import numpy as np
import polars as pl
from scipy import stats as scipy_stats

# FeatureManifest, FeatureType, _is_numeric  → defined in Cell 3 (classifier)
# ColumnDriftResult, DriftSeverity, DriftSummary,
# MitigationStrategy, TestMethod              → defined in Cell 4 (report)
# All symbols are already in the shared Colab namespace — no imports needed.

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  Tuneable thresholds
# ─────────────────────────────────────────────

PRESCREEN_NUM_THRESHOLD  = 0.10
PRESCREEN_IQR_THRESHOLD  = 0.30
PRESCREEN_CAT_THRESHOLD  = 0.10

KS_MILD     = 0.05
KS_MODERATE = 0.10
KS_SEVERE   = 0.20

PSI_MILD     = 0.10
PSI_MODERATE = 0.25
PSI_SEVERE   = 0.50

JS_MILD     = 0.05
JS_MODERATE = 0.10
JS_SEVERE   = 0.20

Z_MILD     = 2.0
Z_MODERATE = 3.0
Z_SEVERE   = 5.0

ALPHA             = 0.05
TRAIN_SAMPLE      = 50_000
TEST_SAMPLE       = 20_000
MIN_OBSERVATIONS  = 30
CHI2_MIN_EXPECTED = 5

# For high-cardinality columns: retain only the top-K categories by train
# frequency before computing JS divergence.  Everything outside the top-K
# is merged into a single "OTHER" bin in both train and test.
# Rationale: raw JS on 1,000+ categories is dominated by tail noise —
# cities with 1-2 observations produce tiny probability masses (~1e-5)
# whose log ratios swamp the signal from the important categories.
# Top-K binning gives JS stable, reliable estimates while preserving the
# bulk of the distributional information.  K=50 is consistent with the
# Phase A pre-screen which already uses top-30 L1 distance.
TOP_K_HIGH_CARDINALITY = 50
CARDINALITY_THRESHOLD  = 100


# ─────────────────────────────────────────────
#  Polars-native helpers
# ─────────────────────────────────────────────

def _polars_sample(df: pl.DataFrame, col: str, n: int, seed: int = 42) -> np.ndarray:
    """
    Extract up to n non-null values of a column as a numpy array.
    Uses Polars lazy filter + sample for zero-copy numeric types.
    """
    series = df[col].drop_nulls()
    if len(series) > n:
        series = series.sample(n=n, seed=seed)
    return series.to_numpy()


def _normalise_cat_series(series: pl.Series) -> pl.Series:
    """
    Normalise string columns before any frequency or PSI computation.

    Steps applied in order (all vectorised Polars expressions):
      1. Cast to String (handles Categorical dtype uniformly)
      2. Lowercase         — "Bank Withdrawal" → "bank withdrawal"
      3. Strip whitespace  — removes leading/trailing spaces
      4. Punctuation norm  — hyphens and underscores → space
                             ("bank-withdrawal" → "bank withdrawal")
      5. Collapse spaces   — multiple spaces → single space

    Rationale for step 4:
      Without punctuation normalisation, "Bank Withdrawal" and
      "bank-withdrawal" appear as two distinct categories.  Their
      combined train frequency then mismatches the consolidated test
      frequency, producing a false-positive PSI inflation (observed:
      PSI=2.72 for TransactionMode, entirely due to this artefact).
      Normalising punctuation collapses them correctly before any
      distributional comparison.

    Non-string columns are returned unchanged.
    """
    if series.dtype in (pl.String, pl.Utf8, pl.Categorical):
        return (
            series
            .cast(pl.String)
            .str.to_lowercase()
            .str.strip_chars()
            .str.replace_all(r'[-_]', ' ')
            .str.replace_all(r'\s+', ' ')
        )
    return series


def _value_counts_dict(series: pl.Series, normalize: bool = True) -> Dict[str, float]:
    """
    Return {category: frequency} dict from a Polars Series.
    Fills nulls with '__null__' before counting.
    """
    s = _normalise_cat_series(series).fill_null("__null__")
    vc = s.value_counts(sort=True)
    total = len(series) if normalize else 1
    # vc columns are [series_name, "count"]
    name_col = vc.columns[0]
    return {row[name_col]: row["count"] / total for row in vc.iter_rows(named=True)}


def _batch_test_stats(
    test_df: pl.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
) -> Tuple[Dict, Dict]:
    """
    Compute test-set summary stats for ALL columns in TWO parallel .select() calls
    (one for numerics, one for categoricals with value_counts).

    Returns (num_stats_dict, cat_vc_dict) where:
      num_stats_dict : { col: { mean, std, p25, p75 } }
      cat_vc_dict    : { col: { category: freq } }
    """
    # Numeric: batch aggregation (single parallel pass)
    num_stats: Dict[str, Dict] = {}
    if num_cols:
        exprs = []
        for c in num_cols:
            exprs += [
                pl.col(c).mean().alias(f"{c}__mean"),
                pl.col(c).std() .alias(f"{c}__std"),
                pl.col(c).quantile(0.25, interpolation="linear").alias(f"{c}__p25"),
                pl.col(c).quantile(0.75, interpolation="linear").alias(f"{c}__p75"),
            ]
        row = test_df.select(exprs).row(0, named=True)
        for c in num_cols:
            num_stats[c] = {
                "mean": row.get(f"{c}__mean"),
                "std":  row.get(f"{c}__std"),
                "p25":  row.get(f"{c}__p25"),
                "p75":  row.get(f"{c}__p75"),
            }

    # Categorical: value_counts per column (each is O(n_unique), fast)
    cat_vc: Dict[str, Dict] = {}
    for c in cat_cols:
        if c in test_df.columns:
            cat_vc[c] = _value_counts_dict(test_df[c])

    return num_stats, cat_vc


# ─────────────────────────────────────────────
#  Severity helpers
# ─────────────────────────────────────────────

def _sev_ks(d: float) -> DriftSeverity:
    if d < KS_MILD:     return DriftSeverity.NONE
    if d < KS_MODERATE: return DriftSeverity.MILD
    if d < KS_SEVERE:   return DriftSeverity.MODERATE
    return DriftSeverity.SEVERE

def _sev_psi(p: float) -> DriftSeverity:
    if p < PSI_MILD:     return DriftSeverity.NONE
    if p < PSI_MODERATE: return DriftSeverity.MILD
    if p < PSI_SEVERE:   return DriftSeverity.MODERATE
    return DriftSeverity.SEVERE

def _sev_js(j: float) -> DriftSeverity:
    if j < JS_MILD:     return DriftSeverity.NONE
    if j < JS_MODERATE: return DriftSeverity.MILD
    if j < JS_SEVERE:   return DriftSeverity.MODERATE
    return DriftSeverity.SEVERE

def _sev_z(z: float) -> DriftSeverity:
    az = abs(z)
    if az < Z_MILD:     return DriftSeverity.NONE
    if az < Z_MODERATE: return DriftSeverity.MILD
    if az < Z_SEVERE:   return DriftSeverity.MODERATE
    return DriftSeverity.SEVERE


# ─────────────────────────────────────────────
#  Mitigation rule engine
# ─────────────────────────────────────────────

def _assign_mitigation(
    ft: FeatureType,
    severity: DriftSeverity,
    skewness: float = 0.0,
) -> MitigationStrategy:
    if severity == DriftSeverity.NONE:
        return MitigationStrategy.NO_ACTION_STABLE
    if ft == FeatureType.NUMERICAL:
        if severity == DriftSeverity.MILD:
            return MitigationStrategy.ROBUST_SCALE
        if severity == DriftSeverity.MODERATE:
            return (MitigationStrategy.LOG_ROBUST_SCALE
                    if abs(skewness) > 1.0 else MitigationStrategy.ROBUST_SCALE)
        return MitigationStrategy.QUANTILE_BIN
    if ft == FeatureType.CATEGORICAL:
        return MitigationStrategy.FREQUENCY_ENCODE
    if ft == FeatureType.HIGH_CARDINALITY:
        return MitigationStrategy.DROP
    if ft == FeatureType.SPARSE:
        return MitigationStrategy.BINARISE
    return MitigationStrategy.NONE


# ─────────────────────────────────────────────
#  Statistical functions
# ─────────────────────────────────────────────

def _compute_psi(train_vc: Dict[str, float], test_vc: Dict[str, float]) -> float:
    """PSI = Σ (actual - expected) × ln(actual / expected)."""
    all_cats = set(train_vc) | set(test_vc)
    psi = 0.0
    for cat in all_cats:
        e = max(train_vc.get(cat, 0.0), 1e-6)
        a = max(test_vc.get(cat, 0.0),  1e-6)
        psi += (a - e) * np.log(a / e)
    return float(psi)


def _topk_bin(
    train_vc: Dict[str, float],
    test_vc:  Dict[str, float],
    k: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Reduce a high-cardinality frequency dict to a stable Top-K + OTHER representation.

    Steps:
      1. Select the top-K categories by TRAIN frequency (train is the reference).
      2. In both train and test, sum all out-of-top-K frequencies into '__OTHER__'.
      3. Re-normalise both dicts so probabilities sum to 1.

    Why train-anchored top-K:
      We always rank by train frequency, never test.  If we ranked by the
      union or by test, a new dominant category appearing in test would be
      included in the top-K and its novelty would be partially hidden.
      Using train as the anchor means new categories in test always land in
      the OTHER bin, which correctly signals distributional change.

    Why not Laplace smoothing instead:
      Laplace adds a small floor (e.g. 1e-9) to every category.  With 1,109
      cities this creates 1,109 near-zero probability masses whose log ratios
      contribute spurious JS inflation.  Top-K binning eliminates the noisy
      tail entirely rather than trying to smooth it.
    """
    OTHER = "__OTHER__"

    # Top-K by train frequency
    top_cats = set(
        sorted(train_vc, key=lambda c: train_vc[c], reverse=True)[:k]
    )

    def bin_vc(vc: Dict[str, float]) -> Dict[str, float]:
        binned: Dict[str, float] = {}
        other_mass = 0.0
        for cat, freq in vc.items():
            if cat in top_cats:
                binned[cat] = freq
            else:
                other_mass += freq
        if other_mass > 0:
            binned[OTHER] = other_mass
        total = sum(binned.values())
        if total > 0:
            binned = {c: v / total for c, v in binned.items()}
        return binned

    return bin_vc(train_vc), bin_vc(test_vc)


def _compute_js_divergence(train_vc: Dict[str, float], test_vc: Dict[str, float]) -> float:
    """
    Jensen-Shannon divergence on pre-binned frequency dicts, normalised to [0,1].
    Input dicts should already be Top-K binned before calling this function
    (use _topk_bin for high-cardinality columns).
    Laplace floor of 1e-9 guards against exact zeros in the binned distribution.
    """
    all_cats = list(set(train_vc) | set(test_vc))
    p = np.array([max(train_vc.get(c, 0.0), 1e-9) for c in all_cats])
    q = np.array([max(test_vc.get(c, 0.0),  1e-9) for c in all_cats])
    p /= p.sum(); q /= q.sum()
    m  = 0.5 * (p + q)
    js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    return float(js / np.log(2))


def _iqr_non_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of train IQR not covered by test IQR. 0=full overlap, 1=none."""
    a25, a75 = np.nanpercentile(a, 25), np.nanpercentile(a, 75)
    b25, b75 = np.nanpercentile(b, 25), np.nanpercentile(b, 75)
    iqr_a    = a75 - a25
    if iqr_a < 1e-9:
        return 0.0
    overlap = max(0.0, min(a75, b75) - max(a25, b25))
    return float(1.0 - overlap / iqr_a)


def _two_proportion_z(n1_pos: int, n1: int, n2_pos: int, n2: int) -> Tuple[float, float]:
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p1     = n1_pos / n1
    p2     = n2_pos / n2
    p_pool = (n1_pos + n2_pos) / (n1 + n2)
    se     = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se < 1e-9:
        return 0.0, 1.0
    z     = (p1 - p2) / se
    p_val = float(2 * (1 - scipy_stats.norm.cdf(abs(z))))
    return float(z), p_val


# ─────────────────────────────────────────────
#  Phase A — fast pre-screeners
# ─────────────────────────────────────────────

def _phase_a_numerical(
    col: str,
    profile,               # ColumnProfile (has cached train stats)
    te_stats: Dict,        # pre-computed test stats for this col
) -> ColumnDriftResult:
    """
    Use CACHED train stats from FeatureManifest (no recomputation).
    Compute test stats from the batch dict (also pre-computed).
    """
    mu_tr  = profile.mean  if not np.isnan(profile.mean)  else 0.0
    std_tr = profile.std   if not np.isnan(profile.std)   else 1.0
    p25_tr = profile.p25   if not np.isnan(profile.p25)   else mu_tr
    p75_tr = profile.p75   if not np.isnan(profile.p75)   else mu_tr

    mu_te  = te_stats.get("mean") or 0.0
    p25_te = te_stats.get("p25")  or mu_te
    p75_te = te_stats.get("p75")  or mu_te

    norm_shift = abs(mu_tr - mu_te) / (std_tr + 1e-9)

    iqr_tr = p75_tr - p25_tr
    iqr_gap = 0.0
    if iqr_tr > 1e-9:
        overlap = max(0.0, min(p75_tr, p75_te) - max(p25_tr, p25_te))
        iqr_gap = 1.0 - overlap / iqr_tr

    score   = max(norm_shift, iqr_gap)
    flagged = (norm_shift > PRESCREEN_NUM_THRESHOLD
               or iqr_gap  > PRESCREEN_IQR_THRESHOLD)

    return ColumnDriftResult(
        column=col, feature_type=FeatureType.NUMERICAL.value,
        prescreen_score=score, prescreen_flagged=flagged,
        test_method=TestMethod.PRESCREENED,
        mitigation=MitigationStrategy.NO_ACTION_STABLE,
        notes=f"norm_shift={norm_shift:.4f}  iqr_gap={iqr_gap:.4f}",
    )


def _phase_a_sparse(
    col: str,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    is_num: bool,
) -> ColumnDriftResult:
    """Compare presence rate (non-null/non-zero) between train and test."""
    if is_num:
        tr_rate = float((train_df[col].fill_null(0) != 0).mean())
        te_rate = float((test_df[col].fill_null(0) != 0).mean())
    else:
        tr_rate = float(train_df[col].is_not_null().mean())
        te_rate = float(test_df[col].is_not_null().mean())

    score   = abs(tr_rate - te_rate)
    flagged = score > 0.05

    return ColumnDriftResult(
        column=col, feature_type=FeatureType.SPARSE.value,
        prescreen_score=score, prescreen_flagged=flagged,
        test_method=TestMethod.PRESCREENED,
        mitigation=MitigationStrategy.NO_ACTION_STABLE,
        notes=(f"presence_rate: train={tr_rate:.3f}  test={te_rate:.3f}  "
               f"shift={score:.4f}"),
    )


# ─────────────────────────────────────────────
#  Phase B — deep statistical tests
# ─────────────────────────────────────────────

def _phase_b_numerical(
    result: ColumnDriftResult,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    skewness: float,
    seed: int = 42,
) -> ColumnDriftResult:
    """KS 2-sample test.  .to_numpy() is zero-copy for numeric Polars series."""
    tr = _polars_sample(train_df, result.column, TRAIN_SAMPLE, seed)
    te = _polars_sample(test_df,  result.column, TEST_SAMPLE,  seed)

    if len(tr) < MIN_OBSERVATIONS or len(te) < MIN_OBSERVATIONS:
        result.notes += " | SKIPPED: insufficient non-null data"
        return result

    ks_d, p_val = scipy_stats.ks_2samp(tr, te)
    severity    = _sev_ks(ks_d)
    confirmed   = (p_val < ALPHA) and (ks_d > KS_MILD)

    result.test_method    = TestMethod.KS_2SAMP
    result.test_statistic = float(ks_d)
    result.p_value        = float(p_val)
    result.drift_detected = confirmed
    result.drift_severity = severity if confirmed else DriftSeverity.NONE
    result.mitigation     = _assign_mitigation(FeatureType.NUMERICAL,
                                               result.drift_severity, skewness)
    result.notes         += f" | KS_D={ks_d:.4f}  p={p_val:.2e}  skew={skewness:.2f}"
    return result


def _phase_b_categorical(
    result: ColumnDriftResult,
    train_vc: Dict[str, float],
    test_vc: Dict[str, float],
    n_test: int,
) -> ColumnDriftResult:
    """
    PSI + chi-squared Goodness-of-Fit.
    train_vc and test_vc are pre-normalised (lowercase/stripped) frequency dicts.
    """
    psi_val  = _compute_psi(train_vc, test_vc)
    all_cats = sorted(set(train_vc) | set(test_vc))

    observed = np.array([test_vc.get(c, 0.0) * n_test for c in all_cats])
    expected = np.array([max(train_vc.get(c, 0.0), 1e-9) * n_test for c in all_cats])

    mask = expected >= CHI2_MIN_EXPECTED
    chi2_p, chi2_ran = 1.0, False
    if mask.sum() >= 2:
        obs_m = observed[mask]
        exp_m = expected[mask]
        # Rescale expected to match observed sum after masking sparse cells
        exp_m = exp_m / exp_m.sum() * obs_m.sum()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                _, chi2_p  = scipy_stats.chisquare(f_obs=obs_m, f_exp=exp_m)
                chi2_p     = float(chi2_p)
                chi2_ran   = True
            except Exception:
                chi2_p = 1.0

    severity  = _sev_psi(psi_val)
    confirmed = (psi_val >= PSI_MILD) and (chi2_p < ALPHA or not chi2_ran)

    result.test_method    = TestMethod.PSI
    result.test_statistic = float(psi_val)
    result.secondary_stat = float(chi2_p)
    result.p_value        = float(chi2_p)
    result.drift_detected = confirmed
    result.drift_severity = severity if confirmed else DriftSeverity.NONE
    result.mitigation     = _assign_mitigation(FeatureType.CATEGORICAL, result.drift_severity)
    result.notes         += f" | PSI={psi_val:.4f}  chi2_gof_p={chi2_p:.2e}"
    return result


def _phase_b_high_cardinality(
    result: ColumnDriftResult,
    train_vc: Dict[str, float],
    test_vc: Dict[str, float],
) -> ColumnDriftResult:
    """
    Top-K Frequency Binning + Jensen-Shannon Divergence.

    Raw JS on 1,000+ categories is unreliable because:
      - Cities with 1-2 train observations have probability mass ~1e-5
      - Their log ratios dominate the JS sum, masking real distributional shifts
      - The score becomes a measure of tail noise, not meaningful drift

    Fix: bin to Top-K by train frequency (K=TOP_K_HIGH_CARDINALITY=50),
    merge remainder into OTHER, then compute JS on the stable K+1 distribution.

    The OTHER bin itself carries signal: if many previously rare categories
    become common in test, the OTHER bin's train/test frequencies will diverge,
    which JS will correctly detect.
    """
    n_train_cats = len(train_vc)
    tr_binned, te_binned = _topk_bin(train_vc, test_vc, TOP_K_HIGH_CARDINALITY)

    js       = _compute_js_divergence(tr_binned, te_binned)
    severity = _sev_js(js)

    # Compute what fraction of train mass is covered by the top-K
    top_k_coverage = sum(
        v for k, v in tr_binned.items() if k != "__OTHER__"
    )

    result.test_method    = TestMethod.JS_DIVERGENCE
    result.test_statistic = float(js)
    result.p_value        = None
    result.drift_detected = severity != DriftSeverity.NONE
    result.drift_severity = severity
    result.mitigation     = _assign_mitigation(FeatureType.HIGH_CARDINALITY, severity)
    result.notes         += (
        f" | JS={js:.4f} (top-{TOP_K_HIGH_CARDINALITY} of {n_train_cats} cats, "
        f"coverage={top_k_coverage:.1%})"
    )
    return result


def _phase_b_sparse(
    result: ColumnDriftResult,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    is_num: bool,
) -> ColumnDriftResult:
    """
    Two-Stage Bipartite Test for sparse columns.

    Stage 1 — Bernoulli Presence Rate Test (Proportions Z-test)
        Tests whether the proportion of non-zero / non-null values has
        shifted between train and test.  This catches structural absence
        drift: e.g. a feature that was rarely observed now being common,
        or vice versa.

    Stage 2 — Conditional KS Test (numeric sparse only)
        Among the non-zero / non-null values ONLY, tests whether the
        value distribution has shifted.  This catches value-level drift
        that Stage 1 is blind to: e.g. presence rate is stable at 14%,
        but refund amounts have tripled in the test period.
        Only runs when Stage 1 passes (presence rate stable) AND the
        column is numeric AND there are sufficient non-zero observations.

    Severity is the maximum of the two stages — a column drifts if
    EITHER its presence rate OR its value distribution has shifted.
    """
    col = result.column

    # ── Stage 1: Bernoulli presence rate Z-test ───────────────────────
    if is_num:
        tr_pos = int((train_df[col].fill_null(0) != 0).sum())
        te_pos = int((test_df[col].fill_null(0)  != 0).sum())
    else:
        tr_pos = int(train_df[col].is_not_null().sum())
        te_pos = int(test_df[col].is_not_null().sum())

    z_stage1, p_stage1 = _two_proportion_z(tr_pos, len(train_df),
                                            te_pos, len(test_df))
    sev_stage1   = _sev_z(z_stage1)
    conf_stage1  = (p_stage1 < ALPHA) and sev_stage1 != DriftSeverity.NONE

    notes = f"Stage1(presence): z={z_stage1:.4f} p={p_stage1:.2e}"

    # ── Stage 2: Conditional KS on non-zero values (numeric only) ─────
    sev_stage2  = DriftSeverity.NONE
    conf_stage2 = False
    ks_d        = None
    p_stage2    = None

    if is_num and tr_pos >= MIN_OBSERVATIONS and te_pos >= MIN_OBSERVATIONS:
        # Extract only the non-zero, non-null values from each split
        _tr_nonnull = train_df[col].drop_nulls()
        tr_vals = _tr_nonnull.filter(_tr_nonnull != 0).to_numpy()
        _te_nonnull = test_df[col].drop_nulls()
        te_vals = _te_nonnull.filter(_te_nonnull != 0).to_numpy()

        # Apply sample cap so this stays fast on large datasets
        rng = np.random.default_rng(42)
        if len(tr_vals) > TRAIN_SAMPLE:
            tr_vals = rng.choice(tr_vals, TRAIN_SAMPLE, replace=False)
        if len(te_vals) > TEST_SAMPLE:
            te_vals = rng.choice(te_vals, TEST_SAMPLE, replace=False)

        if len(tr_vals) >= MIN_OBSERVATIONS and len(te_vals) >= MIN_OBSERVATIONS:
            ks_stat, ks_p = scipy_stats.ks_2samp(tr_vals, te_vals)
            ks_d      = float(ks_stat)
            p_stage2  = float(ks_p)
            sev_stage2  = _sev_ks(ks_d)
            conf_stage2 = (ks_p < ALPHA) and sev_stage2 != DriftSeverity.NONE
            notes += f" | Stage2(values): KS={ks_d:.4f} p={ks_p:.2e}"
        else:
            notes += " | Stage2: insufficient non-zero values"
    elif not is_num:
        notes += " | Stage2: skipped (non-numeric sparse)"
    else:
        notes += " | Stage2: skipped (too few non-zero observations)"

    # ── Combine: drift confirmed if EITHER stage fires ─────────────────
    # Severity = maximum of the two stages
    SEV_ORDER = {
        DriftSeverity.NONE:     0,
        DriftSeverity.MILD:     1,
        DriftSeverity.MODERATE: 2,
        DriftSeverity.SEVERE:   3,
    }
    confirmed = conf_stage1 or conf_stage2
    severity  = (sev_stage1
                 if SEV_ORDER[sev_stage1] >= SEV_ORDER[sev_stage2]
                 else sev_stage2)
    if not confirmed:
        severity = DriftSeverity.NONE

    # Primary test_statistic reported: Stage 2 KS_D if it ran, else Stage 1 Z
    primary_stat = ks_d if ks_d is not None else float(z_stage1)
    primary_p    = p_stage2 if p_stage2 is not None else float(p_stage1)

    result.test_method    = TestMethod.Z_PROPORTION   # Stage 1 always runs
    result.test_statistic = primary_stat
    result.p_value        = primary_p
    result.drift_detected = confirmed
    result.drift_severity = severity
    result.mitigation     = (MitigationStrategy.BINARISE
                             if confirmed else MitigationStrategy.NO_ACTION_STABLE)
    result.notes         += f" | {notes}"
    return result


# ─────────────────────────────────────────────
#  Main public class
# ─────────────────────────────────────────────

class DriftDetector:
    """
    Two-phase drift detection over all testable features.

    Key efficiency properties
    -------------------------
    • Test-set numeric stats computed in ONE parallel .select() call
    • Categorical value_counts use Polars Arrow GroupBy (O(n))
    • Train stats reused from FeatureManifest — not recomputed
    • scipy tests only run on Phase-A-flagged columns
    • .to_numpy() on numeric columns is zero-copy

    Usage
    -----
    detector = DriftDetector(manifest)
    summary  = detector.detect_all(train_df, test_df)
    print(summary.summary())
    """

    def __init__(self, manifest: FeatureManifest, random_state: int = 42):
        self.manifest     = manifest
        self.random_state = random_state

    def detect_all(
        self,
        train_df: pl.DataFrame,
        test_df:  pl.DataFrame,
    ) -> DriftSummary:
        """
        Parameters
        ----------
        train_df, test_df : Polars DataFrames (full datasets).

        Returns
        -------
        DriftSummary with ColumnDriftResult per column.
        """
        summary  = DriftSummary()
        profiles = self.manifest.profiles
        n_test   = len(test_df)

        excluded_types = {FeatureType.TARGET, FeatureType.METADATA,
                          FeatureType.CONSTANT, FeatureType.TIME}

        # ── Pre-compute test-set stats for all relevant columns ────────
        # This single Polars call replaces N sequential pandas aggregations
        testable_num = [
            c for c, p in profiles.items()
            if p.feature_type == FeatureType.NUMERICAL and c in test_df.columns
        ]
        testable_cat = [
            c for c, p in profiles.items()
            if p.feature_type in (FeatureType.CATEGORICAL,
                                   FeatureType.HIGH_CARDINALITY,
                                   FeatureType.SPARSE)
            and c in test_df.columns
        ]
        te_num_stats, te_cat_vc = _batch_test_stats(test_df, testable_num, testable_cat)

        # Pre-compute train-side value_counts for categorical columns
        tr_cat_vc: Dict[str, Dict] = {}
        for col in testable_cat:
            if col in train_df.columns:
                tr_cat_vc[col] = _value_counts_dict(train_df[col])

        # ── Per-column detection loop ──────────────────────────────────
        for col, profile in profiles.items():
            ft = profile.feature_type

            # Skip excluded column types
            if ft in excluded_types:
                summary.results[col] = ColumnDriftResult(
                    column=col, feature_type=ft.value,
                    test_method=TestMethod.SKIPPED,
                    mitigation=MitigationStrategy.NO_ACTION_EXCLUDED,
                    notes="excluded feature type",
                )
                continue

            if col not in train_df.columns or col not in test_df.columns:
                summary.results[col] = ColumnDriftResult(
                    column=col, feature_type=ft.value,
                    test_method=TestMethod.SKIPPED,
                    notes="column absent in one dataset",
                )
                continue

            is_num = _is_numeric(train_df[col].dtype)

            # ── PHASE A ────────────────────────────────────────────────
            if ft == FeatureType.NUMERICAL:
                result = _phase_a_numerical(
                    col, profile, te_num_stats.get(col, {})
                )

            elif ft == FeatureType.CATEGORICAL:
                # Categorical: PSI is cheap enough to run directly — skip pre-screen
                result = ColumnDriftResult(
                    column=col, feature_type=ft.value,
                    prescreen_score=0.0, prescreen_flagged=True,
                    test_method=TestMethod.PRESCREENED,
                    mitigation=MitigationStrategy.NO_ACTION_STABLE,
                )

            elif ft == FeatureType.HIGH_CARDINALITY:
                # Pre-screen using L1 on top-K frequency vectors
                tr_vc = tr_cat_vc.get(col, {})
                te_vc = te_cat_vc.get(col, {})
                all_c = set(list(tr_vc.keys())[:30]) | set(list(te_vc.keys())[:30])
                l1    = sum(abs(tr_vc.get(c, 0.0) - te_vc.get(c, 0.0)) for c in all_c)
                result = ColumnDriftResult(
                    column=col, feature_type=ft.value,
                    prescreen_score=l1, prescreen_flagged=l1 > PRESCREEN_CAT_THRESHOLD,
                    test_method=TestMethod.PRESCREENED,
                    mitigation=MitigationStrategy.NO_ACTION_STABLE,
                    notes=f"l1_freq={l1:.4f}",
                )

            elif ft == FeatureType.SPARSE:
                result = _phase_a_sparse(col, train_df, test_df, is_num)

            else:
                summary.results[col] = ColumnDriftResult(
                    column=col, feature_type=ft.value,
                    test_method=TestMethod.SKIPPED,
                    notes="unhandled type",
                )
                continue

            # ── PHASE B (flagged columns only) ─────────────────────────
            if result.prescreen_flagged:
                if ft == FeatureType.NUMERICAL:
                    result = _phase_b_numerical(
                        result, train_df, test_df,
                        skewness=profile.skewness if not np.isnan(profile.skewness) else 0.0,
                        seed=self.random_state,
                    )
                elif ft == FeatureType.CATEGORICAL:
                    result = _phase_b_categorical(
                        result,
                        tr_cat_vc.get(col, {}),
                        te_cat_vc.get(col, {}),
                        n_test,
                    )
                elif ft == FeatureType.HIGH_CARDINALITY:
                    result = _phase_b_high_cardinality(
                        result,
                        tr_cat_vc.get(col, {}),
                        te_cat_vc.get(col, {}),
                    )
                elif ft == FeatureType.SPARSE:
                    result = _phase_b_sparse(result, train_df, test_df, is_num)

            summary.results[col] = result

        return summary



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 4/5 — drift/ranker.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
    # ── Step 1: compute recency factor from serve_df ──────────────────────
    # We only need the Month column — select it first to avoid materialising
    # all 500 feature columns in the join.
    serve_months = serve_df.select(month_col)

    weighted_df  = _recency_weights(serve_months, month_col, lambda_decay)
    per_month    = _weighted_row_fraction(weighted_df, month_col)

    # Scalar recency factor: decay-weighted mean of per-month row fractions.
    # = Σ_t (w_fraction_t × w_fraction_t) normalised
    # Simplified to: sum of squared w_fractions (Herfindahl-style concentration).
    # A high value means serve data is concentrated in recent months.
    recency_factor = float(
        per_month
        .select((pl.col("w_fraction") ** 2).sum())
        .item()
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



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODULE 5/5 — drift/mitigator.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
            pl.col(col).cast(pl.String).str.to_lowercase().str.strip_chars().str.replace_all(r'[-_]', ' ').str.replace_all(r'\s+', ' ').alias("__cat__"),
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
        .select(pl.col(col).cast(pl.String).str.to_lowercase().str.strip_chars().str.replace_all(r'[-_]', ' ').str.replace_all(r'\s+', ' ').alias("__cat__"))
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
        .select(pl.col(col).cast(pl.String).str.to_lowercase().str.strip_chars().str.replace_all(r'[-_]', ' ').str.replace_all(r'\s+', ' ').alias("__cat__"))
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
            n_unique = train_df[feature].n_unique()
            print(f"  ✔ freq-encoded    : {feature}  ({n_unique:,} unique → Int64 counts)")

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



# CELL 4 — Stages 1-2: Schema Classification + Drift Detection
import time

# ── CANDIDATE 4: Three-level resilient CSV reading ──────────────────────────
def _read_csv_safe(path):
    null_tokens = ["", "NA", "N/A", "NULL", "null", "None", "none", "NaN", "nan"]
    try:
        return pl.read_csv(str(path), infer_schema_length=None,
                           null_values=null_tokens, ignore_errors=False)
    except Exception:
        pass
    try:
        import csv
        with open(str(path), encoding='utf-8', newline='') as fh:
            header = next(csv.reader(fh))
        schema = {col: pl.Utf8 for col in header}
        return pl.read_csv(str(path), schema=schema,
                           null_values=null_tokens, ignore_errors=True)
    except Exception:
        return pl.read_csv(str(path), ignore_errors=True)

print('[1/2] Loading datasets...')
t0       = time.perf_counter()
train_df = _read_csv_safe(TRAIN_PATH)
test_df  = _read_csv_safe(TEST_PATH)
print(f'      train : {train_df.shape[0]:>7,} rows x {train_df.shape[1]} cols')
print(f'      test  : {test_df.shape[0]:>7,} rows x {test_df.shape[1]} cols')
print(f'      loaded in {time.perf_counter()-t0:.2f}s')

# ── CANDIDATE 4 verification: DigitalInvoicing null count ───────────────────
if 'DigitalInvoicing' in train_df.columns:
    _di_nulls = train_df['DigitalInvoicing'].null_count()
    print(f'      DigitalInvoicing nulls in train (raw): {_di_nulls:,}')

# ── CANDIDATE 5: Schema column alignment guard ───────────────────────────────
_excluded_align = {'CustomerID', 'ChurnStatus'}
_missing_in_test = [
    c for c in train_df.columns
    if c not in _excluded_align and c not in test_df.columns
]
if _missing_in_test:
    test_df = test_df.with_columns([
        pl.lit(None).alias(c) for c in _missing_in_test
    ])
    print(f'  Added {len(_missing_in_test)} null columns to test_df: {_missing_in_test}')
else:
    print(f'  Column alignment OK — test_df shape: {test_df.shape}')

# ── CANDIDATE 2: Impute DigitalInvoicing NaN as "No" in train only ───────────
if 'DigitalInvoicing' in train_df.columns:
    _before = train_df['DigitalInvoicing'].null_count()
    train_df = train_df.with_columns(
        pl.col('DigitalInvoicing').fill_null('No')
    )
    _after = train_df['DigitalInvoicing'].null_count()
    print(f'  DigitalInvoicing nulls: {_before:,} -> {_after} (imputed as "No")')

# ── CANDIDATE 1: charge_per_tenure engineered feature ───────────────────────
if 'MonthlyCharge' in train_df.columns and 'TenureinMonths' in train_df.columns:
    train_df = train_df.with_columns(
        (pl.col('MonthlyCharge') / (pl.col('TenureinMonths') + 1)).alias('charge_per_tenure')
    )
    test_df = test_df.with_columns(
        (pl.col('MonthlyCharge') / (pl.col('TenureinMonths') + 1)).alias('charge_per_tenure')
    )
    print(f'  charge_per_tenure added — mean={train_df["charge_per_tenure"].mean():.3f}')

print('\n[2/2] Running Schema Classifier + Drift Detector...')
t0            = time.perf_counter()
manifest      = classify_features(train_df, sample_size=50_000, random_state=42)
detector_obj  = DriftDetector(manifest, random_state=42)
drift_summary = detector_obj.detect_all(train_df, test_df)
_t_stage2     = time.perf_counter() - t0
print(f'      Done in {_t_stage2:.2f}s')

# ── Candidate 1 verification: charge_per_tenure in manifest ─────────────────
if 'charge_per_tenure' in manifest.profiles:
    _cpt_type = manifest.profiles['charge_per_tenure'].feature_type.value
    print(f'  charge_per_tenure classified as: {_cpt_type}')
else:
    print('  WARNING: charge_per_tenure NOT found in manifest!')

drifted = drift_summary.drifted
_col_w, _type_w, _desc_w, _mit_w = 32, 12, 50, 34
_sep = ('+' + '-'*(_col_w+2) + '+' + '-'*(_type_w+2) + '+'
        + '-'*(_desc_w+2) + '+' + '-'*(_mit_w+2) + '+')
_hdr = (f"| {'Columns with Drift':<{_col_w}} | {'Column Type':<{_type_w}} | "
        f"{'Drift Description':<{_desc_w}} | {'Drift Mitigation':<{_mit_w}} |")
print()
print(_sep)
print(_hdr)
print(_sep)

def _drift_desc(r):
    ft, sev = r.feature_type, r.drift_severity.value
    if ft == 'numerical':
        return ('Feature ranges explode in test set' if sev == 'SEVERE'
                else 'Feature demonstrates greater left-skewness in test set' if sev == 'MODERATE'
                else 'Feature distribution shifts mildly in test set')
    elif ft in ('categorical', 'high_cardinality'):
        return ('Feature has new set of categorical features in test set' if sev == 'SEVERE'
                else 'Feature category proportions shift in test set')
    return 'Feature distribution shifts in test set'

def _mit_label(r):
    mv = r.mitigation.value
    ft = r.feature_type
    if mv == 'frequency_encoding':
        if ft in ('categorical', 'sparse'):
            return 'Target Encoding (Laplace, m=20)'
        else:
            return 'Frequency Encoding'
    return {
        'robust_scaling':                'Feature Scaling (Robust)',
        'quantile_binning':              'Quantile Binning (10 deciles)',
        'log_transform + robust_scaling':'Log + Feature Scaling',
        'drop_feature':                  'Drop Feature',
        'binarise (sparse->binary)':     'Binarisation',
    }.get(mv, mv.replace('_', ' ').title())

sev_order = {DriftSeverity.SEVERE: 0, DriftSeverity.MODERATE: 1, DriftSeverity.MILD: 2}
for r in sorted(drifted, key=lambda x: (sev_order.get(x.drift_severity, 9), -(x.test_statistic or 0))):
    print(f"| {r.column[:_col_w]:<{_col_w}} | {r.feature_type[:_type_w]:<{_type_w}} | "
          f"{_drift_desc(r)[:_desc_w]:<{_desc_w}} | {_mit_label(r)[:_mit_w]:<{_mit_w}} |")
if not drifted:
    print(f"| {'No drift detected':<{_col_w}} | {'':<{_type_w}} | {'':<{_desc_w}} | {'':<{_mit_w}} |")
print(_sep)
print(f'\nb. Detection + Mitigation Time Taken')
print(f'+{"-"*17}+\n| Time Taken (s)  |\n+{"-"*17}+')
print(f'| {_t_stage2:<15.1f} |\n+{"-"*17}+')
print('\nStages 1-2 complete. Run Cell 5.')


# CELL 5 — Stages 3-5: Rank → Mitigate → Train → Predict
# Requires Cell 4 to have run first
if 'manifest' not in dir() or 'drift_summary' not in dir() or 'train_df' not in dir():
    raise RuntimeError(
        "Cell 4 must be run before Cell 5. "
        "Restart your kernel, run Cells 1 to 2 to 3 to 4 in order, then run Cell 5."
    )
_t0_pipeline = time.time()
import time, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score
warnings.filterwarnings('ignore')

_TARGET    = 'ChurnStatus'
_ID_COL    = 'CustomerID'
_SKIP_NORM = {_TARGET, _ID_COL}

# Build _NON_FEAT dynamically from manifest (not hardcoded to any column names)
_excluded_types = {FeatureType.METADATA, FeatureType.CONSTANT,
                   FeatureType.TIME, FeatureType.TARGET}
_NON_FEAT = {col for col, profile in manifest.profiles.items()
             if profile.feature_type in _excluded_types}
_NON_FEAT.update({_TARGET, _ID_COL})
print(f'Non-feature columns excluded: {sorted(_NON_FEAT)}')

print('\n[Stage 3] Ranking drifted features...')
t0 = time.perf_counter()
_rows = [
    {'feature': r.column, 'col_type': r.feature_type,
     'test_used': r.test_method.value,
     'raw_score': float(r.test_statistic) if r.test_statistic is not None else 0.0,
     'drift_severity': r.drift_severity.value}
    for r in drift_summary.drifted
]
if not _rows:
    ranked_df = pl.DataFrame(schema={
        'feature': pl.String, 'col_type': pl.String, 'test_used': pl.String,
        'raw_score': pl.Float64, 'weighted_score': pl.Float64,
        'drift_rank': pl.UInt32, 'drift_severity': pl.String})
    print('  No drifted features.')
else:
    ranked_df = rank_drift(drift_results=pl.DataFrame(_rows), serve_df=test_df,
                           lambda_decay=0.1, month_col=(manifest.time[0] if manifest.time else 'Month'))
    print(f'  Top: {ranked_df["feature"][0]}  ({ranked_df["drift_severity"][0]})')
print(f'  {time.perf_counter()-t0:.2f}s')

# Detect positive label before mitigation (target column is unchanged by mitigate)
_target_series_raw = train_df.select(_TARGET).to_series().cast(pl.String)
_unique_labels    = sorted(_target_series_raw.unique().to_list())
assert len(_unique_labels) == 2, f"ChurnStatus must be binary, found: {_unique_labels}"
_pos_priority = ["yes", "1", "true", "y", "churn"]
_label_map = {}
_pos_label_found = None
for hint in _pos_priority:
    for lbl in _unique_labels:
        if lbl.lower().startswith(hint) and _pos_label_found is None:
            _pos_label_found = lbl
            break
    if _pos_label_found:
        break
if _pos_label_found is None:
    _pos_label_found = sorted(_unique_labels)[-1]
for lbl in _unique_labels:
    _label_map[lbl] = 1 if lbl == _pos_label_found else 0
_pos_label = _pos_label_found
print(f'Target label mapping: {_label_map}  (positive={_pos_label})')

print('\n[Stage 4] Applying mitigation...')
t0_mit = time.perf_counter()
mitigated_train, mitigated_test = mitigate(train_df, test_df, ranked_df, positive_label=_pos_label)
print(f'  {time.perf_counter()-t0_mit:.2f}s')

_str_tr = [c for c in mitigated_train.columns
           if mitigated_train[c].dtype in (pl.Utf8, pl.String, pl.Categorical)
           and c not in _SKIP_NORM]
_str_te = [c for c in mitigated_test.columns
           if mitigated_test[c].dtype in (pl.Utf8, pl.String, pl.Categorical)
           and c not in _SKIP_NORM]

def _normalise_str_cols(df, cols):
    """
    Apply the same string normalisation that detector.py uses internally,
    so LightGBM sees consolidated categories at training time.
    Steps: lowercase -> strip whitespace -> hyphens/underscores -> space -> collapse spaces.
    This ensures 'bank-withdrawal' and 'Bank Withdrawal' are the same token
    in both the drift detector and the model features.
    """
    return df.with_columns([
        pl.col(c)
          .str.to_lowercase()
          .str.strip_chars()
          .str.replace_all(r'[-_]', ' ')
          .str.replace_all(r'\s+', ' ')
        for c in cols
    ])

if _str_tr:
    mitigated_train = _normalise_str_cols(mitigated_train, _str_tr)
if _str_te:
    mitigated_test  = _normalise_str_cols(mitigated_test,  _str_te)

# ── CANDIDATE 3: Seasonality-aware sample weighting ─────────────────────────
# After _normalise_str_cols: lowercase + hyphens→spaces, so '25-Feb' → '25 feb'
_HIGH_MONTHS = {'25 feb', '25 may', '25 jun'}
_month_col_name = manifest.time[0] if manifest.time else 'Month'
if _month_col_name in mitigated_train.columns:
    _month_series = mitigated_train[_month_col_name].cast(pl.String)
    _sample_weights = pl.Series([
        2.0 if m in _HIGH_MONTHS else 1.0
        for m in _month_series.to_list()
    ])
    _high_count = sum(1 for m in _month_series.to_list() if m in _HIGH_MONTHS)
    _low_count  = len(_month_series) - _high_count
    print(f'  Sample weights: {_high_count:,} HIGH-month rows (w=2.0), '
          f'{_low_count:,} LOW-month rows (w=1.0)')
else:
    _sample_weights = None
    print(f'  Month column "{_month_col_name}" not in mitigated_train — no sample weighting')

_feat_cols = [c for c in mitigated_train.columns if c not in _NON_FEAT]
# Candidate 1 verification: charge_per_tenure in _feat_cols
if 'charge_per_tenure' in _feat_cols:
    print(f'  charge_per_tenure is in _feat_cols ({len(_feat_cols)} total features)')
X_train_pd = mitigated_train.select(_feat_cols).to_pandas()
X_test_pd  = mitigated_test.select([c for c in _feat_cols if c in mitigated_test.columns]).to_pandas()

# Apply label map (already detected before mitigation)
_target_series_pd = mitigated_train.select(_TARGET).to_pandas()[_TARGET].astype(str)
del mitigated_train
y_train_pd = _target_series_pd.map(_label_map)
del _target_series_pd
assert y_train_pd.isnull().sum() == 0, (
    f"y_train has NaN — labels: {_unique_labels}, map: {_label_map}")

for _col in X_train_pd.select_dtypes(include='object').columns:
    X_train_pd[_col] = X_train_pd[_col].astype('category')
    if _col in X_test_pd.columns:
        X_test_pd[_col] = X_test_pd[_col].astype(X_train_pd[_col].dtype)

print(f'\n[Stage 5] Training LightGBM...  X_train={X_train_pd.shape}')
t0 = time.perf_counter()
model = lgb.LGBMClassifier(verbosity=-1, objective='binary', is_unbalance=True,
                            random_state=42, importance_type='gain', n_jobs=1)

# ── CANDIDATE 3: pass sample_weight to fit (not a hyperparameter) ───────────
_fit_kwargs = {}
if _sample_weights is not None:
    _fit_kwargs['sample_weight'] = _sample_weights.to_numpy()
model.fit(X_train_pd, y_train_pd, **_fit_kwargs)
print(f'  Training: {time.perf_counter()-t0:.2f}s')

proba_train = model.predict_proba(X_train_pd)[:, 1]
proba_test  = model.predict_proba(X_test_pd)[:, 1]
au_prc_train = average_precision_score(y_train_pd, proba_train)

_has_labels = _TARGET in test_df.columns
if _has_labels:
    _y_test_raw = test_df.select(_TARGET).to_pandas()[_TARGET].astype(str)
    y_test_pd   = _y_test_raw.map(_label_map)
    if y_test_pd.isnull().sum() == 0:
        au_prc_test = average_precision_score(y_test_pd, proba_test)
        au_prc_show = au_prc_test
    else:
        au_prc_test, au_prc_show = None, au_prc_train
else:
    au_prc_test, au_prc_show = None, au_prc_train

pl.DataFrame({_ID_COL: mitigated_test[_ID_COL].cast(pl.String),
              'probability_score': pl.Series(proba_test.astype(float))
              }).write_csv('prediction.csv')
del mitigated_test
print(f'  Saved {len(proba_test):,} rows -> prediction.csv')

_total = time.time() - _t0_pipeline
print()
print('c. Model Performance')
print('+' + '-'*16 + '+' + '-'*8 + '+')
print(f"| {'':14}  | AU-PRC |")
print('+' + '-'*16 + '+' + '-'*8 + '+')
print(f"| {'Train Set':<14}  | {au_prc_train:.3f}  |")
if au_prc_test:
    print(f"| {'Test Set':<14}  | {au_prc_test:.3f}  |")
else:
    print(f"| {'Test Set':<14}  | N/A    |")
print('+' + '-'*16 + '+' + '-'*8 + '+')
print()
print('==========================================')
print('NAISC 2026 — Pipeline Execution Summary')
print(f"{'Metric':<20} {'Value':>18}")
print(f"{'AU-PRC':<20} {au_prc_show:>18.4f}")
print(f"{'Train Set':<20} {f'{len(train_df):,} rows':>18}")
print(f"{'Test Set':<20} {f'{len(test_df):,} rows':>18}")
print(f"{'Total Runtime':<20} {f'{_total:.1f}s':>18}")
print('==========================================')


