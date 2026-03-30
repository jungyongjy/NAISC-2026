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

from __future__ import annotations

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
        return result

    ks_d, p_val = scipy_stats.ks_2samp(tr, te)
    severity    = _sev_ks(ks_d)
    confirmed   = (p_val < ALPHA) and (ks_d > KS_MILD)

    result.test_method    = TestMethod.KS_2SAMP
    result.test_statistic = float(ks_d)
    result.drift_detected = confirmed
    result.drift_severity = severity if confirmed else DriftSeverity.NONE
    result.mitigation     = _assign_mitigation(FeatureType.NUMERICAL,
                                               result.drift_severity, skewness)
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
    result.drift_detected = confirmed
    result.drift_severity = severity if confirmed else DriftSeverity.NONE
    result.mitigation     = _assign_mitigation(FeatureType.CATEGORICAL, result.drift_severity)
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
    result.drift_detected = severity != DriftSeverity.NONE
    result.drift_severity = severity
    result.mitigation     = _assign_mitigation(FeatureType.HIGH_CARDINALITY, severity)
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
    result.drift_detected = confirmed
    result.drift_severity = severity
    result.mitigation     = (MitigationStrategy.BINARISE
                             if confirmed else MitigationStrategy.NO_ACTION_STABLE)
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
                                   FeatureType.HIGH_CARDINALITY)
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


