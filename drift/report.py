"""
drift/report.py
===============
DriftReport — the typed output contract of the Drift Detector.

Each ColumnDriftResult captures the test result for a single feature:
  - Was it tested? If not, why?
  - Which statistical test ran?
  - What was the test statistic?
  - What severity was assigned?

DriftSummary aggregates all per-column results with a drifted accessor
used downstream by the ranker and mitigator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class MitigationStrategy(str, Enum):
    NONE                = "none"
    ROBUST_SCALE        = "robust_scaling"
    LOG_ROBUST_SCALE    = "log_transform + robust_scaling"
    QUANTILE_BIN        = "quantile_binning"
    FREQUENCY_ENCODE    = "frequency_encoding"
    BINARISE            = "binarise (sparse->binary)"
    DROP                = "drop_feature"
    NO_ACTION_STABLE    = "no_action (stable)"
    NO_ACTION_EXCLUDED  = "no_action (excluded type)"


class DriftSeverity(str, Enum):
    NONE     = "NONE"
    MILD     = "MILD"        # detectable but minor
    MODERATE = "MODERATE"    # meaningful distributional shift
    SEVERE   = "SEVERE"      # major shift; high mitigation priority


class TestMethod(str, Enum):
    KS_2SAMP      = "Kolmogorov-Smirnov"
    PSI           = "Population Stability Index"
    JS_DIVERGENCE = "Jensen-Shannon Divergence"
    Z_PROPORTION  = "Two-Proportion Z-Test"
    PRESCREENED   = "Pre-Screen Only (fast)"
    SKIPPED       = "Skipped"


@dataclass
class ColumnDriftResult:
    """Drift assessment for a single column."""

    column:       str
    feature_type: str

    # ── Phase A: fast pre-screen ────────────────────────────────────────
    prescreen_score:   float = 0.0    # normalised mean-shift or L1-distance
    prescreen_flagged: bool  = False  # True → sent to Phase B deep testing

    # ── Phase B: statistical test ───────────────────────────────────────
    test_method:    TestMethod      = TestMethod.SKIPPED
    test_statistic: Optional[float] = None   # KS D, PSI value, JS div, etc.

    # ── Decision ────────────────────────────────────────────────────────
    drift_detected: bool               = False
    drift_severity: DriftSeverity      = DriftSeverity.NONE
    mitigation:     MitigationStrategy = MitigationStrategy.NO_ACTION_EXCLUDED
    notes:          str                = ""


@dataclass
class DriftSummary:
    """Aggregated drift assessment across all columns."""
    results: Dict[str, ColumnDriftResult] = field(default_factory=dict)

    @property
    def drifted(self) -> List[ColumnDriftResult]:
        return [r for r in self.results.values() if r.drift_detected]
