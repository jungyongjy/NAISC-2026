"""
dashboard.py — NAISC 2026 Green Beans Drift Dashboard
=======================================================
Run with:  streamlit run dashboard.py

Requirements:
    pip install streamlit plotly pandas numpy

Expects these files in the same folder (written by Cell 6 of the notebook):
    drift_results.csv         — one row per drifted feature
    ranked_features.csv       — Stage 3 ranked output
    pipeline_meta.json        — AU-PRC scores, row counts, runtime
    feature_importance.csv    — AU-PRC scores, row counts, runtime
    train.csv                 — raw training data (for distribution plots)
    test.csv                  — raw test data   (for distribution plots)
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drift Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], p, li, span, div,
[data-testid="stMarkdownContainer"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    line-height: 1.65 !important;
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
    white-space: normal !important;
}

h1 { font-size: 2.4rem !important; font-family: 'Space Mono', monospace !important; }
h2 { font-size: 1.5rem !important; font-family: 'Space Mono', monospace !important; }
h3 { font-size: 1.2rem !important; font-family: 'Space Mono', monospace !important; }

[data-testid="stSidebar"] .stButton > button {
    background-color: #1e2a4a !important;
    color: #e8edff !important;
    border: 1px solid #3a5cf7 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 10px 14px !important;
    white-space: normal !important;
    text-align: left !important;
    width: 100% !important;
    transition: background-color 0.15s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #2d3f7a !important;
    color: #ffffff !important;
    border-color: #64748b !important;
}

.metric-card {
    background: var(--background-color, #f8f9fa);
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 1.2rem 1.2rem;
    margin-bottom: 0.5rem;
    min-height: 100px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
}
.metric-label {
    font-size: 11px !important;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 6px;
    font-family: 'Space Mono', monospace !important;
    white-space: normal !important;
    min-height: 2.4em;
    line-height: 1.2 !important;
}
.metric-value {
    font-size: 1.8rem !important;
    font-weight: 700;
    font-family: 'Space Mono', monospace !important;
    line-height: 1;
}

.severe   { color: #cc1f1f; }
.moderate { color: #c2640a; }
.mild     { color: #92720a; }
.stable   { color: #06d6a0; }
.accent   { color: #64748b; }

.section-header {
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #888;
    border-bottom: 1px solid #e5e5e5;
    padding-bottom: 8px;
    margin-bottom: 1rem;
}

.pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 12px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.pill-severe   { background: #fff0f0; color: #cc1f1f; border: 1px solid #fca5a5; }
.pill-moderate { background: #fff7ed; color: #c2640a; border: 1px solid #fdba74; }
.pill-mild     { background: #fefce8; color: #92720a; border: 1px solid #fde68a; }
.pill-stable   { background: #f0fdf4; color: #166534; border: 1px solid #86efac; }

[data-testid="stDataFrame"] { font-size: 14px !important; }
.stCaption, [data-testid="stCaptionContainer"] { font-size: 13px !important; }
[data-testid="stMetricValue"] { font-size: 1.6rem !important; }
[data-testid="stMetricLabel"] { font-size: 13px !important; }
[data-baseweb="tab"] { font-size: 14px !important; }

/* Make columns in a row stretch to equal height so cards align */
[data-testid="stHorizontalBlock"] {
    align-items: stretch !important;
}
[data-testid="stHorizontalBlock"] > div {
    display: flex !important;
    flex-direction: column !important;
}
[data-testid="stHorizontalBlock"] > div > div[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stHorizontalBlock"] > div > div {
    flex: 1 !important;
}

.drift-table-header {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    padding: 8px 0 10px 0;
    border-bottom: 2px solid #555;
    margin-bottom: 2px;
}
.drift-table-header div {
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: #888 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SEV_COLOR = {"SEVERE": "#cc1f1f", "MODERATE": "#e63946", "MILD": "#f9c74f"}
SEV_ORDER = {"SEVERE": 0, "MODERATE": 1, "MILD": 2}

PLOTLY_THEME = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8f9fa",
    font=dict(family="Inter", color="#1a1a1a", size=13),
)

FD_PLOT_HEIGHT = 500

# Mitigation display labels (maps pipeline strategy strings → human labels)
MIT_LABELS = {
    "quantile_binning":              "Quantile Binning",
    "robust_scaling":                "Robust Scaling",
    "log_transform + robust_scaling":"Log + Robust Scaling",
    "frequency_encoding":            "Frequency Encoding",
    "target encoding (laplace, m=20)": "Target Encoding",
    "binarise (sparse→binary)":      "Binarisation",
    "drop_feature":                  "Drop Feature",
    "none":                          "—",
    "no_action (stable)":            "—",
    "no_action (excluded type)":     "—",
}

def _mit_display(raw: str) -> str:
    return MIT_LABELS.get(raw.lower().strip(), raw.replace("_", " ").title())

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_pipeline_outputs():
    base = Path(__file__).parent
    search_dirs = [base, base.parent, Path(".")]

    drift_df = ranked_df = meta = None
    for d in search_dirs:
        dr = d / "drift_results.csv"
        rk = d / "ranked_features.csv"
        mt = d / "pipeline_meta.json"

        if dr.exists() and rk.exists():
            drift_df  = pd.read_csv(dr)
            ranked_df = pd.read_csv(rk)

            if mt.exists():
                with open(mt) as f:
                    meta = json.load(f)
            break

    return drift_df, ranked_df, meta


_VIZ_SAMPLE = 50_000    # max rows loaded into browser for distribution plots


@st.cache_data
def load_raw_data():
    """Load train/test for visualisation only — capped at _VIZ_SAMPLE rows each.

    Uses Polars (Arrow columnar, ~3-5x less peak RAM than pandas) for the initial
    read, then samples before handing off to pandas for plotting.  The full file
    is still read — a true streaming random sample would require a two-pass
    row-count then skip, which is overkill for a dashboard — but Polars keeps
    peak memory well below the pandas equivalent.
    """
    import polars as _pl

    base = Path(__file__).parent
    for d in [base, base.parent, Path(".")]:
        tp = d / "train.csv"
        ep = d / "test.csv"
        if tp.exists() and ep.exists():
            try:
                tr_pl = _pl.read_csv(str(tp), infer_schema_length=50_000)
                te_pl = _pl.read_csv(str(ep), infer_schema_length=50_000)
                if len(tr_pl) > _VIZ_SAMPLE:
                    tr_pl = tr_pl.sample(n=_VIZ_SAMPLE, seed=42)
                if len(te_pl) > _VIZ_SAMPLE:
                    te_pl = te_pl.sample(n=_VIZ_SAMPLE, seed=42)
                return tr_pl.to_pandas(), te_pl.to_pandas()
            except Exception:
                # Fallback to pandas if Polars read fails
                tr = pd.read_csv(tp)
                te = pd.read_csv(ep)
                if len(tr) > _VIZ_SAMPLE:
                    tr = tr.sample(n=_VIZ_SAMPLE, random_state=42).reset_index(drop=True)
                if len(te) > _VIZ_SAMPLE:
                    te = te.sample(n=_VIZ_SAMPLE, random_state=42).reset_index(drop=True)
                return tr, te
    return None, None


@st.cache_data
def load_test_labels():
    """Load only CustomerID + ChurnStatus from test.csv for confusion matrix.
    Reading 2 columns is fast even at 10M rows (~80 MB vs 22 GB full load).
    """
    base = Path(__file__).parent
    for d in [base, base.parent, Path(".")]:
        ep = d / "test.csv"
        if ep.exists():
            try:
                return pd.read_csv(ep, usecols=["CustomerID", "ChurnStatus"])
            except ValueError:
                # ChurnStatus absent (hidden test set) — return just IDs
                return pd.read_csv(ep, usecols=["CustomerID"])
    return None


@st.cache_data
def load_predictions():
    base = Path(__file__).parent
    for d in [base, base.parent, Path(".")]:
        pp = d / "prediction.csv"
        if pp.exists():
            return pd.read_csv(pp)
    return None


@st.cache_data
def load_feature_importance():
    base = Path(__file__).parent
    for d in [base, base.parent, Path(".")]:
        fp = d / "feature_importance.csv"
        if fp.exists():
            return pd.read_csv(fp)
    return None


# ── Load Everything ───────────────────────────────────────────────────────────
drift_df, ranked_df, meta = load_pipeline_outputs()
train, test = load_raw_data()
test_labels = load_test_labels()   # full-fidelity labels for confusion matrix
pred_df = load_predictions()
fi_df = load_feature_importance()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 NAISC 2026")
    st.markdown("**Green Beans Drift Dashboard**")
    st.markdown("---")

    if drift_df is None:
        st.error(
            "Pipeline outputs not found.\n\n"
            "Run Cell 6 in the notebook first to export:\n"
            "- `drift_results.csv`\n"
            "- `ranked_features.csv`\n"
            "- `pipeline_meta.json`"
        )
        st.stop()

    if train is None:
        st.warning("train.csv / test.csv not found. Feature Drift page will be limited.")

    n_train = meta.get("n_train", len(train) if train is not None else "—") if meta else "—"
    n_test  = meta.get("n_test",  len(test)  if test  is not None else "—") if meta else "—"
    n_feats = meta.get("n_features", "—") if meta else "—"

    st.markdown(f"**Train:** {n_train:,}" if isinstance(n_train, int) else f"**Train:** {n_train}")
    st.markdown(f"**Test:** {n_test:,}"   if isinstance(n_test,  int) else f"**Test:** {n_test}")
    st.markdown(f"**Features:** {n_feats}")
    st.markdown("---")

    pages = ["Overview", "Feature Drift", "Drift Mitigation", "Model Performance"]
    if "page" not in st.session_state:
        st.session_state.page = "Overview"

    for p in pages:
        if st.button(p, key=f"nav_{p}", use_container_width=True):
            st.session_state.page = p
            st.rerun()

    page = st.session_state.page

# ── Shared derived data ───────────────────────────────────────────────────────
# Normalise column names (pipeline may use slightly different names)
_col_map = {}
for col in drift_df.columns:
    lc = col.lower().strip()
    if lc in ("feature", "column", "col", "feature_name"):
        _col_map["feature"] = col
    elif lc in ("drift_severity", "severity", "drift severity"):
        _col_map["severity"] = col
    elif lc in ("test_statistic", "raw_score", "stat", "score", "test_stat"):
        _col_map["stat"] = col
    elif lc in ("test_method", "test_used", "test", "method"):
        _col_map["test"] = col
    elif lc in ("mitigation", "mitigation_strategy", "mit"):
        _col_map["mitigation"] = col
    elif lc in ("feature_type", "col_type", "type", "column_type"):
        _col_map["type"] = col

def _get(row, key):
    col = _col_map.get(key)
    return row[col] if col else "—"

drifted_rows = drift_df.to_dict("records")
severities   = [str(_get(r, "severity")).upper() for r in drifted_rows]

# Count by severity
sev_counts = {
    "SEVERE":   severities.count("SEVERE"),
    "MODERATE": severities.count("MODERATE"),
    "MILD":     severities.count("MILD"),
}

# Total features analysed: from meta, or infer
total_features = meta.get("n_features", None) if meta else None
if total_features is None and train is not None:
    # Exclude structural columns
    _excl = {"CustomerID", "Month", "ChurnStatus"}
    total_features = len([c for c in train.columns if c not in _excl])
if total_features is None:
    total_features = len(drifted_rows) + 30  # rough fallback

stable_count = total_features - sum(sev_counts.values())

# ════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    n_feats_label = f"{total_features} features analysed across train/test split"
    st.markdown(
        "<h1 style='font-size:2.8rem !important;font-family:Space Mono,monospace;"
        "margin-bottom:0;'>Green Beans Drift Dashboard</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(f"*SingTel · NAISC 2026 · {n_feats_label}*")
    st.markdown("---")

    # ── Metric cards ──
    au_train = meta.get("au_prc_train") if meta else None
    au_test  = meta.get("au_prc_test")  if meta else None
    runtime  = meta.get("runtime_seconds") if meta else None

    # Compute gap
    _gap_val   = None
    _gap_color = "#888"
    _gap_arrow = ""
    if au_train and au_test:
        _gap_val = au_test - au_train
        if _gap_val > 0:
            _gap_color = "#16a34a"
            _gap_arrow = "▲ "
        elif _gap_val < 0:
            _gap_color = "#dc2626"
            _gap_arrow = "▼ "
        else:
            _gap_color = "#888"
            _gap_arrow = ""

    # ── Row 1: Feature drift counts ──────────────────────────────────────────
    st.markdown(
        '<div class="section-header" style="border-bottom:none;">Feature Drift Status Summary:</div>',
        unsafe_allow_html=True,
    )
    _drift_cards = [
        (str(sev_counts["SEVERE"]),   "Severe Drift",   "severe"),
        (str(sev_counts["MODERATE"]), "Moderate Drift", "moderate"),
        (str(sev_counts["MILD"]),     "Mild Drift",     "mild"),
        (str(stable_count),           "Stable",         "stable"),
    ]
    # 4 equal cards + right spacer to prevent them stretching full width
    _dc1, _dc2, _dc3, _dc4, _spacer1 = st.columns([1, 1, 1, 1, 2])
    for col, (val, label, cls) in zip([_dc1, _dc2, _dc3, _dc4], _drift_cards):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="min-height:90px;padding:0.9rem 1rem;">
                <div class="metric-label" style="font-size:10px !important;min-height:2.2em;">{label}</div>
                <div class="metric-value {cls}" style="font-size:2rem !important;">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # ── Row 2: Model performance ──────────────────────────────────────────────
    st.markdown(
        '<div class="section-header" style="border-bottom:none;">Model Performance:</div>',
        unsafe_allow_html=True,
    )
    _pc1, _pc2, _pc3, _spacer2 = st.columns([1, 1, 1, 3])
    _perf_cards = [
        (_pc1, f"{au_train:.4f}" if au_train else "—", "AU-PRC (Train)", "accent"),
        (_pc2, f"{au_test:.4f}"  if au_test  else "—", "AU-PRC (Test)",  "accent"),
    ]
    for col, val, label, cls in _perf_cards:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="min-height:90px;padding:0.9rem 1rem;">
                <div class="metric-label" style="font-size:10px !important;min-height:2.2em;">{label}</div>
                <div class="metric-value {cls}" style="font-size:1.6rem !important;">{val}</div>
            </div>""", unsafe_allow_html=True)

    with _pc3:
        if _gap_val is not None:
            st.markdown(f"""
            <div class="metric-card" style="min-height:90px;padding:0.9rem 1rem;">
                <div class="metric-label" style="font-size:10px !important;min-height:2.2em;">Change (Test − Train)</div>
                <div class="metric-value" style="color:{_gap_color};font-size:1.6rem !important;">
                    {_gap_arrow}{abs(_gap_val):.4f}
                </div>
            </div>""", unsafe_allow_html=True)

    if runtime:
        st.caption(f"Pipeline runtime: {runtime:.1f}s")

    st.markdown("---")

    # ── Donut + drift table ──
    st.markdown('<div class="section-header">Drift Overview</div>', unsafe_allow_html=True)
    col_left, col_right = st.columns([1, 2.8])

    with col_left:
        donut_labels = ["SEVERE", "MODERATE", "MILD", "STABLE"]
        donut_values = [sev_counts["SEVERE"], sev_counts["MODERATE"],
                        sev_counts["MILD"], stable_count]
        donut_colors = ["#cc1f1f", "#e63946", "#f9c74f", "#16a34a"] #1a3a2a

        fig_donut = go.Figure(go.Pie(
            labels=donut_labels,
            values=donut_values,
            hole=0.65,
            marker_colors=donut_colors,
            textinfo="label+value",
            textposition="outside",
            hovertemplate="<b>%{label}</b><br>%{value} features<extra></extra>",
            textfont=dict(size=12, family="Space Mono", color="#333"),
        ))
        fig_donut.add_annotation(
            text=f"<b>{total_features}</b><br>features",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, family="Space Mono", color="#333"),
            align="center",
        )
        fig_donut.update_layout(
            **PLOTLY_THEME, height=360,
            margin=dict(t=60, b=60, l=40, r=20),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top", y=-0.08,
                xanchor="center", x=0.5,
                font=dict(color="#555", size=11),
            ),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        # Grid with fixed columns — no wrapping ever
        GRID = "180px 80px 90px 65px 45px 130px"

        st.markdown(f"""
        <div style="display:grid;grid-template-columns:{GRID};gap:0 10px;
                    padding:6px 8px;background:#f5f5f5;border-radius:6px;
                    font-family:'Space Mono',monospace;font-size:12px;
                    font-weight:600;color:#555;margin-bottom:4px;">
            <div>Feature</div>
            <div>Type</div>
            <div>Severity</div>
            <div>Score</div>
            <div>Test</div>
            <div>Mitigation</div>
        </div>""", unsafe_allow_html=True)

        sorted_rows = sorted(
            drifted_rows,
            key=lambda r: (SEV_ORDER.get(str(_get(r, "severity")).upper(), 9),
            -float(_get(r, "stat") or 0))
                   )

        for d in sorted_rows:
            sev       = str(_get(d, "severity")).upper()
            feat_type = str(_get(d, "type"))
            stat_val  = _get(d, "stat")
            test_name = str(_get(d, "test"))
            mit_raw   = str(_get(d, "mitigation"))
            color     = SEV_COLOR.get(sev, "#aaa")
            pill_cls  = f"pill-{sev.lower()}"
            stat_str  = f"{float(stat_val):.4f}" if stat_val not in (None, "—", "") else "—"

            test_short = (test_name.replace("Kolmogorov-Smirnov", "KS")
                                   .replace("Population Stability Index", "PSI")
                                   .replace("Jensen-Shannon Divergence", "JS")
                                   .replace("Two-Proportion Z-Test", "Z"))

            st.markdown(f"""
            <div style="display:grid;grid-template-columns:{GRID};gap:0 10px;
                        align-items:center;padding:9px 8px;
                        border-bottom:1px solid #eee;">
                <div style="font-family:'Space Mono';font-size:12px;color:#111;
                            overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
                     title="{_get(d, 'feature')}">{_get(d, "feature")}</div>
                <div style="font-size:12px;color:#555;">{feat_type}</div>
                <div><span class="pill {pill_cls}">{sev}</span></div>
                <div style="font-family:'Space Mono';font-size:12px;color:{color};">{stat_str}</div>
                <div style="font-size:12px;color:#444;">{test_short}</div>
                <div style="font-size:12px;color:#3a5cf7;white-space:nowrap;overflow:hidden;
                            text-overflow:ellipsis;" title="{_mit_display(mit_raw)}">{_mit_display(mit_raw)}</div>
            </div>""", unsafe_allow_html=True)

    # ── Feature Importance vs. Drift Severity Scatter plot ────────────────────
    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Feature Importance vs Drift Severity</div>', unsafe_allow_html=True)

    if fi_df is not None and len(fi_df) > 0:

        # Normalize columns
        fi_cols = {c.lower().strip(): c for c in fi_df.columns}
        _feat_col = fi_cols.get("feature", fi_cols.get("feature_name"))
        _imp_col  = fi_cols.get("importance", fi_cols.get("importance_score"))

        if _feat_col and _imp_col:

            # Build drift severity lookup
            severity_map = {}
            for r in drifted_rows:
                fname = str(_get(r, "feature"))
                sev   = str(_get(r, "severity")).upper()
                severity_map[fname] = sev

            # Merge into FI dataframe
            plot_df = fi_df.copy()
            plot_df["severity"] = plot_df[_feat_col].map(severity_map).fillna("STABLE")

            # Encode severity numerically for y-axis
            sev_numeric = {"STABLE": 0, "MILD": 1, "MODERATE": 2, "SEVERE": 3}
            plot_df["severity_num"] = plot_df["severity"].map(sev_numeric)

            # Plot
            fig = px.scatter(
                plot_df,
                x=_imp_col,
                y="severity_num",
                color="severity",
                hover_name=_feat_col,
                color_discrete_map={
                    "SEVERE": "#cc1f1f",
                    "MODERATE": "#e63946",
                    "MILD": "#f9c74f",
                    "STABLE": "#16a34a"
                },
            )

            fig.update_layout(
                **PLOTLY_THEME,
                height=420,
                yaxis=dict(
                    tickmode="array",
                    tickvals=[0,1,2,3],
                    ticktext=["Stable", "Mild", "Moderate", "Severe"],
                    title="Drift Severity"
                ),
                xaxis_title="Feature Importance",
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Feature importance columns not found.")
    else:
        st.info("Feature importance data not available.")

    # ── Feature Importance Table ──────────────────────────────────────────────
    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Feature Importance (Top 10)</div>', unsafe_allow_html=True)

    fi_df = load_feature_importance()
    if fi_df is not None and len(fi_df) > 0:
        # Normalise column names
        fi_cols = {c.lower().strip(): c for c in fi_df.columns}
        _feat_col  = fi_cols.get("feature",    fi_cols.get("feature_name", None))
        _imp_col   = fi_cols.get("importance", fi_cols.get("importance_score", None))
        _rank_col  = fi_cols.get("rank",       None)

        if _feat_col and _imp_col:
            # Build top-10 rows
            fi_top = fi_df.copy()
            if _rank_col:
                fi_top = fi_top.sort_values(_rank_col).head(10)
            else:
                fi_top = fi_top.sort_values(_imp_col, ascending=False).head(10)
            fi_top = fi_top.reset_index(drop=True)

            # Build drift lookup for drift status column
            _drift_status_lookup: dict[str, str] = {}
            for _r in drifted_rows:
                _fn  = str(_get(_r, "feature"))
                _sev = str(_get(_r, "severity")).upper()
                _mit = str(_get(_r, "mitigation"))
                # Compose a human-readable drift status
                if _sev == "SEVERE":
                    _ds = f"HIGH ({_mit_display(_mit)})" if _mit not in ("—", "none", "") else "HIGH"
                elif _sev == "MODERATE":
                    _ds = "MODERATE"
                elif _sev == "MILD":
                    _ds = "Low (but KS significant)" if "ks" in str(_get(_r, "test")).lower() else "Low"
                else:
                    _ds = "Low"
                _drift_status_lookup[_fn] = _ds
                
            # Table header
            FI_GRID = "50px 2fr 180px 1.5fr"
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:{FI_GRID};gap:0 10px;
                    padding:6px 8px;background:#f5f5f5;border-radius:6px;
                    font-family:'Space Mono',monospace;font-size:12px;
                    font-weight:600;color:#555;margin-bottom:4px;">
                <div>Rank</div>
                <div>Feature</div>
                <div>Importance</div>
                <div>Drift Status</div>
            </div>""", unsafe_allow_html=True)

            # Table rows
            _high_drift_in_top7 = 0
            for _i, _row in fi_top.iterrows():
                _rank_num  = int(_row[_rank_col]) if _rank_col and _rank_col in fi_top.columns else (_i + 1)
                _feat_name = str(_row[_feat_col])
                _imp_val   = _row[_imp_col]
                _imp_str   = f"{int(_imp_val):,}" if _imp_val == int(_imp_val) else f"{_imp_val:,.0f}"
                _ds        = _drift_status_lookup.get(_feat_name, "Low")
                _is_high   = _ds.upper().startswith("HIGH") or _ds.upper() == "MODERATE"
                _row_bg    = "#fff" if _rank_num % 2 == 0 else "#fafafa"
                _ds_style  = "font-weight:700;color:#1a1a1a;" if _is_high else "color:#555;"
                
                if _rank_num <= 7 and _is_high:
                    _high_drift_in_top7 += 1

                st.markdown(f"""
                <div style="display:grid;grid-template-columns:{FI_GRID};gap:0 12px;
                            padding:10px 14px;background:{_row_bg};
                            border-bottom:1px solid #eee;align-items:center;">
                    <div style="font-family:'Space Mono',monospace;font-size:13px;
                                color:#444;">{_rank_num}</div>
                    <div style="font-size:14px;color:#1a1a1a;">{_feat_name}</div>
                    <div style="font-family:'Space Mono',monospace;font-size:13px;
                                color:#555;">{_imp_str}</div>
                    <div style="font-size:13px;{_ds_style}">{_ds}</div>
                </div>""", unsafe_allow_html=True)

            # Critical finding callout
            if _high_drift_in_top7 > 0:
                st.markdown(
                    f"<div style='padding:10px 14px 10px 14px;background:#fffbeb;"
                    f"border-left:4px solid #f59e0b;border-radius:0 0 8px 8px;"
                    f"font-size:13px;color:#78350f;line-height:1.5;'>"
                    f"<b>Critical finding:</b> {_high_drift_in_top7} of the top 7 most important features "
                    f"have HIGH drift." 
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div style='padding:10px 14px;background:#f0fdf4;"
                    "border-left:4px solid #16a34a;border-radius:0 0 8px 8px;"
                    "font-size:13px;color:#166534;line-height:1.5;'>"
                    "<b>Note:</b> No high-drift features detected among the top 7 most important features."
                    "</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("feature_importance.csv found but expected columns (feature, importance) are missing.")
    else:
        st.info(
            "Feature importance data not found. "
            "Run the pipeline (main_edited.py) to generate `feature_importance.csv`."
        )

# ════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — FEATURE DRIFT
# ════════════════════════════════════════════════════════════════════════════
elif page == "Feature Drift":
    st.markdown("# Feature Drift")
    st.markdown("Interactive exploration of individual feature distributions")
    st.markdown("---")

    if train is not None and len(train) >= _VIZ_SAMPLE:
        st.info(
            f"ℹ️ Distribution plots show a random sample of {_VIZ_SAMPLE:,} rows "
            f"(dataset has {meta.get('n_train', '?'):,} train rows). "
            "Drift detection ran on the full dataset."
        )

    if train is None or test is None:
        st.error("train.csv and test.csv are required for this page. "
                 "Place them in the same folder as dashboard.py.")
        st.stop()

    num_cols = [c for c in train.select_dtypes(include="number").columns
                if c not in ("CustomerID",)]
    cat_cols = [c for c in train.select_dtypes(include="object").columns
                if c not in ("CustomerID", "ChurnStatus", "Month")]

    all_features   = sorted(num_cols + cat_cols)
    search_term    = st.text_input("🔍 Search features", "")
    filtered_feats = [f for f in all_features if search_term.lower() in f.lower()]

    # Build drift lookup from drift_df
    drift_lookup = {}
    for r in drifted_rows:
        fname = str(_get(r, "feature"))
        drift_lookup[fname] = {
            "severity":   str(_get(r, "severity")).upper(),
            "stat":       _get(r, "stat"),
            "test":       str(_get(r, "test")),
            "mitigation": str(_get(r, "mitigation")),
        }

    def _find_drift(feature):
        """Fuzzy match feature name against drift_lookup keys."""
        if feature in drift_lookup:
            return drift_lookup[feature]
        for k, v in drift_lookup.items():
            if k.replace("...", "").strip() in feature or feature in k:
                return v
        return None

    SEVERITY_ORDER = {"severe": 3, "moderate": 2, "mild": 1, "stable": 0}

    def _drift_rank(feature):
        info = _find_drift(feature)
        if info:
            return SEVERITY_ORDER.get(info["severity"].lower(), 0)
        return 0

    filtered_feats = sorted(filtered_feats, key=_drift_rank, reverse=True)

    # ── View mode toggle: paginated gallery vs single feature ─────────────────
    _view_mode = st.radio(
        "View",
        ["All Features View", "Single Feature View"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("---")

    def render_feature_plot(feature, show_train=True, show_test=True):
        """Render one full-width plot. show_train / show_test control which traces appear."""
        is_numeric = feature in num_cols
        drift_info = _find_drift(feature)

        # ── Drift badge ───────────────────────────────────────────────────────
        if drift_info:
            sev = drift_info["severity"].lower()
            st.markdown(
                f'<span class="pill pill-{sev}">{sev.upper()} DRIFT DETECTED</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<span class="pill pill-stable">STABLE — no drift</span>',
                        unsafe_allow_html=True)

        # ── Build figure ──────────────────────────────────────────────────────
        # Colours: Train = blue, Test = coral/red, both semi-transparent so
        # overlaps appear as purple (numeric) and bars sit cleanly side-by-side
        # (categorical).
        _C_TRAIN = "#3b82f6"   # blue
        _C_TEST  = "#e63946"   # red

        if is_numeric:
            tr_vals = train[feature].dropna().astype(float)
            te_vals = test[feature].dropna().astype(float)

            both_vals = pd.concat([tr_vals, te_vals], ignore_index=True)
            if both_vals.empty:
                st.info(f"No numeric values available for `{feature}`.")
                return

            x_min = float(both_vals.min())
            x_max = float(both_vals.max())
            if np.isclose(x_min, x_max):
                x_min -= 0.5
                x_max += 0.5

            bins = np.linspace(x_min, x_max, 31)

            tr_hist = np.histogram(tr_vals, bins=bins)[0]
            te_hist = np.histogram(te_vals, bins=bins)[0]
            tr_prob = tr_hist / tr_hist.sum() if tr_hist.sum() > 0 else np.zeros_like(tr_hist, dtype=float)
            te_prob = te_hist / te_hist.sum() if te_hist.sum() > 0 else np.zeros_like(te_hist, dtype=float)
            y_max = float(max(tr_prob.max(initial=0.0), te_prob.max(initial=0.0), 0.01))

            bin_size = (x_max - x_min) / 30.0
            fig = go.Figure()
            if show_train:
                fig.add_trace(go.Histogram(
                    x=tr_vals, name="Train",
                    marker_color=_C_TRAIN, opacity=0.55,
                    histnorm="probability",
                    xbins=dict(start=x_min, end=x_max, size=bin_size),
                ))
            if show_test:
                fig.add_trace(go.Histogram(
                    x=te_vals, name="Test",
                    marker_color=_C_TEST, opacity=0.55,
                    histnorm="probability",
                    xbins=dict(start=x_min, end=x_max, size=bin_size),
                ))
            fig.update_layout(
                **PLOTLY_THEME, barmode="overlay", height=FD_PLOT_HEIGHT,
                title=dict(text=feature, x=0.5, xanchor="center"),
                legend=dict(font=dict(color="#555")),
                yaxis=dict(range=[0, y_max * 1.08], title="Proportion"),
            )
        else:
            # Grouped bars — no occlusion for categorical features
            tr_vc = (train[feature].str.lower().str.strip()
                     .fillna("(null)").value_counts(normalize=True))
            te_vc = (test[feature].str.lower().str.strip()
                     .fillna("(null)").value_counts(normalize=True))
            cats = sorted(set(tr_vc.index) | set(te_vc.index))
            tr_y = [tr_vc.get(c, 0) for c in cats]
            te_y = [te_vc.get(c, 0) for c in cats]
            y_max = float(max(max(tr_y, default=0.0), max(te_y, default=0.0), 0.01))
            fig = go.Figure()
            if show_train:
                fig.add_trace(go.Bar(
                    x=cats, y=tr_y,
                    name="Train", marker_color=_C_TRAIN, opacity=0.85,
                ))
            if show_test:
                fig.add_trace(go.Bar(
                    x=cats, y=te_y,
                    name="Test", marker_color=_C_TEST, opacity=0.85,
                ))
            fig.update_layout(
                **PLOTLY_THEME, barmode="group", height=FD_PLOT_HEIGHT,
                title=dict(text=feature, x=0.5, xanchor="center"),
                legend=dict(font=dict(color="#555")),
                bargap=0.20, bargroupgap=0.05,
                yaxis=dict(range=[0, y_max * 1.08], title="Proportion"),
            )

        if drift_info:
            stat_val  = drift_info["stat"]
            test_name = (str(drift_info["test"])
                         .replace("Kolmogorov-Smirnov", "KS")
                         .replace("Population Stability Index", "PSI")
                         .replace("Jensen-Shannon Divergence", "JS"))
            stat_str  = f"{float(stat_val):.4f}" if stat_val not in (None, "—", "") else "—"
            fig.add_annotation(
                text=f"{test_name} = {stat_str}",
                xref="paper", yref="paper", x=0.99, y=0.97,
                showarrow=False, align="right",
                font=dict(size=11, color="#888", family="Space Mono"),
            )
        st.plotly_chart(fig, use_container_width=True)

    # ── Paginated gallery ─────────────────────────────────────────────────────
    if _view_mode == "All Features View":
        _PLOTS_PER_PAGE = 5

        _total_feats = len(filtered_feats)
        _total_pages = max(1, (_total_feats + _PLOTS_PER_PAGE - 1) // _PLOTS_PER_PAGE)

        # Reset page number when search changes
        _page_key = f"fd_page_{search_term}"
        if _page_key not in st.session_state:
            st.session_state[_page_key] = 1
        if st.session_state[_page_key] > _total_pages:
            st.session_state[_page_key] = 1

        _cur_page = st.session_state[_page_key]

        # ── Severity breakdown badges ─────────────────────────────────────────
        _n_stable = _total_feats - sum(1 for f in filtered_feats if _find_drift(f))
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:12px;'>"
            f"<span style='font-family:\"Space Mono\",monospace;font-size:13px;color:#555;'>"
            f"{_total_feats} features</span>"
            f"<span class='pill pill-severe'>{sum(1 for f in filtered_feats if _find_drift(f) and _find_drift(f)['severity']=='SEVERE')} Severe</span>"
            f"<span class='pill pill-moderate'>{sum(1 for f in filtered_feats if _find_drift(f) and _find_drift(f)['severity']=='MODERATE')} Moderate</span>"
            f"<span class='pill pill-mild'>{sum(1 for f in filtered_feats if _find_drift(f) and _find_drift(f)['severity']=='MILD')} Mild</span>"
            f"<span class='pill pill-stable'>{_n_stable} Stable</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Pagination controls (top) ─────────────────────────────────────────
        _start_idx  = (_cur_page - 1) * _PLOTS_PER_PAGE
        _end_idx    = min(_start_idx + _PLOTS_PER_PAGE, _total_feats)
        _page_feats = filtered_feats[_start_idx:_end_idx]

        _c1, _c2, _c3 = st.columns([1.2, 1.5, 2.3])

        def _set_fd_page(new_page: int):
            st.session_state[_page_key] = int(max(1, min(_total_pages, new_page)))

        with _c3:
            st.markdown(
                f"<div style='font-family:\"Space Mono\",monospace;font-size:12px;color:#888;padding-top:8px;'>"
                f"Page {_cur_page} of {_total_pages} · features {_start_idx+1}–{_end_idx} of {_total_feats}</div>",
                unsafe_allow_html=True,
            )

        with _c1:
            _typed_page_top = st.number_input(
                "Page (+/-)",
                min_value=1, max_value=_total_pages,
                value=_cur_page,
                step=1,
                help="Use +/- to move one page at a time.",
            )
            if int(_typed_page_top) != _cur_page:
                _set_fd_page(int(_typed_page_top))
                st.rerun()

        with _c2:
            _picked_page_top = st.selectbox(
                "Select page",
                options=list(range(1, _total_pages + 1)),
                index=_cur_page - 1,
                format_func=lambda p: f"Page {p}",
            )
            if int(_picked_page_top) != _cur_page:
                _set_fd_page(int(_picked_page_top))
                st.rerun()

        st.markdown("<div style='margin-bottom:4px;'></div>", unsafe_allow_html=True)

        # ── Render each feature as a full-width row with toggle buttons ───────
        for feat in _page_feats:
            st.markdown("---")

            # Per-feature session state key for toggle
            _toggle_key = f"fd_toggle_{feat}"
            if _toggle_key not in st.session_state:
                st.session_state[_toggle_key] = "both"

            # Toggle buttons: Train | Test | Reset — right-aligned
            _sp, _btn_train, _btn_test, _btn_reset = st.columns([4, 0.5, 0.5, 0.5])
            with _btn_train:
                _train_active = st.session_state[_toggle_key] == "train"
                if st.button(
                    "Train",
                    key=f"btn_train_{feat}",
                    use_container_width=True,
                    type="primary" if _train_active else "secondary",
                ):
                    st.session_state[_toggle_key] = "train"
                    st.rerun()
            with _btn_test:
                _test_active = st.session_state[_toggle_key] == "test"
                if st.button(
                    "Test",
                    key=f"btn_test_{feat}",
                    use_container_width=True,
                    type="primary" if _test_active else "secondary",
                ):
                    st.session_state[_toggle_key] = "test"
                    st.rerun()
            with _btn_reset:
                if st.button(
                    "Both",
                    key=f"btn_reset_{feat}",
                    use_container_width=True,
                    type="primary" if st.session_state[_toggle_key] == "both" else "secondary",
                ):
                    st.session_state[_toggle_key] = "both"
                    st.rerun()

            _mode       = st.session_state[_toggle_key]
            _show_train = _mode in ("both", "train")
            _show_test  = _mode in ("both", "test")
            render_feature_plot(feat, show_train=_show_train, show_test=_show_test)

        st.markdown("---")

    # ── Single feature view ───────────────────────────────────────────────────
    else:
        _selected   = st.selectbox("Select feature", filtered_feats)
        _toggle_key = f"fd_toggle_single_{_selected}"
        if _toggle_key not in st.session_state:
            st.session_state[_toggle_key] = "both"

        _sp, _btn_train, _btn_test, _btn_reset = st.columns([4, 0.5, 0.5, 0.5])
        with _btn_train:
            if st.button("Train", key="single_train", use_container_width=True,
                         type="primary" if st.session_state[_toggle_key]=="train" else "secondary"):
                st.session_state[_toggle_key] = "train"; st.rerun()
        with _btn_test:
            if st.button("Test", key="single_test", use_container_width=True,
                         type="primary" if st.session_state[_toggle_key]=="test" else "secondary"):
                st.session_state[_toggle_key] = "test"; st.rerun()
        with _btn_reset:
            if st.button("Both", key="single_both", use_container_width=True,
                         type="primary" if st.session_state[_toggle_key]=="both" else "secondary"):
                st.session_state[_toggle_key] = "both"; st.rerun()

        _mode = st.session_state[_toggle_key]
        render_feature_plot(_selected,
                            show_train=_mode in ("both", "train"),
                            show_test=_mode  in ("both", "test"))


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — DRIFT MITIGATION
# ════════════════════════════════════════════════════════════════════════════
elif page == "Drift Mitigation":
    st.markdown("# Drift Mitigation")
    st.markdown("Before and after mitigation")
    st.markdown("---")

    if train is None or test is None:
        st.error("train.csv and test.csv are required for mitigation charts.")
        st.stop()

    # Group drifted features by mitigation strategy
    _MIT_GROUPS = {
        "quantile":  [],
        "robust":    [],
        "target":    [],
        "frequency": [],
        "other":     [],
    }
    for r in drifted_rows:
        mit = str(_get(r, "mitigation")).lower()
        feat = str(_get(r, "feature"))
        sev  = str(_get(r, "severity")).upper()
        ftype = str(_get(r, "type")).lower()
        entry = {"feature": feat, "severity": sev, "type": ftype, "mitigation": mit}

        if "quantile" in mit:
            _MIT_GROUPS["quantile"].append(entry)
        elif "robust" in mit or "scaling" in mit or "log" in mit:
            _MIT_GROUPS["robust"].append(entry)
        elif "target" in mit or (
            "frequency" in mit and ftype in ("categorical", "low_card_cat")
        ):
            _MIT_GROUPS["target"].append(entry)
        elif "frequency" in mit:
            _MIT_GROUPS["frequency"].append(entry)
        else:
            _MIT_GROUPS["other"].append(entry)

    # Build tab list dynamically — only show tabs with at least one feature
    tab_defs = []
    if _MIT_GROUPS["quantile"]:
        tab_defs.append(("Quantile Binning",  "quantile"))
    if _MIT_GROUPS["robust"]:
        tab_defs.append(("Robust Scaling",    "robust"))
    if _MIT_GROUPS["target"]:
        tab_defs.append(("Target Encoding",   "target"))
    if _MIT_GROUPS["frequency"]:
        tab_defs.append(("Freq. Encoding",    "frequency"))
    if _MIT_GROUPS["other"]:
        tab_defs.append(("Other",             "other"))

    if not tab_defs:
        st.info("No drifted features with mitigation strategies found.")
        st.stop()

    tabs = st.tabs([t[0] for t in tab_defs])

    def _sev_pill(sev: str) -> str:
        _s = str(sev).upper()
        return f"<span class='pill pill-{_s.lower()}'>{_s}</span>"

    def _mit_callouts(before_text: str, after_text: str):
        _bcol, _acol = st.columns(2)
        with _bcol:
            st.markdown(
                f"<div style='padding:10px 12px;background:#fffbea;border:1px solid #e3c347;"
                f"border-radius:8px;font-size:12px;color:#555;line-height:1.45;'>"
                f"<b>Before:</b> {before_text}</div>",
                unsafe_allow_html=True,
            )
        with _acol:
            st.markdown(
                f"<div style='padding:10px 12px;background:#f7fbff;border:1px solid #7fb3e6;"
                f"border-radius:8px;font-size:12px;color:#555;line-height:1.45;'>"
                f"<b>After:</b> {after_text}</div>",
                unsafe_allow_html=True,
            )

    for tab, (tab_label, group_key) in zip(tabs, tab_defs):
        with tab:
            entries = _MIT_GROUPS[group_key]

            # ── Quantile Binning ──────────────────────────────────────────
            if group_key == "quantile":
                st.markdown(
                    "**Quantile binning** converts raw continuous values into decile "
                    "ranks (0–9). The rank is stable across temporal shifts — "
                    "the model only needs to know which decile a customer falls in."
                )
                for entry in entries:
                    feat = entry["feature"]
                    if feat not in train.columns:
                        st.warning(f"Column `{feat}` not found in train.csv"); continue

                    st.markdown(f"#### {feat} {_sev_pill(entry['severity'])}", unsafe_allow_html=True)
                    tr_vals = train[feat].dropna()
                    te_vals = test[feat].dropna() if feat in test.columns else pd.Series([], dtype=float)

                    # Compute decile edges from train
                    edges = np.quantile(tr_vals, np.linspace(0, 1, 21)).tolist()
                    mids  = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges) - 1)]
                    tr_hist, _ = np.histogram(tr_vals, bins=edges, density=False)
                    te_hist, _ = np.histogram(te_vals, bins=edges, density=False)
                    tr_n = tr_hist / tr_hist.sum() if tr_hist.sum() > 0 else tr_hist
                    te_n = te_hist / te_hist.sum() if te_hist.sum() > 0 else te_hist

                    # Binned (equal-freq) — train flat, test may pile
                    n_bins = 10
                    bin_edges = np.quantile(tr_vals, np.linspace(0, 1, n_bins + 1))
                    tr_bin_labels = pd.cut(tr_vals, bins=bin_edges, labels=False,
                                           include_lowest=True)
                    te_bin_labels = pd.cut(te_vals, bins=bin_edges, labels=False,
                                           include_lowest=True)
                    tr_bin_counts = [
                        (tr_bin_labels == i).sum() / len(tr_bin_labels)
                        for i in range(n_bins)
                    ]
                    te_bin_counts = [
                        (te_bin_labels == i).sum() / max(len(te_bin_labels), 1)
                        for i in range(n_bins)
                    ]

                    fig = make_subplots(
                        rows=1,
                        cols=2,
                        subplot_titles=["Before: Raw Distribution", "After: Binned"],
                        vertical_spacing=0.14,
                    )
                    fig.add_trace(go.Scatter(x=mids, y=tr_n.tolist(), name="Train",
                                             mode="lines", line=dict(color="#3b82f6", width=2),
                                             fill="tozeroy",
                                             fillcolor="rgba(123,156,247,0.2)"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=mids, y=te_n.tolist(), name="Test",
                                             mode="lines", line=dict(color="#e63946", width=2),
                                             fill="tozeroy",
                                             fillcolor="rgba(255,77,77,0.2)"), row=1, col=1)
                    fig.add_trace(go.Bar(x=list(range(n_bins)), y=tr_bin_counts,
                                         name="Train (binned)", marker_color="#3b82f6",
                                         opacity=0.8), row=1, col=2)
                    fig.add_trace(go.Bar(x=list(range(n_bins)), y=te_bin_counts,
                                         name="Test (binned)",  marker_color="#e63946",
                                         opacity=0.8), row=1, col=2)
                    fig.add_hline(
                        y=1 / n_bins,
                        line_dash="dash",
                        line_color="#888",
                        annotation_text="Expected uniform (10%)",
                        annotation_font_color="#666",
                        row=1,
                        col=2,
                    )
                    fig.update_layout(**PLOTLY_THEME, height=360, margin=dict(t=70, b=16),
                                      showlegend=True,
                                      legend=dict(font=dict(color="#555")))
                    fig.update_annotations(yshift=12)
                    fig.update_xaxes(gridcolor="#e5e5e5", linecolor="#ccc")
                    fig.update_yaxes(gridcolor="#e5e5e5", linecolor="#ccc")
                    st.plotly_chart(fig, use_container_width=True)
                    _mit_callouts(
                        "Raw values show temporal concentration shifts and can look like severe drift.",
                        "Ordinal bins preserve rank structure, so the model learns stable ordering rather than brittle absolute values.",
                    )
                    st.caption("How to read: if test mass concentrates in higher bins while train remains near-uniform by design, the shift is likely seasonal structure mapped into stable ordinal bands.")

            # ── Robust Scaling ────────────────────────────────────────────
            elif group_key == "robust":
                st.markdown(
                    "**Robust scaling** applies `(x − median) / IQR` using "
                    "parameters fitted on train only. Median/IQR are more "
                    "robust to outliers than mean/std."
                )
                valid = [e for e in entries if e["feature"] in train.columns]
                if not valid:
                    st.info("No matching columns found in train.csv."); continue

                for entry in valid:
                    feat = entry["feature"]
                    tr_vals = train[feat].dropna()
                    te_vals = test[feat].dropna() if feat in test.columns else pd.Series([], dtype=float)
                    median  = tr_vals.median()
                    iqr     = tr_vals.quantile(0.75) - tr_vals.quantile(0.25)
                    if iqr == 0:
                        st.warning(f"`{feat}`: IQR = 0, scaling skipped."); continue
                    tr_sc = (tr_vals - median) / iqr
                    te_sc = (te_vals - median) / iqr

                    st.markdown(f"#### {feat} {_sev_pill(entry['severity'])}", unsafe_allow_html=True)
                    st.markdown(f"Median: `{median:.2f}` · IQR: `{iqr:.2f}`")
                    fig2 = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=["Before: Raw Distribution", "After: CDF Post Robust Scaling"],
                        vertical_spacing=0.14,
                    )
                    fig2.add_trace(go.Histogram(x=tr_vals, name="Train (raw)",
                                                nbinsx=30, marker_color="#3b82f6",
                                                opacity=0.65, histnorm="probability"),
                                   row=1, col=1)
                    fig2.add_trace(go.Histogram(x=te_vals, name="Test (raw)",
                                                nbinsx=30, marker_color="#e63946",
                                                opacity=0.65, histnorm="probability"),
                                   row=1, col=1)
                    tr_sc_sorted = np.sort(tr_sc.to_numpy())
                    te_sc_sorted = np.sort(te_sc.to_numpy())
                    tr_cdf = np.arange(1, len(tr_sc_sorted) + 1) / max(len(tr_sc_sorted), 1)
                    te_cdf = np.arange(1, len(te_sc_sorted) + 1) / max(len(te_sc_sorted), 1)
                    fig2.add_trace(go.Scatter(
                        x=tr_sc_sorted, y=tr_cdf,
                        name="Train (scaled)", mode="lines",
                        line=dict(color="#3b82f6", width=2),
                        showlegend=False,
                    ), row=1, col=2)
                    fig2.add_trace(go.Scatter(
                        x=te_sc_sorted, y=te_cdf,
                        name="Test (scaled)", mode="lines",
                        line=dict(color="#e63946", width=2),
                        showlegend=False,
                    ), row=1, col=2)

                    _ks_raw = entry.get("stat", 0.0)
                    ks_stat = float(_ks_raw) if _ks_raw not in (None, "", "—") else 0.0
                    if len(tr_sc_sorted) and len(te_sc_sorted):
                        allx = np.concatenate([tr_sc_sorted, te_sc_sorted])
                        xs = np.linspace(allx.min(), allx.max(), 600)
                        tr_i = np.interp(xs, tr_sc_sorted, tr_cdf)
                        te_i = np.interp(xs, te_sc_sorted, te_cdf)
                        idx = int(np.argmax(np.abs(tr_i - te_i)))
                        gap_x = xs[idx]
                        gap_y1 = tr_i[idx]
                        gap_y2 = te_i[idx]
                        fig2.add_annotation(
                            x=gap_x,
                            y=(gap_y1 + gap_y2) / 2,
                            text=f"KS D = {ks_stat:.3f}",
                            showarrow=False,
                            font=dict(size=11, color="#c0392b", family="Space Mono"),
                            bgcolor="rgba(255,255,255,0.85)",
                            bordercolor="#f0b0b0",
                        )
                    fig2.update_layout(
                        **PLOTLY_THEME, barmode="overlay", height=360,
                        margin=dict(t=70, b=24), showlegend=True,
                        legend=dict(font=dict(color="#555")),
                    )
                    fig2.update_annotations(yshift=12)
                    fig2.update_xaxes(gridcolor="#e5e5e5", linecolor="#ccc")
                    fig2.update_yaxes(gridcolor="#e5e5e5", linecolor="#ccc")
                    fig2.update_yaxes(title_text="Proportion", row=1, col=1)
                    fig2.update_yaxes(title_text="Cumulative proportion", row=1, col=2)
                    st.plotly_chart(fig2, use_container_width=True)
                    _mit_callouts(
                        "Raw distributions are shifted in location and spread, so identical raw values can map to different risk contexts.",
                        "After centring by train median and IQR, CDF curves move closer, reducing split mismatch for model decision thresholds.",
                    )
                    st.caption("How to read: in the CDF panel, smaller vertical separation between train and test implies smaller residual distribution gap after mitigation.")

            # ── Target Encoding ───────────────────────────────────────────
            elif group_key == "target":
                st.markdown(
                    "**Laplace-smoothed target encoding (m=20)** — each category "
                    "is replaced by its smoothed churn rate from training data. "
                    "Formula: `(n × rate + 20 × global_rate) / (n + 20)`"
                )
                target_col = "ChurnStatus"
                if target_col not in train.columns:
                    st.warning(f"`{target_col}` not in train.csv"); continue

                global_rate = (train[target_col] == "Yes").mean()

                valid = [e for e in entries if e["feature"] in train.columns]
                if not valid:
                    st.info("No matching columns found."); continue

                for entry in valid:
                    feat = entry["feature"]
                    is_num_feat = feat in train.select_dtypes(include="number").columns
                    if is_num_feat:
                        grp = (train.groupby(feat)[target_col]
                               .agg(n="count",
                                    raw_rate=lambda x: (x == "Yes").mean())
                               .reset_index())
                    else:
                        grp = (train.assign(feat_norm=train[feat].str.lower().str.strip())
                               .groupby("feat_norm")[target_col]
                               .agg(n="count",
                                    raw_rate=lambda x: (x == "Yes").mean())
                               .reset_index())

                    grp["smoothed"] = (grp["n"] * grp["raw_rate"] + 20 * global_rate) / (grp["n"] + 20)
                    x_col = feat if is_num_feat else "feat_norm"

                    st.markdown(f"#### {feat} {_sev_pill(entry['severity'])}", unsafe_allow_html=True)
                    st.markdown(f"Global churn rate: `{global_rate:.3f}`")

                    fig3 = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=[
                            "Before: Raw Category Distribution",
                            "After: Churn Rate per Category",
                        ],
                        vertical_spacing=0.14,
                    )

                    tr_vc = (train[feat].str.lower().str.strip()
                             .fillna("(null)").value_counts(normalize=True)
                             if not is_num_feat
                             else train[feat].dropna().value_counts(normalize=True))
                    te_vc = (test[feat].str.lower().str.strip()
                             .fillna("(null)").value_counts(normalize=True)
                             if feat in test.columns and not is_num_feat
                             else (test[feat].dropna().value_counts(normalize=True)
                                   if feat in test.columns else pd.Series(dtype=float)))
                    cats_before = sorted(set(tr_vc.index) | set(te_vc.index),
                                         key=lambda c: -tr_vc.get(c, 0))[:20]
                    fig3.add_trace(go.Bar(
                        x=[str(c) for c in cats_before],
                        y=[tr_vc.get(c, 0) for c in cats_before],
                        name="Train", marker_color="#3b82f6", opacity=0.8,
                    ), row=1, col=1)
                    fig3.add_trace(go.Bar(
                        x=[str(c) for c in cats_before],
                        y=[te_vc.get(c, 0) for c in cats_before],
                        name="Test", marker_color="#e63946", opacity=0.8,
                    ), row=1, col=1)

                    if is_num_feat:
                        train_rate = (train.groupby(feat)[target_col]
                                      .apply(lambda x: (x == "Yes").mean())
                                      .to_dict())
                        test_rate = (test.groupby(feat)[target_col]
                                     .apply(lambda x: (x == "Yes").mean())
                                     .to_dict() if target_col in test.columns and feat in test.columns else {})
                        cats_after = sorted(set(train_rate.keys()) | set(test_rate.keys()))
                    else:
                        train_rate = (train.assign(_cat=train[feat].str.lower().str.strip())
                                      .groupby("_cat")[target_col]
                                      .apply(lambda x: (x == "Yes").mean())
                                      .to_dict())
                        test_rate = (test.assign(_cat=test[feat].str.lower().str.strip())
                                     .groupby("_cat")[target_col]
                                     .apply(lambda x: (x == "Yes").mean())
                                     .to_dict() if target_col in test.columns and feat in test.columns else {})
                        cats_after = sorted(set(train_rate.keys()) | set(test_rate.keys()), key=lambda c: -tr_vc.get(c, 0))[:20]

                    fig3.add_trace(go.Bar(
                        x=[str(c) for c in cats_after],
                        y=[train_rate.get(c, 0) for c in cats_after],
                        name="Train churn rate", marker_color="#3b82f6", opacity=0.8,
                        showlegend=True,
                    ), row=1, col=2)
                    fig3.add_trace(go.Bar(
                        x=[str(c) for c in cats_after],
                        y=[test_rate.get(c, 0) for c in cats_after],
                        name="Test churn rate", marker_color="#e63946", opacity=0.8,
                        showlegend=True,
                    ), row=1, col=2)
                    fig3.add_hline(
                        y=global_rate, line_dash="dash", line_color="#555",
                        annotation_text=f"Global Churn Rate {global_rate:.3f}",
                        annotation_font_color="#555", row=1, col=2,
                    )

                    fig3.update_layout(
                        **PLOTLY_THEME, barmode="group", height=420,
                        margin=dict(t=80, b=70),
                        legend=dict(font=dict(color="#555")),
                    )
                    fig3.update_annotations(yshift=12)
                    fig3.update_xaxes(tickangle=-35, gridcolor="#e5e5e5", linecolor="#ccc")
                    fig3.update_yaxes(gridcolor="#e5e5e5", linecolor="#ccc")
                    fig3.update_yaxes(title_text="Proportion", row=1, col=1)
                    fig3.update_yaxes(title_text="Churn Rate", row=1, col=2)
                    st.plotly_chart(fig3, use_container_width=True)
                    _mit_callouts(
                        "Category proportions shifted between train and test, which can make raw category IDs unreliable as model splits.",
                        "Churn-rate signals per category stay comparatively stable, so target encoding turns unstable labels into robust numeric signal.",
                    )
                    st.caption("How to read: if train and test churn-rate bars are similar per category, the predictive signal is stable even when category mix drifts.")

            # ── Frequency Encoding ────────────────────────────────────────
            elif group_key == "frequency":
                st.markdown(
                    "**Frequency encoding** replaces each high-cardinality string "
                    "with its train row count. Provides a population-density proxy "
                    "with zero target leakage."
                )
                for entry in entries:
                    feat = entry["feature"]
                    if feat not in train.columns:
                        st.warning(f"`{feat}` not found in train.csv"); continue
                    freq_map = train[feat].value_counts()
                    top20 = freq_map.head(20)
                    fig4 = go.Figure(go.Bar(
                        x=top20.index.astype(str),
                        y=top20.values,
                        marker_color="#64748b", opacity=0.85,
                    ))
                    fig4.update_layout(
                        **PLOTLY_THEME, height=320,
                        title=dict(text=f"{feat} — {entry['severity']} — Top 20 categories by train frequency",
                                   x=0.5, xanchor="center"),
                        xaxis=dict(tickangle=-35, gridcolor="#e5e5e5"),
                        yaxis=dict(title="Train frequency", gridcolor="#e5e5e5"),
                        margin=dict(t=50, b=60),
                    )
                    st.plotly_chart(fig4, use_container_width=True)
                    st.caption(f"{feat}: {train[feat].nunique():,} unique values → "
                               "replaced by Int64 frequency count")

            # ── Other ─────────────────────────────────────────────────────
            else:
                for entry in entries:
                    feat = entry["feature"]
                    mit  = _mit_display(entry["mitigation"])
                    st.markdown(f"- **{feat}** ({entry['severity']}) → {mit}")

# ════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown(
        "<h1 style='font-size:2.8rem !important;font-family:Space Mono,monospace;"
        "margin-bottom:0;'>Model Performance</h1>",
        unsafe_allow_html=True,
    )
    st.markdown("*Confusion Matrix*")
    st.markdown("---")

    # Build confusion matrix from prediction.csv + test.csv (if labels exist)
    _cm_ready   = False
    _cm_message = ""

    if pred_df is not None and test_labels is not None and "ChurnStatus" in test_labels.columns:
        try:
            # Merge predictions with true labels on CustomerID.
            # Uses test_labels (CustomerID + ChurnStatus only) — fast even at 10M rows.
            _pred_col = "probability_score"
            _id_col   = "CustomerID"

            if _id_col in pred_df.columns and _id_col in test_labels.columns:
                _merged = pred_df[[_id_col, _pred_col]].merge(
                    test_labels[[_id_col, "ChurnStatus"]], on=_id_col, how="inner"
                )
            else:
                # Fall back to positional join if no CustomerID
                _merged = pred_df[[_pred_col]].copy()
                _merged["ChurnStatus"] = test_labels["ChurnStatus"].values[:len(_merged)]

            # Detect positive label
            _pos_hints  = {"yes", "1", "true", "y", "churn"}
            _all_labels = _merged["ChurnStatus"].astype(str).str.lower().unique().tolist()
            _pos_label  = next((l for l in _all_labels if l in _pos_hints), _all_labels[-1])

            _y_true = (_merged["ChurnStatus"].astype(str).str.lower() == _pos_label).astype(int)
            _y_prob = _merged[_pred_col].astype(float)

            _cm_ready = True
        except Exception as e:
            _cm_message = f"Could not compute confusion matrix: {e}"
    elif pred_df is None:
        _cm_message = "prediction.csv not found. Run the pipeline first."
    elif test_labels is None:
        _cm_message = "test.csv not found. Place it in the same folder as dashboard.py."
    else:
        _cm_message = "test.csv has no ChurnStatus column — confusion matrix not available on the hidden test set."

    if not _cm_ready:
        st.info(f"ℹ️ {_cm_message}")
    else:
        # Threshold slider — with flanking min/max labels and clear current value
        st.markdown(
            "<div style='font-size:13px;font-family:\"Space Mono\",monospace;"
            "color:#555;margin-bottom:4px;'>Classification threshold</div>",
            unsafe_allow_html=True,
        )
        _slider_l, _slider_mid, _slider_r = st.columns([0.06, 0.88, 0.06])
        with _slider_l:
            st.markdown(
                "<div style='text-align:center;font-family:\"Space Mono\",monospace;"
                "font-size:13px;font-weight:700;color:#555;padding-top:6px;'>0.1</div>",
                unsafe_allow_html=True,
            )
        with _slider_mid:
            _thresh = st.slider(
                "Classification threshold",
                min_value=0.10, max_value=0.90, value=0.50, step=0.01,
                help="Predict churn (positive) when probability ≥ threshold. "
                     "Lower threshold → more churners caught, more false alarms.",
                label_visibility="collapsed",
            )
        with _slider_r:
            st.markdown(
                "<div style='text-align:center;font-family:\"Space Mono\",monospace;"
                "font-size:13px;font-weight:700;color:#555;padding-top:6px;'>0.9</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            f"<div style='text-align:center;font-family:\"Space Mono\",monospace;"
            f"font-size:15px;font-weight:700;color:#1a1a1a;margin:-4px 0 8px;'>"
            f"Selected: {_thresh:.2f}</div>",
            unsafe_allow_html=True,
        )

        # Context-sensitive hint below slider
        if _thresh < 0.30:
            _thresh_hint = (
                f"Low threshold ({_thresh:.2f}) — model flags anyone with ≥{_thresh:.0%} churn probability. "
                "Catches most real churners but also generates many false alarms. "
                "Best when missing a churner is very costly."
            )
        elif _thresh > 0.70:
            _thresh_hint = (
                f"High threshold ({_thresh:.2f}) — model only flags customers it's highly confident about. "
                "Very few false alarms, but many real churners go undetected. "
                "Best when outreach budget is tight and precision matters more than coverage."
            )
        else:
            _thresh_hint = (
                f"Balanced threshold ({_thresh:.2f}) — moderate trade-off between catching churners "
                "and avoiding false alarms. Adjust left to improve recall, right to improve precision."
            )
        st.markdown(
            f"<div style='font-size:12px;color:#666;margin:-8px 0 12px;line-height:1.5;'>{_thresh_hint}</div>",
            unsafe_allow_html=True,
        )

        _y_pred = (_y_prob >= _thresh).astype(int)

        # Compute the 4 cells
        _tp = int(((_y_pred == 1) & (_y_true == 1)).sum())
        _fp = int(((_y_pred == 1) & (_y_true == 0)).sum())
        _fn = int(((_y_pred == 0) & (_y_true == 1)).sum())
        _tn = int(((_y_pred == 0) & (_y_true == 0)).sum())
        _total = _tp + _fp + _fn + _tn

        # Derived metrics
        _precision  = _tp / (_tp + _fp)  if (_tp + _fp)  > 0 else 0.0
        _recall     = _tp / (_tp + _fn)  if (_tp + _fn)  > 0 else 0.0
        _f1         = (2 * _precision * _recall / (_precision + _recall)
                       if (_precision + _recall) > 0 else 0.0)
        _accuracy   = (_tp + _tn) / _total if _total > 0 else 0.0
        _specificity = _tn / (_tn + _fp)   if (_tn + _fp)  > 0 else 0.0

        # ── Confusion matrix — centred ────────────────────────────────────────
        _pad_l, _cm_centre, _pad_r = st.columns([0.15, 0.70, 0.15])
        with _cm_centre:
            # Build heatmap — annotated 2×2 grid
            # Rows = Actual (Churn, No Churn), Cols = Predicted (Churn, No Churn)
            _z = [[_tp, _fn], [_fp, _tn]]

            # Colour scale: TP/TN green, FP/FN red
            _colorscale = [
                [0.0,  "#fff0f0"],   # low  → light red
                [0.5,  "#f8f9fa"],   # mid  → neutral
                [1.0,  "#f0fdf4"],   # high → light green
            ]

            fig_cm = go.Figure(go.Heatmap(
                z=_z,
                colorscale=_colorscale,
                showscale=False,
                hoverinfo="skip",
                xgap=6, ygap=6,
            ))

            # Cell annotations: large count → label → percentage
            _cell_defs = [
                (0, 0, _tp, "True Positive",  True),
                (1, 0, _fn, "False Negative", False),
                (0, 1, _fp, "False Positive", False),
                (1, 1, _tn, "True Negative",  True),
            ]
            for col_i, row_i, val, name, is_correct in _cell_defs:
                txt_color = "#166534" if is_correct else "#7f1d1d"
                muted     = "#2d6a4f" if is_correct else "#9b2226"
                fig_cm.add_annotation(
                    x=col_i, y=row_i, text=f"<b>{val:,}</b>",
                    showarrow=False,
                    font=dict(size=20, color=txt_color, family="Space Mono"),
                    align="center", yshift=12,
                )
                fig_cm.add_annotation(
                    x=col_i, y=row_i, text=name,
                    showarrow=False,
                    font=dict(size=11, color=muted, family="Space Mono"),
                    align="center", yshift=-4,
                )
                fig_cm.add_annotation(
                    x=col_i, y=row_i,
                    text=f"{val / _total * 100:.1f}% of total dataset",
                    showarrow=False,
                    font=dict(size=10, color="#aaa", family="Inter"),
                    align="center", yshift=-20,
                )

            fig_cm.update_layout(
                **PLOTLY_THEME,
                height=320,
                margin=dict(t=80, b=80, l=130, r=30),
                xaxis=dict(
                    tickvals=[0, 1],
                    ticktext=["Predicted Churn", "Predicted No Churn"],
                    tickfont=dict(size=12, family="Space Mono", color="#166534"),
                    side="top",
                    showgrid=False, zeroline=False,
                ),
                yaxis=dict(
                    tickvals=[0, 1],
                    ticktext=["Actual Churn", "Actual No Churn"],
                    tickfont=dict(size=12, family="Space Mono", color="#166534"),
                    autorange="reversed",
                    showgrid=False, zeroline=False,
                ),
            )
            st.plotly_chart(fig_cm, use_container_width=True,
                            config={"displayModeBar": False})
            st.caption(f"Threshold = {_thresh:.2f} · {_total:,} test samples")

            # Business impact summary below the matrix
            st.markdown(
                f"<div style='font-size:12px;line-height:1.6;padding:10px 14px;"
                f"background:#fffbeb;border-left:3px solid #f59e0b;"
                f"border-radius:6px;margin-top:6px;color:#78350f;'>"
                f"<b>Business impact:</b> "
                f"{_fn:,} actual churners missed (revenue at risk) · "
                f"{_fp:,} non-churners incorrectly flagged (wasted outreach)."
                f"</div>",
                unsafe_allow_html=True,
            )

        # ── Metric cards — Row 1: Accuracy | Recall | Precision ──────────────
        st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

        def _render_metric_card(col, label, raw_val, display_val, desc):
            if raw_val >= 0.70:
                _cls = "stable"
            elif raw_val >= 0.40:
                _cls = "moderate"
            else:
                _cls = "severe"
            _accent = {"stable": "#16a34a", "moderate": "#d97706", "severe": "#dc2626"}[_cls]
            _bar    = int(raw_val * 100)
            with col:
                st.markdown(f"""
                <div class="metric-card" style="
                    margin-bottom:8px;padding:0.75rem 1rem 0.75rem 1rem;
                    border-left:4px solid {_accent};border-radius:0 8px 8px 0;
                    min-height:150px;height:150px;box-sizing:border-box;
                    display:flex;flex-direction:column;justify-content:space-between;
                ">
                    <div class="metric-label" style="font-size:11px !important;line-height:1.2;">{label}</div>
                    <div class="metric-value {_cls}" style="font-size:1.8rem !important;line-height:1.1;">
                        {display_val}
                    </div>
                    <div>
                        <div style="height:4px;background:#e5e7eb;border-radius:99px;
                                    margin:6px 0 5px;overflow:hidden;">
                            <div style="width:{_bar}%;height:100%;
                                        background:{_accent};border-radius:99px;"></div>
                        </div>
                        <div style="font-size:11px;color:#888;
                                    font-family:'Space Mono',monospace;line-height:1.3;
                                    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
                             title="{desc}">{desc}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

        # Row 1 — Accuracy, Recall, Precision
        _r1c1, _r1c2, _r1c3 = st.columns(3)
        _render_metric_card(_r1c1, "Accuracy",  _accuracy,  f"{_accuracy*100:.1f}%",
                            "Overall correct predictions")
        _render_metric_card(_r1c2, "Recall",    _recall,    f"{_recall*100:.1f}%",
                            "Of actual churners, % correctly caught")
        _render_metric_card(_r1c3, "Precision", _precision, f"{_precision*100:.1f}%",
                            "Of predicted churners, % actually churned")

        # Row 2 — Specificity, F1 Score (centred with a spacer)
        _r2c1, _r2c2, _r2c3 = st.columns(3)
        _render_metric_card(_r2c1, "Specificity", _specificity, f"{_specificity*100:.1f}%",
                            "Of non-churners, % correctly identified")
        _render_metric_card(_r2c2, "F1 Score",    _f1,          f"{_f1:.4f}",
                            "Harmonic mean of precision and recall")
        # _r2c3 intentionally left empty
