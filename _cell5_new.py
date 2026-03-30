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
