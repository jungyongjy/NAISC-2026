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
