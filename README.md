# NAISC 2026: SingTel Customer Churn Prediction

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-green)](https://lightgbm.readthedocs.io/)
[![Polars](https://img.shields.io/badge/Polars-1.x-orange)](https://pola.rs/)

**Team Green Beans** Singapore University of Social Sciences | NAISC 2026 SingTel Challenge

---

## The Problem

SingTel loses customers every month. Some leave because of price, some because of service, and some because life happens. But here is the hard part: by the time you know who left, it is already too late. The real challenge is spotting who *will* leave next month, before they actually go.

The NAISC 2026 challenge gave us a dataset of 70,430 customers from January to October 2025, and asked us to predict churn for November and December. Sounds straightforward — train a model, make predictions, done.

But there is a catch.

Customer behaviour changes month to month. A model trained on January-October data assumes the world stays the same. It does not. November and December bring holiday spending, different billing patterns, and seasonal shifts. When the data itself changes under your model, performance drops. This is called **data drift**, and fixing it was the core of our challenge.

---

## What We Built

An end-to-end pipeline that does five things automatically:

1. **Classifies** every column by its statistical type (numerical, categorical, sparse, etc.)
2. **Detects** distributional drift using tailored statistical tests per column type
3. **Ranks** drifted features by how recent and how severe the shift is
4. **Fixes** each drifted column with an appropriate mitigation
5. **Trains** a LightGBM model and outputs churn predictions

Everything runs on CPU only, handles up to 10 million rows and 500 features, and completes in under 10 minutes.

![Validation Summary](assets/validation_summary.png)

---

## How It Works

### Stage 1 — Classify Every Column

You cannot test every column the same way. Monthly charges are numerical, contract type is categorical, some columns are mostly empty (sparse), and some have so many unique values they are basically identifiers. The pipeline automatically sorts each of the 45 columns into the right bucket using a single parallel pass through the data.

### Stage 2 — Detect Drift in Three Phases

**Phase A (Fast pre-screen):** Before running any expensive statistical tests, we compare summary statistics (mean, IQR, frequency distribution) between train and test. Columns that look stable skip the deeper checks entirely. This saves time.

**Phase B (Deep statistical tests):** Flagged columns get a proper statistical test matched to their type:
- Numerical columns → Kolmogorov-Smirnov test (compares distribution shape)
- Categorical columns → PSI + chi-squared (both must agree before we flag it)
- High-cardinality columns → Top-K binning + Jensen-Shannon divergence
- Sparse columns → Two-proportion Z-test + conditional KS

**Phase C (Sub-population checks):** Even if a column passes Phase B, we check whether individual categories or time segments shifted. This catches cases where drift in one customer segment is masked by opposite movement in another.

### One Real Story: The TransactionMode Bug

Early in development, the pipeline flagged a feature called TransactionMode as severely drifted with a PSI score of **2.72** (anything above 0.50 is already considered severe). After some digging, we found the cause:

In the training data, the same payment method was recorded two different ways: "Bank Withdrawal" and "bank-withdrawal". To the model, these looked like two separate categories. In the test data, this formatting inconsistency had been corrected. The apparent drift was not a change in customer behaviour at all — it was a data entry quirk.

The pipeline resolved this automatically by normalising all category labels (lowercase, strip, replace hyphens) *before* running any statistical test. After normalisation, the PSI dropped from 2.72 to **0.006**. This feature does not appear in our final drift table because it was never actually drifted.

This story shaped two key design decisions in our pipeline: always clean the text before you test, and always verify that your statistical flags reflect real shifts, not formatting artefacts.

![Drift Detection Summary](assets/chart_categorical.png)

### Stage 3 — Rank by Urgency with Recency Weighting

Not all drift is equally urgent. A feature that drifted six months ago matters less than one that drifted last month. Stage 3 applies exponential decay weighting — more recent months get higher weight — and multiplies each drifted feature's score by its severity level (severe = 3x, moderate = 2x, mild = 1x).

This produces a ranked list so the most urgent fixes are applied first.

### Stage 4 — Fix Each Feature Appropriately

| Drift Type | What We Apply | Why |
|---|---|---|
| Numerical (severe) | Quantile binning (10 bins) | Converts a shifted distribution into stable ordinal ranks |
| Categorical | Laplace-smoothed target encoding (m=10) | Replaces labels with smoothed churn rates, unseen categories fall back to global average |
| High-cardinality | Frequency encoding | Maps each value to how many training rows share it |
| Sparse | Binarisation | Collapses to a simple 0/1 presence indicator |

Every fix is calculated from training data only. No test data leaks into the fitting process.

![Numerical Drift Mitigation](assets/chart_numerical_scaling.png)

### Stage 5 — Train and Predict

A LightGBM classifier trains on the mitigated data and outputs churn probability predictions. Hyperparameters are locked per competition rules — no tuning after submission freeze.

---

## Results

### The Numbers

| Metric | Before Mitigation | After Mitigation | Improvement |
|---|---|---|---|
| Test AU-PRC | **0.5082** | **0.7774** | **+0.2692** |
| Train AU-PRC | 0.9249 | 0.9249 | — |
| Train-test gap | 0.4167 | 0.1475 | Reduced by 64.6% |

The baseline model (no drift mitigation) scored 0.5082 on the test set — barely better than guessing. After applying our pipeline, the test AU-PRC jumped to **0.7774**, a **52.7% relative improvement**. The train-test gap narrowed from 0.4167 to 0.1475, meaning the model's performance on new data is much closer to its training performance.

### What We Found

**Eight features** showed confirmed drift between the training and test periods:

| Feature | Type | Severity | Fix |
|---|---|---|---|
| DigitalInvoicing | Categorical | Severe | Target encoding |
| AvgMonthlyLongDistanceCharges | Numerical | Severe | Quantile binning |
| MonthlyCharge | Numerical | Moderate | Quantile binning |
| NumberofReferrals | Categorical | Mild | Target encoding |
| ReferredaFriend | Categorical | Mild | Target encoding |
| TotalLongDistanceCharges | Numerical | Mild | Quantile binning |
| PrioritySupport | Categorical | Mild | Target encoding |
| Contract | Categorical | Mild | Target encoding |

![Seasonal Binning](assets/chart_seasonal_binning.png)

### What We Tried That Did Not Work

We tested several alternative approaches:

- **Sliding window (last 4 months only):** AU-PRC dropped to 0.7265. Removing older data weakened the model.
- **Frequency encoding instead of target encoding:** Slightly worse at 0.7719.
- **Robust scaling instead of quantile binning:** Lower at 0.7726.
- **Dropping all drifted features:** Produced 0.8689 AU-PRC, but increased false negatives at the 0.5 threshold. Removing drifted features hurts recall more than it helps precision.

The takeaway: **drifted features still carry predictive signal.** Dropping them or using the wrong mitigation costs performance. The right fix applied to each feature type is what works.

### Feature Importance vs Drift

The chart below shows each feature's predictive importance (from LightGBM) plotted against its recency-weighted drift score. High-importance, high-drift features are the ones that need attention first.

![Importance vs Drift](assets/importance_vs_drift.png)

---

## Getting Started

### Prerequisites

Python 3.12+.

### Install

```bash
pip install -r requirements.txt
```

### Run the Pipeline

```bash
python src/main.py --train_data_filepath train.csv --test_data_filepath test.csv
```

This executes all 5 stages end-to-end. Output (`prediction.csv` and `model.joblib`) is written to the project root.

### Run the Dashboard

The dashboard auto-launches after the pipeline completes. To skip:

```bash
python src/main.py --train_data_filepath train.csv --test_data_filepath test.csv --skip_dashboard
```

Run it independently:

```bash
python -m streamlit run src/dashboard.py
```

---

## Repository Structure

```
.
├── assets/                   # Charts and screenshots
├── src/
│   ├── main.py               # Pipeline orchestrator
│   ├── dashboard.py           # Streamlit dashboard
│   ├── drift/
│   │   ├── detector.py        # Three-phase drift detection
│   │   ├── mitigator.py       # Mitigation strategies
│   │   ├── ranker.py          # Recency-weighted ranking
│   │   └── report.py          # Data types and contracts
│   └── schema/
│       └── classifier.py      # Dynamic schema classifier
├── prediction.csv             # Competition predictions
├── model.joblib               # Trained LightGBM model
├── report.pdf                 # Full technical report
├── requirements.txt
└── README.md
```

---

## What We Learned

This project taught us that drift detection is not just about running statistical tests. Three things mattered most:

1. **Normalise before you test.** A PSI of 2.72 turned into 0.006 just by cleaning text formatting. The order of operations matters.
2. **Use the right test for the right column type.** Applying KS to categorical data or PSI to everything would produce misleading results.
3. **Do not drop drifted features blindly.** They still carry signal. The right encoding preserves that signal despite the distribution shift.

---

## Built With

- **Polars** — parallel columnar processing (no pandas until the final handoff)
- **LightGBM** — gradient boosting with locked hyperparameters
- **SciPy** — statistical tests (KS, chi-squared, Z-test)
- **Streamlit + Plotly** — interactive dashboard
- **joblib** — model serialisation

---

## Team

**Team Green Beans** — Singapore University of Social Sciences, Singapore
