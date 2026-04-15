# Team Green beans

NAISC Singtel 2026 drift detection and mitigation pipeline.

## Run Instructions

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline:

```bash
python ./src/main.py --train_data_filepath train.csv --test_data_filepath test.csv
```

Notes:

- The pipeline is built for Python 3.12 and CPU execution.
- LightGBM hyperparameters are fixed as required by the challenge.
- `prediction.csv` and `model.joblib` are written to the repository root.
- Dashboard feed files are exported after each run.

## Optional Dashboard

By default, the pipeline auto-launches Streamlit dashboard after completion.

To skip dashboard launch:

```bash
python ./src/main.py --train_data_filepath train.csv --test_data_filepath test.csv --skip_dashboard
```

To run dashboard manually:

```bash
python -m streamlit run ./src/dashboard.py
```

## Repository Layout

```
.
├── src/
│   ├── main.py
│   └── dashboard.py
├── drift/
├── schema/
├── prediction.csv
├── model.joblib
├── report.pdf
├── .gitignore
├── requirements.txt
└── README.md
```

## Team Details

Team name: Team Green beans
