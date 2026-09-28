# Bank Cash Optimization

Half-daily cash demand forecasting for 15 bank branches (Arcana Info internship project).

The goal is to predict how much cash each branch will pay out in the morning and afternoon of each day, so vault replenishment can be planned ahead: no shortages at the counter, and no large amounts of idle cash.

## Results

Final model: a blend of XGBoost and LightGBM trained on `log1p(debit)`, with hyperparameters tuned by Optuna. It is evaluated with 5-fold rolling-origin cross-validation and a hold-out of the last 20% of dates (3,457 rows, 29 Oct 2025 to 4 Apr 2026), always against the raw, uncapped actuals.

| Debit target | Cross-validation | Hold-out |
|---|---|---|
| R² | 0.581 ± 0.066 | 0.401 |
| MAE | 9.77M ± 0.66M PKR | 11.46M PKR |
| WMAPE | 39.5% ± 2.4% | 44.9% |

WMAPE is the main error metric. MAPE is misleading on this data because a few half-days have near-zero demand; the notebook explains this in Section 6.

## Repository layout

```
Bank_Cash_Optimization_Workflow.ipynb   main notebook: data, features, models, per-branch analysis, SHAP, forecast
src/
  v3_pipeline.py            trains the ensemble and writes models/v3/
  prophet_pipeline.py       per-branch Prophet models (benchmark shown in the dashboard)
  forecast_pipeline.py      30-day recursive forecast from the saved models
  generate_shap.py          SHAP values for the dashboard
  update_dashboard_data.py  refreshes the dashboard's input files after retraining
  dashboard.py              Streamlit dashboard
  metrics_utils.py          MAPE, SMAPE and WMAPE
experiments/
  adf_stationarity_test.py     ADF stationarity test, overall and per branch
  differencing_experiment.py   levels vs differenced target
notebooks/                  earlier step-by-step notebooks (01-07), kept for reference
models/                     saved models and the files the dashboard reads
reports/figures/            figures exported by the earlier notebooks
data/                       local data, not committed
```

## Running

The notebook is self-contained and is meant to be run in Google Colab: it installs its packages, downloads the dataset from Google Drive and trains everything from scratch.

To run the scripts locally, work from the repository root:

```bash
pip install -r requirements.txt

# 1. Run Sections 1-4 of the notebook once; this writes data/processed/half_daily_features.csv
# 2. Train and save the models
python src/v3_pipeline.py
# 3. Refresh the dashboard inputs and start the dashboard
python src/update_dashboard_data.py
streamlit run src/dashboard.py
```

The dashboard also reads `data/processed/daily_full.csv`, which comes from `notebooks/03_Model_Preparation.ipynb`.

## Known issues

- `src/update_dashboard_data.py` and `src/generate_shap.py` still pick the test set by row position (`int(len(df) * 0.8)`), which the notebook replaced with a date-based split. The per-branch report and SHAP values shown in the dashboard therefore come from a 164-row test set instead of the 3,457-row hold-out.
- The 30-day forecast in the notebook (Section 9) sizes its uncertainty bands from per-branch MAPE, which makes the bands very wide. The WMAPE-based bands from Section 7.0 would be a better fit.
