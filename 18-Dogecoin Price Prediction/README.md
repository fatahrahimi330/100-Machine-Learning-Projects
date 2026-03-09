# Dogecoin Price Prediction (RandomForestRegressor)

This project predicts Dogecoin closing price using a `RandomForestRegressor` model in scikit-learn.
The implementation is in a Jupyter notebook:
- [dogecoin_price_prediction.ipynb](dogecoin_price_prediction.ipynb)

## Dataset
- File: [DOGE-USD.csv](DOGE-USD.csv)
- Main columns used:
  - `Date`
  - `Close`
  - `Volume`
  - `High`
  - `Low`

## Workflow
1. Import libraries (`numpy`, `pandas`, `matplotlib`, `seaborn`, `sklearn`).
2. Load dataset and inspect correlations/statistics.
3. Convert `Date` to datetime and set it as index.
4. Engineer features:
   - `gap = (High - Low) * Volume`
   - `a = High / Low`
   - `b = (High / Low) * Volume`
5. Build modeling table with features: `Close`, `Volume`, `gap`, `a`, `b`.
6. Use the last 30 rows, split into:
   - Train: first 11 rows
   - Test: last 19 rows
7. Train `RandomForestRegressor`.
8. Evaluate with MAE, MSE, RMSE, and R².
9. Visualize actual vs predicted values and feature importances.

## Model Settings
```python
RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=8,
    min_samples_split=2,
    min_samples_leaf=1
)
```

## Current Results
From the latest notebook run:
- Train rows used: `11`
- Test rows used: `18`
- MAE: `0.005720`
- MSE: `0.000040`
- RMSE: `0.006329`
- R²: `-0.601202`

## How to Run
1. Open [dogecoin_price_prediction.ipynb](dogecoin_price_prediction.ipynb).
2. Run cells from top to bottom.
3. Review printed metrics, prediction table, plot, and feature importance output.

## Requirements
Install dependencies before running:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

## Notes
- Because this is time-series data, keeping chronological order in train/test split is important.
- The negative R² suggests the model can be improved (for example, with lag features and a larger training window).
