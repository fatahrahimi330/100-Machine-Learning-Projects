# Tesla Stock Price Prediction (Classification)

This project builds a machine learning pipeline to predict whether Tesla stock will move **up (1)** or **down (0)** on the next trading day.

The implementation is provided in a Jupyter notebook using `scikit-learn`, `xgboost`, `pandas`, `numpy`, `matplotlib`, and `seaborn`.

## Project Objective

Given historical Tesla stock data, train multiple classification models and compare their performance using ROC-AUC on training and validation sets.

---

## Dataset

- **File:** `Tesla.csv`
- **Primary columns used:** `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`

### Target Definition

The binary target is created as:

- `1` if next day `Close` > current day `Close`
- `0` otherwise

Formally:

\[
\text{target}_t =
\begin{cases}
1, & \text{if } Close_{t+1} > Close_t \\
0, & \text{otherwise}
\end{cases}
\]

---

## Workflow

1. **Import libraries**
2. **Load data**
3. **EDA & preprocessing**
   - Null checks and summary stats
   - Distribution and box plots
   - Correlation analysis
4. **Feature engineering**
   - Extract `Day`, `Month`, `Year` from `Date`
   - Add `is_quarter_end`
   - Add price-difference features:
     - `close-open = Close - Open`
     - `low-high = Low - High`
   - Drop less useful/redundant columns
5. **Train-test split** (`test_size=0.1`, `random_state=42`)
6. **Feature scaling** with `StandardScaler`
7. **Model training**
8. **Evaluation**
   - ROC-AUC on train and validation sets
   - Confusion matrix
   - Comparison bar chart (training vs validation)
   - Scatter of actual vs predicted labels

---

## Models Used

- `LogisticRegression`
- `SVC` with polynomial kernel (`probability=True`)
- `XGBClassifier`

---

## Evaluation Metrics

- **Primary:** ROC-AUC (`roc_auc_score`)
- **Additional:** Confusion Matrix

---

## Project Structure

- `Tesla_stock_price_prediction.ipynb` → full pipeline and visualizations
- `Tesla.csv` → dataset
- `README.md` → project documentation

---

## Requirements

Install dependencies before running:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

---

## How to Run

1. Open `Tesla_stock_price_prediction.ipynb`.
2. Run cells from top to bottom.
3. Ensure `Tesla.csv` is in the same folder as the notebook.
4. Review:
   - Model ROC-AUC values
   - Combined bar chart of training/validation scores
   - Confusion matrix
   - Actual vs predicted scatter plot

---

## Notes

- This project predicts **direction**, not exact future price.
- Results can vary with different random seeds, feature sets, and hyperparameters.
- To improve performance, consider:
  - Time-series-aware split (instead of random split)
  - Hyperparameter tuning (`GridSearchCV` / `RandomizedSearchCV`)
  - Additional technical indicators (RSI, MACD, moving averages)

---

## Disclaimer

This project is for educational purposes only and not financial advice.
