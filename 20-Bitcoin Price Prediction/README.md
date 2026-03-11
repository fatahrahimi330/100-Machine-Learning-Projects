# Bitcoin Price Prediction

This project builds a **binary classification** pipeline to predict whether Bitcoin's next-day closing price will go **up (1)** or **down/no change (0)** using:

- Logistic Regression
- Support Vector Classifier (SVC)
- XGBoost Classifier (XGBClassifier)

The full workflow is implemented in the notebook: `bitcoin_price_prediction.ipynb`.

---

## Project Structure

- `bitcoin_price_prediction.ipynb` — end-to-end notebook (EDA, preprocessing, training, evaluation, GridSearchCV)
- `bitcoin.csv` — historical Bitcoin OHLCV dataset

---

## Workflow Summary

1. Load dataset
2. Perform preprocessing and feature engineering
   - Parse date features (`Year`, `Month`, `Day`)
   - Remove redundant columns
   - Build target label:
     - `Target = 1` if next day `Close` > current day `Close`
     - else `Target = 0`
3. Split train/test data
4. Scale features with `StandardScaler`
5. Train baseline models
6. Evaluate models using ROC-AUC and confusion matrices
7. Tune hyperparameters with **GridSearchCV** + **StratifiedKFold**

---

## Requirements

Install dependencies before running the notebook:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

---

## How to Run

1. Open `bitcoin_price_prediction.ipynb` in Jupyter or VS Code.
2. Run cells from top to bottom.
3. Check the final GridSearchCV section to see the best hyperparameters and model ranking.

---

## Notes

- This is a classification task, not direct price regression.
- Current metric focus is ROC-AUC.
- You can improve performance by:
  - adding lag/technical indicator features,
  - using time-series-aware validation,
  - testing larger hyperparameter grids.

---

## Author

Fatah Rahimi
