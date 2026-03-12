# Online Payment Fraud Detection

Machine learning project for detecting fraudulent online payment transactions using multiple classification algorithms.

## Project Overview

This project builds and compares three models on a large transaction dataset:

- `LogisticRegression`
- `RandomForestClassifier`
- `XGBClassifier` (XGBoost)

The workflow includes:

1. Data loading
2. Exploratory Data Analysis (EDA)
3. Preprocessing
4. Train/test split
5. Feature scaling
6. Model training and evaluation
7. Visualization of training and test ROC-AUC scores
8. Confusion matrix and sample predictions

---

## Project Structure

- [online_payment_fraud_detection.ipynb](online_payment_fraud_detection.ipynb) — Full notebook with EDA, training, and evaluation
- [new_data.csv](new_data.csv) — Transaction dataset used in the notebook

---

## Dataset

The dataset contains **6,362,620 rows** and **11 columns**.

Target variable:
- `isFraud` (`0` = non-fraud, `1` = fraud)

Class distribution:
- Non-fraud (`0`): **6,354,407**
- Fraud (`1`): **8,213**

This is a highly imbalanced classification problem.

Key columns used in modeling:
- `step`
- `amount`
- `oldbalanceOrg`
- `newbalanceOrig`
- `oldbalanceDest`
- `newbalanceDest`
- `isFlaggedFraud`

Columns dropped before training:
- `type`
- `nameOrig`
- `nameDest`

---

## Installation

Use Python 3.9+ (recommended 3.10+).

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

---

## How to Run

1. Open the notebook: [online_payment_fraud_detection.ipynb](online_payment_fraud_detection.ipynb)
2. Run all cells from top to bottom.
3. Review:
   - EDA plots
   - model training output
   - training vs test accuracy plots
   - confusion matrix

---

## Modeling Details

### Data split
- `train_test_split(test_size=0.1, random_state=42)`

### Scaling
- `StandardScaler` is applied to `X_train` and `X_test`.

### Models (current notebook configuration)
- `LogisticRegression(max_iter=1000, random_state=42)`
- `RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)`
- `XGBClassifier(n_estimators=50, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42, n_jobs=-1)`

### Metric
- ROC-AUC score is used for both train and test evaluation.

---

## Results

### Training ROC-AUC
- Logistic Regression: **0.5962**
- Random Forest: **0.7334**
- XGBoost: **0.7650**

### Test ROC-AUC
- Logistic Regression: **0.5900**
- Random Forest: **0.7375**
- XGBoost: **0.7766**

Best performing model in this run: **XGBoost**.

---

## Visual Outputs

The notebook generates:

- Transaction type distribution plot
- Fraud vs non-fraud distribution plot
- Distribution plot for `step`
- Correlation heatmap
- Training accuracy line plot
- Test accuracy line plot
- Confusion matrix visualization

---

## Notes and Limitations

- The dataset is severely imbalanced, so accuracy alone is not reliable.
- ROC-AUC is used, but additional metrics (`precision`, `recall`, `F1`, `PR-AUC`) are recommended.
- Further improvements can include:
  - class weighting / cost-sensitive learning
  - threshold tuning
  - stratified cross-validation
  - hyperparameter optimization

---

## Future Improvements

- Add a proper evaluation report (`classification_report`, precision-recall curve)
- Save trained model artifacts
- Build a small inference script/API for transaction-level fraud prediction
- Add experiment tracking (for reproducibility)

---

## Author

Project prepared as part of the **100+ Machine Learning Projects** series.