# Loan Approval Prediction

A machine learning project that predicts loan approval status using multiple Scikit-learn classification algorithms.

## Project Structure

- [loan_approval_prediction.ipynb](loan_approval_prediction.ipynb) — end-to-end notebook (EDA, preprocessing, training, evaluation)
- [LoanApprovalPrediction.csv](LoanApprovalPrediction.csv) — dataset

## Features

- Exploratory data analysis (EDA)
- Label encoding for categorical features
- Missing-value handling
- Train/test split
- Model training with 4 algorithms:
  - Random Forest Classifier
  - K-Nearest Neighbors (KNN)
  - Support Vector Classifier (SVC)
  - Logistic Regression
- Performance evaluation with:
  - Accuracy score
  - Confusion matrix visualization

## Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## How to Run

1. Clone the repository.
2. Open [loan_approval_prediction.ipynb](loan_approval_prediction.ipynb) in Jupyter or VS Code.
3. Install dependencies:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```

4. Run notebook cells from top to bottom.

## Workflow

1. Import libraries
2. Load dataset
3. Preprocess data
4. Train models
5. Make predictions
6. Evaluate models with confusion matrices

## Notes

- The notebook includes a download step (`wget`) for the dataset, but the CSV is already included in this folder.
- You can tune model hyperparameters to improve performance.

## License

This project is for educational purposes.