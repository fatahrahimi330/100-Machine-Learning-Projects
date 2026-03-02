# Recognizing Handwritten Digits

A machine learning project that classifies handwritten digits (0–9) using a **Multi-Layer Perceptron (MLP)** classifier from **Scikit-learn**.

This project uses the built-in `sklearn.datasets.load_digits` dataset and walks through:
- loading and exploring the dataset,
- visualizing sample digit images,
- training an MLP neural network,
- evaluating model performance.

---

## 📌 Project Overview

Handwritten digit recognition is a classic supervised classification task in machine learning.  
In this notebook, each digit image is represented as an 8×8 grayscale image, flattened into a 64-feature vector.

The workflow includes:
1. Importing required libraries
2. Loading the digits dataset
3. Visualizing sample images
4. Splitting training and testing data
5. Building and training an MLP model
6. Evaluating prediction accuracy

---

## 🧠 Model Details

The model used is `MLPClassifier` with:
- **Hidden layers:** `(15,)`
- **Activation:** `logistic`
- **Solver:** `sgd`
- **Learning rate init:** `0.1`
- **Regularization (`alpha`):** `1e-4`
- **Tolerance (`tol`):** `1e-4`
- **Random state:** `1`
- **Verbose:** `True`

The notebook also plots the **loss curve** to visualize convergence across iterations.

---

## 📂 Project Structure

```text
6-Recognizing Handwritten Digits/
├── recognizing_handwritten_digits.ipynb
└── README.md
```

---

## ⚙️ Requirements

Install Python packages:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `jupyter` (or run notebook in VS Code/Jupyter environment)

You can install dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

---

## ▶️ How to Run

1. Open the project folder.
2. Launch Jupyter Notebook (or open the notebook in VS Code).
3. Open:
   - `recognizing_handwritten_digits.ipynb`
4. Run all cells in order.
5. Check:
   - digit visualizations,
   - training logs,
   - loss curve,
   - final accuracy output.

---

## 📊 Evaluation

The notebook evaluates the model using:
- `accuracy_score(y_test, y_pred)`

Output example format:

```text
Model Accuracy: 0.xxxx
```

(Exact value may vary slightly depending on environment and execution.)

---

## 🚀 Possible Improvements

You can improve this project by:
- using `train_test_split` with shuffling and stratification,
- scaling features (`StandardScaler`) before MLP training,
- tuning hyperparameters with `GridSearchCV` or `RandomizedSearchCV`,
- adding a confusion matrix and classification report,
- trying other models (SVM, Random Forest, CNN with TensorFlow/PyTorch).

---

## 🎯 Learning Outcomes

By completing this project, you practice:
- image-based classification basics,
- neural network training with Scikit-learn,
- model evaluation for multiclass problems,
- plotting and interpreting training loss curves.

---

## 📄 License

This project is for educational purposes.  
Feel free to use and adapt it for learning and experimentation.
