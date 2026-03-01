# Classify Handwritten Digits with TensorFlow

A beginner-friendly deep learning project that trains a neural network on the MNIST dataset to classify handwritten digits (0–9).

## 📌 Project Overview

This project uses `TensorFlow/Keras` to:
- load and preprocess the MNIST dataset,
- build a fully connected neural network,
- train and evaluate the model,
- make predictions,
- save and reload the trained model.

The full implementation is in the notebook:
- [classify_handwritten_digits.ipynb](classify_handwritten_digits.ipynb)

## 🧠 Dataset

**MNIST** is a standard benchmark dataset for image classification:
- 70,000 grayscale images of handwritten digits
- Image size: `28 x 28`
- Classes: `10` (`0` to `9`)
- Split:
  - Training set: `60,000`
  - Test set: `10,000`

Loaded directly via:
- `tf.keras.datasets.mnist.load_data()`

## 🏗️ Model Architecture

The model is a simple feed-forward neural network:
1. `Flatten(input_shape=(28, 28))`
2. `Dense(128, activation='relu')`
3. `Dense(128, activation='relu')`
4. `Dense(10, activation='softmax')`

Compile settings:
- Optimizer: `adam`
- Loss: `sparse_categorical_crossentropy`
- Metric: `accuracy`

## 🔄 Workflow in the Notebook

1. Import required libraries (`numpy`, `matplotlib`, `tensorflow`, etc.)
2. Load and normalize MNIST data
3. Visualize sample digits
4. Build and train the model (`epochs=3`)
5. Evaluate on test data
6. Run predictions and compare with true labels
7. Save model as `handwritten.h5`
8. Reload saved model and validate performance
9. Compute classification accuracy with `sklearn.metrics.accuracy_score`

## ✅ Accuracy Notes

When using `accuracy_score`, predictions from softmax are probabilities. Convert them to class labels first:

- `y_pred_labels = np.argmax(y_pred, axis=1)`
- `accuracy_score(y_test, y_pred_labels)`

This avoids the error:
> `ValueError: Classification metrics can't handle a mix of multiclass and continuous-multioutput targets`

## 🛠️ Requirements

Install Python dependencies:

- tensorflow
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- jupyter

## 🚀 How to Run

1. Open the project folder.
2. Launch Jupyter Notebook (or open in VS Code Notebook view).
3. Open [classify_handwritten_digits.ipynb](classify_handwritten_digits.ipynb).
4. Run all cells from top to bottom.

## 📁 Project Structure

- [classify_handwritten_digits.ipynb](classify_handwritten_digits.ipynb) — complete training, evaluation, prediction, and model persistence workflow
- `handwritten.h5` — saved trained model (generated after running notebook)

## 📈 Expected Result

With this architecture and preprocessing, the model typically reaches high test accuracy (often around ~97%+ depending on runtime/session).

## 🔮 Possible Improvements

- Add `Dropout` for regularization
- Use `EarlyStopping`
- Switch to a CNN for higher accuracy
- Add confusion matrix and classification report
- Save plots and metrics automatically

## 👤 Author

Project from: **100+ Machine Learning Projects**
