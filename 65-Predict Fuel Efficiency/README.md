# Predict Fuel Efficiency using ANN in TensorFlow

This project demonstrates how to build and train an Artificial Neural Network (ANN) using TensorFlow and Keras to predict a vehicle's fuel efficiency (Measured in Miles Per Gallon - MPG) based on its technical specifications.

## Project Overview

Fuel efficiency prediction is a classic regression problem in machine learning. By analyzing attributes such as the number of cylinders, horsepower, weight, and model year, we can estimate the MPG of a vehicle with high accuracy. This project explores data preprocessing, feature selection, and the implementation of a deep learning model for regression.

## Dataset

The project uses the **Auto-MPG dataset**, which contains the following features:

- **MPG**: Miles Per Gallon (Target variable)
- **Cylinders**: Multi-valued discrete
- **Displacement**: Continuous (Dropped in this project due to high correlation with weight)
- **Horsepower**: Continuous
- **Weight**: Continuous
- **Acceleration**: Continuous
- **Model Year**: Multi-valued discrete
- **Origin**: Multi-valued discrete
- **Car Name**: String (Unique for each model)

## Model Architecture

The model is built using a sequential ANN architecture:

1.  **Input Layer**: Dense layer with 256 units and ReLU activation.
2.  **Batch Normalization**: To accelerate training and provide regularization.
3.  **Hidden Layer**: Dense layer with 256 units and ReLU activation.
4.  **Dropout**: 30% dropout rate to prevent overfitting.
5.  **Batch Normalization**: Second normalization layer.
6.  **Output Layer**: Dense layer with 1 unit and ReLU activation (Targeting continuous MPG values).

## Training Configuration

- **Optimizer**: Adam
- **Loss Function**: Mean Absolute Error (MAE)
- **Metrics**: Mean Absolute Percentage Error (MAPE)
- **Epochs**: 50
- **Validation Split**: 20% of the training data

## Installation & Usage

### Prerequisites

- Python 3.x
- TensorFlow 2.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### Running the Project

1.  Clone the repository or download the project files.
2.  Ensure `auto-mpg.csv` is in the same directory as the notebook.
3.  Open `PredictFuelEfficiency.ipynb` in Jupyter Notebook or Google Colab.
4.  Run all cells to execute the data preprocessing, training, and evaluation.

## Results

The model achieves performance reflected in the MAE and MAPE metrics. Training history plots for loss and accuracy provide insights into the convergence of the model during 50 epochs.

---
*This project is part of the 100+ Machine Learning Projects series.*
