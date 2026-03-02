# Handwritten Digits Recognition using Neural Network

A machine learning project that recognizes handwritten digits (0-9) using a neural network built with TensorFlow and Keras.

## Project Overview

This project implements a digit recognition system that can identify handwritten numbers from images. The model is trained on a dataset of 28x28 pixel grayscale images and achieves high accuracy in classifying digits.

## Features

- Data preprocessing and normalization
- Neural network architecture with multiple dense layers
- Training with validation split
- Model evaluation with accuracy metrics
- Visualization of predictions vs actual labels
- Training history visualization

## Requirements

```
numpy
pandas
matplotlib
seaborn
tensorflow
scikit-learn
```

## Installation

1. Clone the repository or download the project files
2. Install the required dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn tensorflow scikit-learn
   ```

## Dataset

The project uses a CSV dataset containing:
- 28x28 pixel images (784 features)
- Labels for digits 0-9
- Training data in `Train.csv` format

The dataset is automatically downloaded from the GitHub repository when running the notebook.

## Model Architecture

The neural network consists of:
- **Input Layer**: 28x28x1 (grayscale images)
- **Flatten Layer**: Converts 2D images to 1D vectors
- **Hidden Layer 1**: 128 neurons with ReLU activation
- **Hidden Layer 2**: 64 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation (for 10 digit classes)

**Optimizer**: Adam  
**Loss Function**: Categorical Crossentropy

## Usage

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook handwritten_recognition.ipynb
   ```

2. Run all cells in sequence to:
   - Import libraries
   - Load and preprocess the dataset
   - Build the neural network model
   - Train the model (10 epochs)
   - Evaluate performance
   - Visualize predictions

## Model Training

- **Epochs**: 10
- **Batch Size**: 32
- **Train/Test Split**: 80/20
- **Validation**: Performed on test set during training

## Results

The model achieves high validation accuracy on the test set. Training and validation accuracy curves are plotted to visualize the model's learning progress.

## Project Structure

```
├── handwritten_recognition.ipynb  # Main Jupyter notebook
├── Train.csv                       # Dataset (downloaded automatically)
└── README.md                       # Project documentation
```

## Key Steps

1. **Data Loading**: Import dataset from CSV file
2. **Preprocessing**: 
   - Normalize pixel values (0-255 → 0-1)
   - Reshape images to 28x28x1
   - One-hot encode labels
3. **Model Building**: Create sequential neural network
4. **Training**: Fit model on training data
5. **Evaluation**: Assess performance on test data
6. **Prediction**: Test model on unseen examples

## Visualization

The notebook includes:
- Training vs validation accuracy plots
- Sample predictions with actual labels
- Visual comparison of predicted and actual digits

## Future Improvements

- Implement Convolutional Neural Networks (CNN) for better accuracy
- Add data augmentation techniques
- Experiment with different architectures and hyperparameters
- Deploy the model as a web application
- Add real-time drawing interface for digit recognition

## License

This project is part of the 100+ Machine Learning Projects collection.

## Author

Fatah Rahimi

## Acknowledgments

- Dataset sourced from the 100+ Machine Learning Projects repository
- Built with TensorFlow and Keras frameworks
