# Detecting Spam Emails Using TensorFlow

A deep learning project that classifies emails as **Spam** or **Ham (Not Spam)** using TensorFlow and LSTM neural networks.

## Overview

This project builds a machine learning model to automatically detect spam emails by analyzing email text content. The model uses a sequential deep learning architecture with embedding and LSTM layers to capture patterns in email text and make predictions.

## Dataset

- **Source**: `spam_ham_dataset.csv`
- **Size**: 5,171 emails
- **Features**: Email text and label (spam/ham)
- **Classes**: Binary classification (Spam vs. Ham)

## Project Structure

The notebook follows these main steps:

### 1. **Data Loading & Exploration**
   - Load dataset from CSV
   - Explore dataset structure and distribution
   - Visualize class imbalance

### 2. **Data Balancing**
   - Address class imbalance using undersampling
   - Balance spam and ham email counts
   - Ensures fair model training

### 3. **Text Preprocessing & Cleaning**
   - Remove "Subject" prefix from text
   - Remove punctuation
   - Remove stopwords using Gensim
   - Visualize text patterns using WordCloud

### 4. **Tokenization & Padding**
   - Convert text to sequences of integers
   - Standardize sequence length (max 100 tokens)
   - Train-test split (80-20)

### 5. **Model Architecture**
   - **Embedding Layer**: Converts words to dense vectors
   - **LSTM Layer**: Captures sequential patterns in text
   - **Dense Layer**: Extracts learned features
   - **Output Layer**: Binary classification (sigmoid activation)

### 6. **Model Training**
   - Optimizer: Adam
   - Loss Function: Binary Cross-Entropy
   - Callbacks: Early Stopping & Learning Rate Reduction
   - Epochs: Up to 20

### 7. **Evaluation & Visualization**
   - Test accuracy and loss metrics
   - Training/validation accuracy curves
   - Training/validation loss curves

## Requirements

```
numpy
pandas
matplotlib
seaborn
tensorflow
gensim
wordcloud
scikit-learn
```

## Installation

1. Install required packages:
```bash
pip install numpy pandas matplotlib seaborn tensorflow gensim wordcloud scikit-learn
```

2. The dataset is automatically downloaded from GitHub in the notebook

## Usage

1. Open the notebook in Jupyter:
```bash
jupyter notebook Detecting_Spam_Emails.ipynb
```

2. Run cells sequentially from top to bottom

3. The model will:
   - Load and balance the data
   - Clean and preprocess emails
   - Train an LSTM neural network
   - Evaluate performance and display results

## Model Performance

The model achieves high accuracy in classifying emails as spam or legitimate mail. Performance metrics are displayed after training:
- **Test Accuracy**: Percentage of correct predictions
- **Test Loss**: Binary cross-entropy loss on test set

## Key Features

- **Automatic Dataset Download**: Downloads dataset from GitHub
- **Data Balancing**: Handles imbalanced classes
- **Comprehensive Text Cleaning**: Multi-step text preprocessing
- **Deep Learning**: LSTM architecture for sequence modeling
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Optimizes training convergence
- **Visualization**: Plots accuracy/loss curves and word clouds

## Technical Stack

- **Deep Learning Framework**: TensorFlow/Keras
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, WordCloud
- **Text Processing**: Gensim
- **Machine Learning**: Scikit-learn

## Results

The notebook produces:
- Model summary with layer details
- Training and validation metrics
- Accuracy and loss curves
- Test set evaluation results

## Notes

- The model uses undersampling to balance the dataset
- Text sequences are padded to 100 tokens
- LSTM is particularly effective for sequential text data
- Early stopping helps prevent overfitting

## Author

Based on 100 Machine Learning Projects series

## License

Open source
