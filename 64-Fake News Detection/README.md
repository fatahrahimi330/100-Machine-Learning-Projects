# Fake News Detection using ANN and TensorFlow

This repository contains a machine learning project focused on detecting fake news using Artificial Neural Networks (ANN) with a combination of Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) layers, implemented in TensorFlow.

## 📌 Project Overview

The primary goal of this project is to classify news articles as either **REAL** or **FAKE**. By leveraging Natural Language Processing (NLP) techniques and deep learning architectures, we can identify patterns in news titles and text that distinguish reliable sources from misinformation.

## 📊 Dataset

The project uses the `news.csv` dataset, which contains:
- **title**: The title of the news article.
- **text**: The full content of the article.
- **label**: The classification of the news (`REAL` or `FAKE`).

The dataset is preprocessed using **Label Encoding** to convert categorical labels into numeric values (0 and 1).

## 🛠️ Machine Learning Workflow

1.  **Data Loading & Exploratory Data Analysis (EDA)**: Importing the dataset and examining its structure using `pandas`.
2.  **Text Preprocessing**:
    -   Implementing `Tokenizer` to convert text into sequences of integers.
    -   Applying `pad_sequences` to ensure uniform input length for the model.
    -   Using `LabelEncoder` for target variable transformation.
3.  **Word Embeddings**: Integrating pre-trained **GloVe (Global Vectors for Word Representation)** 50-dimensional embeddings to capture semantic meanings of words.
4.  **Model Architecture**:
    -   **Embedding Layer**: Maps vocabulary indices to dense embedding vectors.
    -   **Dropout Layer**: Regularization technique to prevent overfitting.
    -   **Conv1D Layer**: Filters and extracts local spatial features from text sequences.
    -   **MaxPooling1D Layer**: Reduces the dimensionality of features.
    -   **LSTM Layer**: Captures long-range dependencies and temporal patterns in the text.
    -   **Dense Layer**: Fully connected layer with a Sigmoid activation function for binary classification.
5.  **Training**: The model is compiled with the `adam` optimizer and `binary_crossentropy` loss function, trained for 50 epochs.
6.  **Evaluation**: Visualizing performance through Accuracy plots, ROC Curves, and Confusion Matrices.

## 🚀 Performance

The model achieves significant performance in detecting fake news:
- **Training Accuracy**: ~97%
- **Validation Accuracy**: ~74%

## 💻 How to Run

1.  Clone the repository or download the project files.
2.  Ensure you have the required dependencies installed:
    ```bash
    pip install numpy pandas matplotlib seaborn tensorflow scikit-learn tqdm
    ```
3.  Download the GloVe 50d embeddings (if not included) from [Stanford NLP](https://nlp.stanford.edu/projects/glove/).
4.  Open and run the `FakeNewsDetection.ipynb` Jupyter Notebook.

## 📦 Dependencies

-   Python 3.x
-   TensorFlow
-   scikit-learn
-   Pandas
-   NumPy
-   Matplotlib & Seaborn
-   tqdm
