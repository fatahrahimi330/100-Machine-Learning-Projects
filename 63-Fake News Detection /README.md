# Fake News Detection

This project aim to build a machine learning model to classify news as **Fake** or **True** using Natural Language Processing (NLP) techniques and various classification algorithms.

## Project Overview

The proliferation of misinformation and fake news on the internet is a significant concern. This project develops a classifier that can distinguish between real and fake news articles based on their text content.

## Dataset

The dataset used in this project is `News.csv`, which contains:
- **text**: The content of the news article.
- **class**: The label (0 for Fake News, 1 for True News).

## Workflow

### 1. Data Cleaning and Preprocessing
- Removed unnecessary columns such as `title`, `subject`, and `date`.
- Shuffled the dataset to avoid bias.
- Performed text cleaning:
    - Lowercasing the text.
    - Removing special characters and punctuation.
    - Removing English stopwords.

### 2. Feature Extraction
- Used `TfidfVectorizer` (TF-IDF) to convert text data into numerical features that machine learning models can understand.

### 3. Model Training
Four different classification models were trained and evaluated:
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Gradient Boosting Classifier**

### 4. Evaluation
Models were evaluated using Accuracy and Classification Reports (Precision, Recall, F1-Score). A comparison of accuracies was visualized to identify the best-performing model.

### 5. Manual Testing
A testing function is included to predict the authenticity of any user-provided news snippet using all four trained models.

## Libraries Used
- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`, `wordcloud`
- **NLP**: `nltk`, `re`
- **Machine Learning**: `scikit-learn`
- **Tools**: `tqdm` (for progress bars)

## How to Run
1. Clone the repository.
2. Ensure you have the required libraries installed:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn nltk tqdm wordcloud
   ```
3. Open the Jupyter Notebook `Fake News Detection .ipynb`.
4. Run all cells to process the data, train the models, and see the results.
5. Use the `manual_testing` function at the end of the notebook to test custom news articles.

## Results
The project compares the performance of multiple classifiers. Typically, ensemble methods like **Random Forest** and **Gradient Boosting** provide high accuracy for text classification tasks like this.
