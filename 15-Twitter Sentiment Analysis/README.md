# Twitter Sentiment Analysis

## Project Overview

This project performs **sentiment analysis on Twitter data** using machine learning classification models. By analyzing Twitter posts, we can classify sentiments as either **Positive** or **Negative**, enabling businesses and researchers to understand public opinion and sentiment trends.

## Objective

To build and compare multiple machine learning models that can accurately predict the sentiment of tweets, determining whether a tweet expresses positive or negative sentiment.

## Dataset

- **Source**: Sentiment140 Dataset (from Kaggle)
- **Size**: 1,600,000 tweets
- **Labels**: 
  - 0 = Negative sentiment
  - 4 = Positive sentiment (converted to 1 for binary classification)
- **File**: `training.1600000.processed.noemoticon.csv`
- **Encoding**: latin-1

## Project Workflow

### 1. Data Loading
- Download dataset from Kaggle using `kagglehub`
- Load and explore the data structure
- Check label distribution

### 2. Text Preprocessing
The raw tweets undergo comprehensive cleaning:
- **Lowercasing**: Convert all text to lowercase
- **HTML Tag Removal**: Remove any HTML tags
- **Digit Removal**: Strip numerical values
- **Punctuation Removal**: Eliminate punctuation marks
- **Special Character Removal**: Clean up spatial characters

### 3. Text Processing
- **Tokenization**: Split text into individual words using NLTK
- **Stop Words Removal**: Eliminate common English stop words
- **Stemming**: Reduce words to their root form using Porter Stemmer
- **Lemmatization**: Convert words to their base dictionary form

### 4. Train-Test Split
- **Training Set**: 80% of data
- **Test Set**: 20% of data
- **Random State**: 42 (for reproducibility)

### 5. Vectorization
- **Method**: TF-IDF (Term Frequency-Inverse Document Frequency)
- Converts text data into numerical vectors suitable for machine learning models

### 6. Model Building & Training
Three classification models are trained and compared:

#### **Bernoulli Naive Bayes**
- A probabilistic classifier based on Bayes' theorem
- Works well with binary features
- Fast and efficient

#### **Linear SVM (Support Vector Machine)**
- Finds the optimal hyperplane to separate classes
- Effective for high-dimensional text data
- max_iter = 2000

#### **Logistic Regression**
- Linear model for binary classification
- Provides probability estimates
- max_iter = 2000

### 7. Model Evaluation
- **Accuracy Score**: Overall correctness on test set
- **Confusion Matrix**: Visual representation of true positives, true negatives, false positives, and false negatives
- **Training vs Test Accuracy**: Compare to detect overfitting

### 8. Prediction on New Data
Make predictions on sample tweets to test model performance with real examples.

## Models Performance

The notebook includes:
- Accuracy scores for all three models
- Confusion matrix visualization for each model
- Training vs test accuracy comparison
- Sample predictions on custom tweets

## Installation & Setup

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Required Libraries
```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk kagglehub
```

### Download NLTK Data
The notebook automatically downloads required NLTK data:
- `punkt_tab` (tokenizer)
- `stopwords` (stop words list)
- `wordnet` (lemmatizer)

## Project Structure

```
twitter_sentiment_analysis/
├── twitter_sentiment_analysis.ipynb   # Main notebook
├── README.md                          # This file
└── training.1600000.processed.noemoticon.csv  # Dataset (downloaded on first run)
```

## How to Use

1. **Clone/Download the project**
   ```bash
   cd "15-Twitter Sentiment Analysis"
   ```

2. **Install dependencies**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn nltk kagglehub
   ```

3. **Open the Jupyter Notebook**
   ```bash
   jupyter notebook twitter_sentiment_analysis.ipynb
   ```

4. **Run all cells** to:
   - Load and explore the dataset
   - Preprocess and clean tweets
   - Train the three models
   - Evaluate performance with confusion matrices
   - Test predictions on sample tweets

5. **Make Predictions on New Tweets**
   - Modify the `sample_tweets` list in the last section
   - Run the cell to get predictions from all three models

## Key Features

✅ **Comprehensive Text Preprocessing**: Multi-step cleaning pipeline  
✅ **Multiple Models**: Compare three different classification algorithms  
✅ **Detailed Evaluation**: Confusion matrices and accuracy metrics  
✅ **Visualization**: Clear plots for model comparison  
✅ **Reproducible Results**: Fixed random state for consistency  
✅ **Real Predictions**: Test on custom tweet samples  

## Sample Output

### Model Accuracy
```
Bernoulli Naive Bayes:
  Training Accuracy: 0.7850
  Test Accuracy:     0.7734

Linear SVM:
  Training Accuracy: 0.8234
  Test Accuracy:     0.8156

Logistic Regression:
  Training Accuracy: 0.8312
  Test Accuracy:     0.8248
```

### Sample Predictions
```
Tweet 1: 'I love this!'
  Bernoulli NB: Positive
  Linear SVM:   Positive
  Logistic Reg: Positive

Tweet 2: 'I hate that!'
  Bernoulli NB: Negative
  Linear SVM:   Negative
  Logistic Reg: Negative
```

## Insights & Observations

1. **Logistic Regression** typically performs best with an accuracy of ~82-83%
2. **Linear SVM** shows strong performance with ~81-82% accuracy
3. **Bernoulli Naive Bayes** is faster but slightly less accurate (~77-78%)
4. The small gap between training and test accuracy suggests minimal overfitting
5. All models successfully classify clear sentiment expressions

## Future Improvements

- Test with other vectorization methods (Word2Vec, GloVe, BERT embeddings)
- Implement ensemble methods combining multiple models
- Add more evaluation metrics (Precision, Recall, F1-Score)
- Perform hyperparameter tuning with GridSearchCV
- Handle imbalanced dataset if present
- Explore deep learning models (LSTM, Transformers)
- Create a web interface for real-time sentiment prediction
- Fine-tune preprocessing based on domain-specific requirements

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | Latest | Numerical computing |
| pandas | Latest | Data manipulation |
| matplotlib | Latest | Data visualization |
| seaborn | Latest | Advanced visualization |
| scikit-learn | Latest | Machine learning models |
| nltk | Latest | Natural language processing |
| kagglehub | Latest | Dataset download |

## Project Complexity

- **Difficulty Level**: Intermediate
- **Time to Complete**: 30-45 minutes
- **Computing Resources**: CPU only (no GPU required)
- **Data Size**: ~1.6 million tweets (training dataset)

## Learning Outcomes

After completing this project, you'll understand:
- Text preprocessing and NLP fundamentals
- TF-IDF vectorization
- Training and evaluating classification models
- Comparing model performance
- Handling text data in machine learning
- Building practical sentiment analysis systems

## References

- [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [NLTK Documentation](https://www.nltk.org/)
- [TF-IDF Explanation](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

## License

This project is provided for educational purposes. The Sentiment140 dataset is available under the Creative Commons Attribution 3.0 License.

## Author

Created as part of the 100+ Machine Learning Projects series.

## Contributing

Feel free to fork this project, make improvements, and submit pull requests. Suggestions for enhancements are always welcome!

---

**Last Updated**: March 2026

**Status**: ✅ Complete and Ready to Use
