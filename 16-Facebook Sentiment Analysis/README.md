# Facebook Sentiment Analysis using NLTK

A Natural Language Processing (NLP) project that performs sentiment analysis on text data using Python's NLTK library. This project demonstrates various text processing techniques including tokenization, stemming, lemmatization, POS tagging, and sentiment analysis using VADER.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Results](#results)
- [License](#license)

## 🔍 Overview

This project analyzes sentiment from text data (specifically Kindle reviews) using the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool from NLTK. The project demonstrates essential NLP preprocessing techniques and performs sentiment classification to determine whether text expresses positive, negative, or neutral sentiment.

## ✨ Features

- **Text Tokenization**: Splits text into words and sentences
- **Stemming**: Reduces words to their root form using Porter Stemmer
- **Lemmatization**: Converts words to their dictionary base form using WordNet Lemmatizer
- **POS Tagging**: Identifies parts of speech for each word
- **Sentiment Analysis**: Analyzes text sentiment using VADER with compound, positive, negative, and neutral scores

## 🛠️ Technologies Used

- **Python 3.x**
- **Libraries**:
  - `nltk` - Natural Language Toolkit for NLP tasks
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computations
  - `matplotlib` - Data visualization
  - `re` - Regular expressions for text processing

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/facebook-sentiment-analysis.git
cd facebook-sentiment-analysis
```

2. Install required packages:
```bash
pip install nltk pandas numpy matplotlib
```

3. Download required NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('vader_lexicon')
```

## 🚀 Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook facebook_sentiment_analysis.ipynb
```

2. Run all cells sequentially to:
   - Download the text data (kindle.txt)
   - Perform text preprocessing
   - Execute sentiment analysis
   - View results and scores

## 📁 Project Structure

```
16-Facebook Sentiment Analysis/
│
├── facebook_sentiment_analysis.ipynb  # Main Jupyter notebook
├── kindle.txt                         # Text data for analysis
└── README.md                          # Project documentation
```

## 🔄 How It Works

### 1. Data Loading
The project loads text data from a file containing customer reviews or comments.

### 2. Tokenization
- **Word Tokenization**: Breaks text into individual words
- **Sentence Tokenization**: Splits text into sentences

### 3. Text Normalization
- **Stemming**: Uses Porter Stemmer to reduce words to root forms (e.g., "running" → "run")
- **Lemmatization**: Uses WordNet Lemmatizer for more accurate base forms (e.g., "better" → "good")

### 4. POS Tagging
Identifies grammatical parts of speech (noun, verb, adjective, etc.) for each word.

### 5. Sentiment Analysis
Uses VADER SentimentIntensityAnalyzer to calculate:
- **Compound Score**: Overall sentiment (-1 to +1)
- **Positive Score**: Proportion of positive sentiment
- **Negative Score**: Proportion of negative sentiment
- **Neutral Score**: Proportion of neutral sentiment

### Sentiment Classification:
- **Positive**: compound score ≥ 0.05
- **Neutral**: -0.05 < compound score < 0.05
- **Negative**: compound score ≤ -0.05

## 📊 Results

The notebook outputs:
- Tokenized words and sentences
- Stemmed and lemmatized word examples
- POS tags for text samples
- Sentiment scores for each line of text including:
  - Compound, positive, negative, and neutral scores
  - Overall sentiment classification

## 📝 Example Output

```
Text: "I love this product! It works great."
compound: 0.836 neg: 0.0 neu: 0.294 pos: 0.706

Text: "This is terrible and disappointing."
compound: -0.663 neg: 0.533 neu: 0.467 pos: 0.0
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

Part of the 100+ Machine Learning Projects series.

## 🙏 Acknowledgments

- NLTK library for providing comprehensive NLP tools
- VADER sentiment analysis tool
- Open source community for continuous support

---

**Note**: This project is for educational purposes and demonstrates fundamental NLP and sentiment analysis techniques using Python and NLTK.
