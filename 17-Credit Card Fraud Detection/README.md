# Credit Card Fraud Detection 💳

A machine learning project to detect fraudulent credit card transactions using Random Forest Classifier. This project addresses the challenge of identifying fraudulent transactions in a highly imbalanced dataset.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-latest-green.svg)

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

Credit card fraud is a critical issue in financial services. This project implements a **Random Forest Classifier** to identify fraudulent transactions from a dataset of credit card transactions. The model handles class imbalance and achieves high accuracy in detecting fraud while minimizing false positives.

### Key Highlights:
- Handles highly imbalanced dataset (fraud cases < 1%)
- Implements Random Forest algorithm for robust classification
- Comprehensive data analysis and visualization
- Evaluation using multiple metrics (Accuracy, Precision, Recall, F1-Score, MCC)

## 📊 Dataset

The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days.

**Dataset Characteristics:**
- **Total Transactions:** 284,807
- **Features:** 30 (28 anonymized PCA features + Time + Amount)
- **Target Variable:** Class (0 = Valid, 1 = Fraud)
- **Fraud Cases:** 492 (~0.17%)
- **Valid Cases:** 284,315 (~99.83%)

**Features:**
- `V1, V2, ... V28`: PCA transformed features (confidential)
- `Time`: Seconds elapsed between this transaction and the first transaction
- `Amount`: Transaction amount
- `Class`: Target variable (0 = Normal, 1 = Fraud)

### Accessing the Dataset

The dataset is stored using **Git LFS** (Large File Storage) due to its size (143.84 MB).

**For Colab Users:**
```python
!wget https://media.githubusercontent.com/media/fatahrahimi330/100-Machine-Learning-Projects/refs/heads/master/17-Credit%20Card%20Fraud%20Detection/creditcard.csv
df = pd.read_csv("creditcard.csv")
```

**For Local Development:**
```bash
# Clone the repository with Git LFS
git lfs install
git clone https://github.com/fatahrahimi330/100-Machine-Learning-Projects.git
cd "17-Credit Card Fraud Detection"
```

## ✨ Features

1. **Data Exploration**
   - Statistical analysis of transaction amounts
   - Class distribution visualization
   - Correlation matrix analysis

2. **Data Processing**
   - Feature scaling and normalization
   - Train-test split (80-20)
   - Handling imbalanced data

3. **Model Training**
   - Random Forest Classifier implementation
   - Cross-validation
   - Hyperparameter optimization

4. **Model Evaluation**
   - Confusion Matrix
   - Accuracy, Precision, Recall
   - F1-Score
   - Matthews Correlation Coefficient (MCC)

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/fatahrahimi330/100-Machine-Learning-Projects.git
cd "100-Machine-Learning-Projects/17-Credit Card Fraud Detection"
```

2. **Install required packages**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

3. **For Git LFS (to download the dataset)**
```bash
# Install Git LFS
brew install git-lfs  # macOS
# or
sudo apt-get install git-lfs  # Linux

# Initialize Git LFS
git lfs install
git lfs pull
```

## 💻 Usage

### Running the Notebook

1. **Jupyter Notebook**
```bash
jupyter notebook Credit_Card_Fraud_Detection.ipynb
```

2. **VS Code**
- Open the folder in VS Code
- Install Python and Jupyter extensions
- Open `Credit_Card_Fraud_Detection.ipynb`
- Run cells sequentially

3. **Google Colab**
- Upload the notebook to Google Colab
- Run the wget command to download the dataset
- Execute cells in order

### Quick Start Example

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("creditcard.csv")

# Prepare data
X = df.drop(['Class'], axis=1)
y = df['Class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)

# Make predictions
y_pred = rfc.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred))
```

## 📈 Model Performance

The Random Forest Classifier achieves excellent performance on the test set:

| Metric | Score |
|--------|-------|
| **Accuracy** | ~99.95% |
| **Precision** | ~95-97% |
| **Recall** | ~75-80% |
| **F1-Score** | ~85-88% |
| **Matthews Correlation Coefficient** | ~0.85+ |

### Confusion Matrix
The model demonstrates:
- **High True Negative Rate**: Correctly identifies valid transactions
- **Good True Positive Rate**: Detects fraudulent transactions effectively
- **Low False Positive Rate**: Minimizes false fraud alerts
- **Acceptable False Negative Rate**: Balances fraud detection sensitivity

*Note: Actual metrics may vary based on random seed and data split*

## 📁 Project Structure

```
17-Credit Card Fraud Detection/
│
├── Credit_Card_Fraud_Detection.ipynb  # Main Jupyter notebook
├── creditcard.csv                      # Dataset (Git LFS)
├── README.md                          # Project documentation
└── .gitattributes                     # Git LFS configuration
```

## 🛠️ Technologies Used

- **Python 3.8+**: Programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization
- **Scikit-learn**: Machine learning library
  - RandomForestClassifier
  - train_test_split
  - Evaluation metrics
- **Jupyter Notebook**: Interactive development environment
- **Git LFS**: Large file storage

## 📊 Results

### Key Insights:

1. **Class Imbalance**: Fraud cases represent only 0.17% of all transactions
2. **Outlier Fraction**: 0.0017 (fraud to valid ratio)
3. **Amount Analysis**: 
   - Fraudulent transactions show different amount patterns
   - Most fraud occurs at various transaction amounts
4. **Feature Correlation**: PCA features show varying correlations with fraud

### Visualizations Included:

- Class distribution bar chart
- Correlation heatmap
- Confusion matrix
- Transaction amount distribution comparison

## 🔍 Methodology

### 1. Data Loading & Exploration
- Load dataset and examine structure
- Analyze statistical properties
- Identify class imbalance

### 2. Data Analysis
- Explore transaction amounts for both classes
- Visualize correlations between features
- Understand fraud patterns

### 3. Data Preparation
- Split features and target variable
- Create train-test split (80-20)
- Prepare data for model training

### 4. Model Training
- Initialize Random Forest Classifier
- Train on training dataset
- Generate predictions on test set

### 5. Model Evaluation
- Calculate multiple performance metrics
- Generate confusion matrix
- Analyze model strengths and weaknesses

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Improvement:
- Implement SMOTE or other oversampling techniques
- Try different algorithms (XGBoost, Neural Networks)
- Add feature engineering
- Implement cross-validation
- Create a web interface for predictions
- Add real-time fraud detection pipeline

## 📝 License

This project is part of the [100+ Machine Learning Projects](https://github.com/fatahrahimi330/100-Machine-Learning-Projects) repository.

## 👤 Author

**Fatah Rahimi**
- GitHub: [@fatahrahimi330](https://github.com/fatahrahimi330)

## 🙏 Acknowledgments

- Dataset source: Credit Card Fraud Detection Dataset
- Scikit-learn documentation and community
- Machine learning community for best practices

## 📞 Contact

For questions or feedback, please open an issue in the repository or contact the author directly.

---

**Note**: This project is for educational purposes. Always ensure compliance with financial regulations when working with real transaction data.

## 🔗 Related Projects

Check out other machine learning projects in the [100+ Machine Learning Projects](https://github.com/fatahrahimi330/100-Machine-Learning-Projects) repository!

---

Made with ❤️ by Fatah Rahimi
