# Waiter's Tip Prediction 🍽️💰

This project focuses on predicting the tip amount a waiter might receive based on various factors such as the total bill, the guest's gender, whether they are smokers, the day of the week, the time of day, and the size of the party.

## 📌 Project Overview
The goal of this project is to build a regression model that can accurately estimate the tip amount. This kind of analysis is valuable for restaurant management and service staff to understand tipping behaviors and factors that influence customer generosity.

## 📊 Dataset Description
The dataset used in this project is the famous `tips` dataset, which contains 244 records of restaurant transactions.

**Features:**
- `total_bill`: Total amount of the bill (in USD).
- `sex`: Gender of the person paying the bill (Male/Female).
- `smoker`: Whether the party included smokers (Yes/No).
- `day`: Day of the week (Thur, Fri, Sat, Sun).
- `time`: Time of the day (Lunch, Dinner).
- `size`: Number of people in the party.

**Target Variable:**
- `tip`: The amount of tip given to the waiter.

## 🛠️ Technologies Used
- **Language:** Python
- **Libraries:**
  - `Pandas` & `NumPy`: Data manipulation and analysis.
  - `Matplotlib` & `Seaborn`: Data visualization.
  - `Scikit-Learn`: Machine learning modeling and evaluation.
  - `XGBoost`: Gradient boosting framework.

## 🚀 Machine Learning Workflow
1. **Data Loading:** Importing the dataset from a CSV file.
2. **Exploratory Data Analysis (EDA):** 
   - Visualizing the distribution of tips and total bills.
   - Analyzing relationships between features (e.g., tip vs. day, tip vs. gender).
   - Checking for missing values.
3. **Data Preprocessing:**
   - Encoding categorical variables (sex, smoker, day, time) into numerical values.
   - Splitting the data into training and testing sets.
4. **Model Building:** Implementing and comparing multiple regression models:
   - **Linear Regression**
   - **Random Forest Regressor**
   - **XGBoost Regressor**
   - **AdaBoost Regressor**
5. **Evaluation:** Testing the models to find the most accurate predictor.

## 📈 Key Visualizations
Included in the notebook are several plots providing insights:
- Distribution of total bills and tips.
- Average tip by day and time.
- Impact of smoking status on tipping.
- Heatmaps showing correlation between numerical features.

## 🏁 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/fatahrahimi330/100-Machine-Learning-Projects.git
   ```
2. Navigate to the project directory:
   ```bash
   cd "62-Waiter's Tip Prediction"
   ```
3. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```
4. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Waiter'sTipPrediction.ipynb
   ```

## 📄 License
This project is part of the "100 Machine Learning Projects" series. Feel free to use the code for learning purposes.
