# Text Document Classification using Naive Bayes

## Project Overview
This project demonstrates text document classification using the Naive Bayes algorithm, a popular machine learning method in Natural Language Processing (NLP). The model classifies documents into predefined categories based on their text content using probability theory and Bayes' theorem.

## What is Naive Bayes?
Naive Bayes is a probabilistic classification algorithm that:
- Uses Bayes' theorem to calculate the likelihood of words occurring in different classes
- Assumes feature independence (hence "naive")
- Works particularly well for text classification and NLP tasks
- Is fast, efficient, and requires relatively small training datasets

## Dataset
The project uses synthetic text data (`synthetic_text_data.csv`) containing:
- **text**: The input text documents
- **label**: The category or class each document belongs to

## Project Structure
```
text_classification.ipynb       # Main notebook with complete pipeline
synthetic_text_data.csv         # Dataset with text and labels
README.md                       # This file
```

## Workflow

### 1. **Import Libraries**
   - NumPy, Pandas for data manipulation
   - Matplotlib and Seaborn for visualization
   - scikit-learn for machine learning

### 2. **Load and Explore Data**
   - Load the synthetic text dataset
   - View sample data

### 3. **Data Preparation**
   - Split data into training (80%) and test (20%) sets
   - Use stratified split with `train_test_split()`

### 4. **Text Preprocessing**
   - Convert text to numeric vectors using `CountVectorizer`
   - This transforms raw text into a bag-of-words representation
   - Extracts word frequency features for model training

### 5. **Model Training**
   - Initialize and train `MultinomialNB` classifier
   - The model learns word-to-category associations from training data

### 6. **Model Evaluation**
   - Calculate accuracy score
   - Generate confusion matrix
   - Visualize results with a heatmap

### 7. **Prediction on New Data**
   - Make predictions on unseen text documents
   - Example: Classify custom input text

## Key Features

- **No Label Encoding Required**: Naive Bayes in scikit-learn handles string labels natively
- **Vectorization**: Text is converted to numerical features using Count Vectorization
- **Visualization**: Confusion matrix heatmap for easy interpretation
- **Real-time Prediction**: Classify new documents on-the-fly

## Usage

### Running the Notebook
1. Open `text_classification.ipynb` in Jupyter Notebook or VS Code
2. Run cells sequentially from top to bottom
3. View results and visualizations

### Predicting on Custom Text
Modify the `user_input` variable in the prediction cell:
```python
user_input = "Your custom text here"
user_input_vectorized = vectorizer.transform([user_input])
predicted_label = model.predict(user_input_vectorized)
print(f"The input text belongs to the '{predicted_label[0]}' category.")
```

## Results
- **Accuracy**: Displays the percentage of correct predictions
- **Confusion Matrix**: Shows true positives, false positives, true negatives, and false negatives
- **Heatmap Visualization**: Easy-to-read graphical representation of model performance

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Installation
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Performance Metrics
- **Accuracy Score**: Overall correctness of predictions
- **Confusion Matrix**: Detailed breakdown of prediction results by class

## Advantages of Naive Bayes for Text Classification
✓ Fast training and prediction  
✓ Works well with high-dimensional data (many features)  
✓ Requires relatively small training datasets  
✓ Interpretable results  
✓ Handles multiple classes efficiently  

## Limitations
✗ Assumes feature independence (rarely true in practice)  
✗ May be outperformed by more complex models on large datasets  
✗ Can be affected by imbalanced datasets  

## Future Improvements
- Implement other vectorizers (TF-IDF, Word2Vec)
- Try alternative algorithms (SVM, Random Forest, Neural Networks)
- Add cross-validation for better model evaluation
- Implement hyperparameter tuning
- Handle imbalanced datasets with techniques like SMOTE
- Add data augmentation for more robust training

## Author
Machine Learning Projects Series

## License
Open Source
