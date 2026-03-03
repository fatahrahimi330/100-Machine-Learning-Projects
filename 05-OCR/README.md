# Optical Character Recognition (OCR) with OpenCV and KNN

## Overview

This project implements an **Optical Character Recognition (OCR)** system for **handwritten digit classification** using OpenCV and the K-Nearest Neighbors (KNN) algorithm. The model is trained on handwritten digit images and can accurately predict and classify unseen digits.

## About the Project

OCR is a Computer Vision technique used for automatic recognition of handwritten characters and digits. In this implementation, we use the **KNN (K-Nearest Neighbors)** classifier which:
- Detects the k nearest neighbors of a particular point
- Classifies that point based on the majority class of its neighbors
- Provides high accuracy for handwritten digit recognition tasks

## Features

- ✅ Download and load handwritten digit dataset from GitHub
- ✅ Convert images to grayscale for processing
- ✅ Divide images into 5,000 cells of 20×20 pixels each
- ✅ Split data into training (50% × 10 digits × 250 samples) and test sets
- ✅ Train KNN classifier on flattened digit representations
- ✅ Evaluate model accuracy on test set
- ✅ Make predictions on individual digit samples
- ✅ Visualize predictions with color-coded results (green for correct, red for incorrect)

## Requirements

```
numpy
pandas
matplotlib
seaborn
opencv-python (cv2)
```

## Installation

1. Clone or download the project
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn opencv-python
   ```
3. Run the Jupyter notebook:
   ```bash
   jupyter notebook ocr_handwritten_digits.ipynb
   ```

## Project Structure

### 1. Importing Libraries
Loads essential libraries:
- `numpy` - numerical operations
- `pandas` - data manipulation
- `matplotlib` & `seaborn` - visualization
- `cv2` - image processing with OpenCV

### 2. Importing Dataset
- Downloads handwritten digit image from GitHub
- Uses `wget` command to fetch the dataset
- Image contains all digit samples organized in a grid

### 3. Gray Scale Conversion
- Converts the colored/RGB image to grayscale
- Uses `cv2.cvtColor()` with `COLOR_BGR2GRAY` conversion

### 4. Divide the Image into 5000 Dimensions
- Splits the grayscale image vertically into 50 sections
- Splits each section horizontally into 100 sections
- Results in 5,000 individual 20×20 digit cells

### 5. Convert into NumPy Array
- Converts the nested list structure into a NumPy array
- Final shape: **(50, 100, 20, 20)**
  - 50 = vertical divisions
  - 100 = horizontal divisions
  - 20, 20 = height and width of each cell

### 6. Training and Test Sets
- First 50 columns used for training (2,500 samples = 10 digits × 250 each)
- Last 50 columns used for testing (2,500 samples = 10 digits × 250 each)
- Flattens 20×20 images into 400-dimensional vectors
- Converts to float32 for OpenCV compatibility

### 7. Creating Labels
- Creates labels 0-9 for digits
- Repeats each label 250 times for training and test sets
- Reshapes into column vectors for compatibility with KNN

### 8. Building and Training the Model
- Initializes KNearest classifier: `cv2.ml.KNearest_create()`
- Trains on flattened feature vectors and labels
- Uses `ROW_SAMPLE` format indicating samples are in rows

### 9. Evaluation
- Makes predictions on all test samples using k=3 neighbors
- Compares predictions with actual labels
- Calculates accuracy percentage

### 10. Making a Prediction
- Selects a single test sample by index
- Makes a prediction using the trained model
- Prints predicted digit, actual digit, neighbors, and distance metrics
- Visualizes the digit with color-coded title (green for correct, red for incorrect)

## Usage Example

### Run the notebook:
1. Execute all cells in sequence
2. Adjust `sample_index` in the prediction cell to test different digits
3. View the accuracy metric in the evaluation section

### Change prediction sample:
```python
sample_index = 100  # Change this to test different digits
```

## Model Performance

The KNN model with k=3 neighbors typically achieves:
- **Accuracy**: 90%+ on handwritten digit test set
- **Fast predictions**: Real-time classification of new digit samples
- **Simple yet effective**: No complex neural networks required

## How KNN Works for Digit Classification

1. **Feature Extraction**: Each 20×20 digit image is flattened into 400 features
2. **Distance Calculation**: For a new digit, calculate distance to all training samples
3. **Neighbor Selection**: Find the 3 nearest training samples (k=3)
4. **Voting**: Classify based on majority class among the 3 neighbors
5. **Prediction**: Return the most common digit class

## Color-Coded Predictions

- 🟢 **Green Title**: Prediction matches actual digit (Correct)
- 🔴 **Red Title**: Prediction differs from actual digit (Incorrect)

## Files in This Project

```
ocr_handwritten_digits.ipynb  - Main Jupyter notebook
image.png.1                   - Downloaded handwritten digit dataset
README.md                     - This file
```

## Key Insights

- **Grid Division**: The 50×100 grid creates 5,000 training and 5,000 test samples
- **Dimensionality**: 400 features per digit (20×20 pixels flattened)
- **KNN Simplicity**: No complex training required, just stores training data
- **k Value**: k=3 provides good balance between accuracy and speed

## Future Enhancements

- Implement cross-validation for better accuracy estimation
- Try different k values (k=1, 3, 5, 7) and compare performance
- Add support for other characters (not just digits 0-9)
- Implement more advanced algorithms (SVM, Neural Networks)
- Add real-time handwritten digit input via camera
- Deploy as web application for digit recognition

## Author

Created as part of the 100 Machine Learning Projects series.

## License

This project is open source and available for educational purposes.

---

**Happy Learning! 🎉**
