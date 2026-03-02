# Identify Handwritten Digits Using Logistic Regression in PyTorch

## Project Overview

This project implements a handwritten digit classification system using Logistic Regression in PyTorch. The model is trained on the MNIST dataset to recognize digits from 0-9 with high accuracy.

## Dataset

**MNIST (Modified National Institute of Standards and Technology)**
- Training samples: 60,000 images
- Test samples: 10,000 images
- Image size: 28x28 pixels (grayscale)
- Number of classes: 10 (digits 0-9)

The MNIST dataset is automatically downloaded when running the notebook.

## Model Architecture

### Logistic Regression Model
- **Input Layer**: 784 features (28×28 flattened image)
- **Output Layer**: 10 classes (digits 0-9)
- **Activation**: Softmax (applied via CrossEntropyLoss)

### Hyperparameters
- **Learning Rate**: 0.001
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 64
- **Number of Epochs**: 5

## Requirements

```
numpy
pandas
matplotlib
seaborn
torch
torchvision
```

## Installation

1. Clone or download this project
2. Install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn torch torchvision
```

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook identify_handwritten_digits.ipynb
```

2. Run all cells in sequence:
   - Import libraries
   - Load and prepare the MNIST dataset
   - Define the model architecture
   - Train the model
   - Evaluate performance
   - Visualize results

## Project Structure

```
7-Identify Handwritten Digits/
│
├── identify_handwritten_digits.ipynb    # Main notebook with complete implementation
├── README.md                            # Project documentation
└── data/                                # MNIST dataset (auto-downloaded)
    └── MNIST/
```

## Results

### Model Performance

The logistic regression model achieves:
- **Test Accuracy**: ~92% (approximately)
- **Training Time**: Fast convergence within 5 epochs

### Training and Test Loss per Epoch

The model shows consistent learning behavior with decreasing loss over epochs:

![Training and Test Loss Curve](train_test_loss_per_epoch.png)

**Key Observations:**
- Training loss decreases steadily across epochs
- Test loss follows a similar pattern, indicating good generalization
- No significant overfitting observed (train and test losses are close)
- Model converges quickly due to the relatively simple linear architecture

## Implementation Details

### 1. Data Loading
- Uses `torchvision.datasets.MNIST` for easy data loading
- Applies `ToTensor()` transformation to normalize images
- Creates DataLoader with batch_size=64 and shuffling for training

### 2. Model Training
- Tracks both training and test loss per epoch
- Evaluates model on test set after each epoch
- Prints progress with loss and accuracy metrics
- Uses `model.train()` and `model.eval()` modes appropriately

### 3. Model Evaluation
- Computes accuracy on 10,000 test images
- Tracks predictions vs ground truth
- Reports final accuracy percentage

### 4. Visualization
- Plots training and test loss curves
- Uses matplotlib for clear visualization
- Displays both curves on the same plot for comparison

## Technical Approach

**Why Logistic Regression?**
- Simple yet effective baseline for digit classification
- Fast training and inference
- Good interpretability
- Demonstrates fundamental concepts in deep learning

**PyTorch Framework:**
- Provides flexible and dynamic computational graphs
- Offers automatic differentiation via autograd
- Enables easy GPU acceleration (if available)
- Industry-standard framework with extensive community support

## Future Improvements

1. **Model Architecture**
   - Implement CNN (Convolutional Neural Network) for better accuracy
   - Add hidden layers for increased model capacity
   - Experiment with dropout for regularization

2. **Hyperparameter Tuning**
   - Try different learning rates
   - Experiment with optimizers (Adam, RMSprop)
   - Adjust batch size and epochs

3. **Data Augmentation**
   - Apply random rotations
   - Add noise for robustness
   - Use elastic deformations

4. **Visualization**
   - Display sample predictions with images
   - Show confusion matrix
   - Visualize misclassified examples

## References

- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Logistic Regression for Multi-class Classification](https://en.wikipedia.org/wiki/Multinomial_logistic_regression)

## License

This project is open source and available for educational purposes.

## Author

Created as part of the 100+ Machine Learning Projects series.

## Acknowledgments

- MNIST dataset by Yann LeCun, Corinna Cortes, and Christopher Burges
- PyTorch team for the excellent deep learning framework
- Open source community for tools and resources
