# Cartooning Image using OpenCV

A Python-based image processing project that transforms regular photos into cartoon-style images using OpenCV.

## 📝 Description

This project converts normal photographs into cartoon-styled images by applying various computer vision techniques including edge detection, bilateral filtering, and image masking. The result is a fun, artistic cartoon version of your original photo.

## ✨ Features

- Converts any image into a cartoon-style illustration
- Uses adaptive thresholding for edge detection
- Applies bilateral filtering for smooth color transitions
- Preserves important features while creating artistic effects
- Simple and easy-to-use implementation

## 🔧 Requirements

- Python 3.x
- OpenCV (`cv2`)
- Matplotlib
- NumPy (typically installed with OpenCV)

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/fatahrahimi330/100-Machine-Learning-Projects.git
cd "100-Machine-Learning-Projects/8-Cartooning Image"
```

2. Install required packages:
```bash
pip install opencv-python matplotlib numpy
```

## 🚀 Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook cartooning_image.ipynb
```

2. Run all cells in sequence to:
   - Import necessary libraries
   - Download and load the sample image
   - Apply cartoon effect
   - Display and save the result

## 🎨 How It Works

The cartooning algorithm consists of several steps:

1. **Grayscale Conversion**: Convert the original image to grayscale
2. **Median Blur**: Apply median blur to reduce noise (kernel size: 5)
3. **Edge Detection**: Use adaptive thresholding to detect edges
   - Method: `ADAPTIVE_THRESH_MEAN_C`
   - Block size: 9
   - Constant: 9
4. **Bilateral Filtering**: Smooth colors while preserving edges
   - Filter size: 9
   - Sigma color: 250
   - Sigma space: 250
5. **Combining**: Use bitwise AND operation to combine edges with smoothed colors

## 📊 Algorithm Parameters

```python
# Grayscale and Blur
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blure_gray_image = cv2.medianBlur(gray_image, 5)

# Edge Detection
edge = cv2.adaptiveThreshold(blure_gray_image, 255, 
                              cv2.ADAPTIVE_THRESH_MEAN_C, 
                              cv2.THRESH_BINARY, 9, 9)

# Smooth Colors
color = cv2.bilateralFilter(image, 9, 250, 250)

# Combine
cartoon = cv2.bitwise_and(color, color, mask=edge)
```

## 📸 Output

The cartoonized image is saved as `cartoon_output.png` in the project directory and displayed inline in the notebook.

## 🎯 Use Cases

- Creating artistic versions of photographs
- Social media content creation
- Image preprocessing for artistic applications
- Learning computer vision and image processing techniques

## 🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements.

## 📄 License

This project is part of the [100+ Machine Learning Projects](https://github.com/fatahrahimi330/100-Machine-Learning-Projects) collection.

## 👤 Author

**Fatah Rahimi**
- GitHub: [@fatahrahimi330](https://github.com/fatahrahimi330)

## 🙏 Acknowledgments

- OpenCV community for excellent documentation
- Computer vision techniques for cartoon effect implementation

---

**Note**: Make sure you have a proper GUI environment if you want to use `cv2.imshow()`. For Jupyter notebooks, use `matplotlib.pyplot.imshow()` for displaying images.
