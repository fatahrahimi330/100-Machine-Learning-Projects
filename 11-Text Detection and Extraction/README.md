# Text Detection and Extraction using OpenCV

A comprehensive machine learning project that demonstrates text detection and extraction from images using OpenCV and Tesseract OCR.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [How It Works](#how-it-works)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a complete pipeline for detecting and extracting text from images. It uses computer vision techniques with OpenCV and performs Optical Character Recognition (OCR) using Tesseract to identify, localize, and extract text content from images.

## ✨ Features

- **Image Loading**: Load images from URLs or local files
- **Image Preprocessing**: Convert images to grayscale for optimal text detection
- **Text Extraction**: Extract text content using Tesseract OCR
- **Text Localization**: Draw bounding boxes around detected text regions
- **Visualization**: Display original images and processed results with matplotlib
- **Multiple Format Support**: Works with various image formats (PNG, JPG, JPEG, etc.)

## 📦 Prerequisites

Before running this project, ensure you have the following installed:

- Python 3.7 or higher
- pip (Python package installer)
- Tesseract OCR engine

### Installing Tesseract OCR

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Download and install from [Tesseract at GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/fatahrahimi330/100-Machine-Learning-Projects.git
   cd "100-Machine-Learning-Projects/11-Text Detection and Extraction"
   ```

2. **Install required Python packages:**
   ```bash
   pip install opencv-python pytesseract matplotlib
   ```

3. **Verify Tesseract installation:**
   ```bash
   tesseract --version
   ```

## 💻 Usage

### Running the Jupyter Notebook

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook text_detection_and_extraction.ipynb
   ```

2. Run the cells sequentially to:
   - Import required libraries
   - Load a sample image
   - Convert to grayscale
   - Extract text using OCR
   - Draw bounding boxes around detected text
   - Visualize results

### Basic Python Script Example

```python
import cv2
import pytesseract
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('your_image.png')
img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to grayscale
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Extract text
extracted_text = pytesseract.image_to_string(img_gray)
print("Extracted Text:", extracted_text)

# Get bounding box data
data = pytesseract.image_to_data(img_rgb, output_type=pytesseract.Output.DICT)
num_boxes = len(data['level'])

# Draw bounding boxes
for i in range(num_boxes):
    (x, y, w, h) = (data['left'][i], data['top'][i], 
                    data['width'][i], data['height'][i])
    cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display result
plt.imshow(img_rgb)
plt.title("Text Detection Results")
plt.axis('off')
plt.show()
```

## 📁 Project Structure

```
11-Text Detection and Extraction/
├── text_detection_and_extraction.ipynb   # Main Jupyter notebook
├── README.md                              # Project documentation
├── image.png                              # Sample image (downloaded)
└── image1.png                             # Processed image
```

## 🛠️ Technologies Used

- **OpenCV (cv2)**: Image processing and computer vision operations
- **Pytesseract**: Python wrapper for Tesseract OCR engine
- **Matplotlib**: Data visualization and image display
- **NumPy**: Numerical computing (included with OpenCV)
- **Tesseract OCR**: Optical Character Recognition engine

## 🔍 How It Works

### 1. **Image Loading**
The project downloads or loads an image from a URL or local path using OpenCV's `imread()` function.

### 2. **Color Space Conversion**
- Converts BGR (OpenCV default) to RGB for proper display with matplotlib
- Converts to grayscale for improved OCR accuracy

### 3. **Text Extraction**
Uses Tesseract OCR's `image_to_string()` function to extract all readable text from the image.

### 4. **Text Localization**
- Uses `image_to_data()` to get detailed information about each detected text element
- Retrieves bounding box coordinates (x, y, width, height) for each text region
- Draws rectangles around detected text using OpenCV's `rectangle()` function

### 5. **Visualization**
Displays the processed image with bounding boxes using matplotlib for easy analysis.

## 📊 Results

The project successfully:
- ✅ Detects text in various fonts and sizes
- ✅ Extracts text content with high accuracy
- ✅ Localizes text regions with bounding boxes
- ✅ Handles different image formats and qualities
- ✅ Provides visual feedback for verification

### Sample Output

```
Extracted Text:
[The text content extracted from your image will appear here]
```

## 🚀 Future Enhancements

Potential improvements for this project:

- [ ] Add support for multiple languages
- [ ] Implement text detection for rotated/skewed images
- [ ] Add deep learning-based text detection (EAST, CRAFT)
- [ ] Create a web interface using Flask/Streamlit
- [ ] Add batch processing for multiple images
- [ ] Implement confidence scoring for detected text
- [ ] Add text preprocessing (noise removal, contrast enhancement)
- [ ] Support for handwritten text recognition
- [ ] Export results to JSON/CSV format
- [ ] Real-time text detection from video streams

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is part of the [100+ Machine Learning Projects](https://github.com/fatahrahimi330/100-Machine-Learning-Projects) collection.

## 👨‍💻 Author

**Fatah Rahimi**
- GitHub: [@fatahrahimi330](https://github.com/fatahrahimi330)

## 🙏 Acknowledgments

- OpenCV community for excellent computer vision tools
- Tesseract OCR team for the powerful text recognition engine
- The open-source community for continuous inspiration

## 📞 Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Refer to [Tesseract documentation](https://github.com/tesseract-ocr/tesseract)
- Refer to [OpenCV documentation](https://docs.opencv.org/)

---

⭐ **Star this repository if you found it helpful!**

Made with ❤️ as part of 100+ Machine Learning Projects
